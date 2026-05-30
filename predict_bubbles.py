"""
这个文件是用来使用训练好的模型检测聊天气泡的位置的，会读取before_img里面的文件
检测后检测的结果保存在 pre_dict_out 文件夹


uv run --active python D:\java_object\new_model\predict_bubbles.py --weights D:\java_object\new_model\runs\detect\wechat_bubbles-3\weights\best.pt --source D:\java_object\new_model\test --device 0

"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from bubble_role import classify_role_bgr

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

CLASS_COLORS_BGR = {
    "self": (0, 200, 0),
    "peer": (0, 140, 255),
    "chat_object": (160, 160, 160),
    "distraction": (200, 120, 220),
}


def bgr_for_detection(class_id: int, label: str, bubble_key: int) -> tuple[int, int, int] | None:
    if class_id == bubble_key:
        return None
    key = str(label).lower()
    if key == "distraction":
        return CLASS_COLORS_BGR["distraction"]
    if key == "chat_object":
        return CLASS_COLORS_BGR["chat_object"]
    if key == "others":
        return CLASS_COLORS_BGR["chat_object"]
    return CLASS_COLORS_BGR["chat_object"]


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix.lower() in IMG_EXT else []
    return sorted(
        p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT
    )


def detect_vertical_chat_boundaries(im_bgr) -> tuple[int, int]:
    # 用 Sobel Y 找“水平边缘”最强的行，近似当作聊天区域的上/下边界
    h = int(im_bgr.shape[0])
    if h <= 1:
        return 0, 0
    gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    row_energy = np.mean(np.abs(sobel_y), axis=1).astype(np.float32)
    # 无有效边缘时，退化成整张图都可视，避免误杀
    if row_energy.size != h or float(np.max(row_energy)) <= 1e-6:
        return 0, h - 1
    # 上边界只在上 1/3 搜索，下边界只在下 1/3 搜索，减少中间内容干扰
    top_limit = max(1, h // 3)
    bottom_start = min(h - 1, max(0, (2 * h) // 3))
    top_y = int(np.argmax(row_energy[:top_limit]))
    bottom_y = int(bottom_start + np.argmax(row_energy[bottom_start:]))
    if bottom_y <= top_y:
        return 0, h - 1
    return top_y, bottom_y


def calc_vertical_overlap_ratio(
    box_xyxy, top_boundary_y: int, bottom_boundary_y: int
) -> float:
    # 计算气泡框有多少高度落在可视聊天区域之外（上溢出 + 下溢出）
    y1 = float(box_xyxy[1])
    y2 = float(box_xyxy[3])
    bubble_h = max(1.0, y2 - y1)
    overflow_top = max(0.0, float(top_boundary_y) - y1)
    overflow_bottom = max(0.0, y2 - float(bottom_boundary_y))
    ratio = (overflow_top + overflow_bottom) / bubble_h
    return float(min(1.0, max(0.0, ratio)))


def main() -> None:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--weights",
        type=Path,
        # 指定你的模型pt文件
        default = root / "runs" / "detect" / "wechat_bubbles-3" / "weights" / "best.pt",
        # default=root / "runs" / "detect" / "test" / "weights" / "best.pt",
    )
    ap.add_argument("--source", type=Path, default=root / "test")  # 默认识别test文件夹
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--output", type=Path, default=root / "predict_out")
    ap.add_argument("--bubble-class", type=int, default=0)
    ap.add_argument("--min-green-ratio", type=float, default=0.08)
    ap.add_argument("--shrink-ratio", type=float, default=0.12)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--edge-overlap-threshold", type=float, default=0.30)
    ap.add_argument("--name-pad-x", type=int, default=12)
    ap.add_argument("--name-pad-y", type=int, default=6)
    args = ap.parse_args()

    weights = args.weights.resolve()
    if not weights.is_file():
        raise SystemExit(f"找不到权重: {weights}")

    model = YOLO(str(weights))
    names = model.names
    bubble_key = args.bubble_class
    args.output.mkdir(parents=True, exist_ok=True)

    images = collect_images(args.source.resolve())
    if not images:
        raise SystemExit(f"未找到图片: {args.source}")

    all_records: list[dict] = []

    for img_path in images:
        im_bgr = cv2.imread(str(img_path))
        if im_bgr is None:
            continue

        results = model.predict(
            im_bgr,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )
        record: dict = {"image": str(img_path), "detections": []}
        vis = im_bgr.copy()
        # 每张图只算一次上下边界，后续所有气泡复用
        top_boundary_y, bottom_boundary_y = detect_vertical_chat_boundaries(im_bgr)

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            for i in range(len(boxes)):
                c = int(cls[i])
                label = str(names[c])
                box = xyxy[i]
                cf = float(confs[i])

                if c == bubble_key:
                    # 半截气泡过滤：与上下边界外区域重叠比例超过阈值就跳过
                    overlap_ratio = calc_vertical_overlap_ratio(
                        box, top_boundary_y, bottom_boundary_y
                    )
                    if overlap_ratio > args.edge_overlap_threshold:
                        continue
                    role, gr = classify_role_bgr(
                        im_bgr,
                        box,
                        shrink_ratio=args.shrink_ratio,
                        min_green_ratio=args.min_green_ratio,
                    )
                    color = CLASS_COLORS_BGR[role]
                    tag = f"{label} {role} {gr:.2f}"
                else:
                    role, gr = None, 0.0
                    color = bgr_for_detection(c, label, bubble_key)
                    assert color is not None
                    tag = f"{label} {cf:.2f}"

                draw_box = box.copy()
                if c == 0:
                    h, w = im_bgr.shape[:2]
                    draw_box[0] = max(0.0, draw_box[0] - float(args.name_pad_x))
                    draw_box[1] = max(0.0, draw_box[1] - float(args.name_pad_y))
                    draw_box[2] = min(float(w - 1), draw_box[2] + float(args.name_pad_x))
                    draw_box[3] = min(float(h - 1), draw_box[3] + float(args.name_pad_y))

                record["detections"].append(
                    {
                        "xyxy": [float(x) for x in draw_box],
                        "class_id": c,
                        "class_name": label,
                        "conf": cf,
                        "role": role,
                        "green_ratio": gr,
                    }
                )

                x1, y1, x2, y2 = map(int, draw_box)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    vis,
                    tag,
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        out_img = args.output / f"{img_path.stem}_vis.jpg"
        cv2.imwrite(str(out_img), vis)
        out_json = args.output / f"{img_path.stem}.json"
        out_json.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        all_records.append(record)

    summary = args.output / "summary.json"
    summary.write_text(json.dumps(all_records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
