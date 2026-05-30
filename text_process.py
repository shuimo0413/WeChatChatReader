from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

"""
这个文件是用来提取聊天信息的

本文件会读取predict_out文件夹里面的img和json，截取出每个聊天气泡

结果保存在text_img,text_json
"""


try:
    from rapidocr_onnxruntime import RapidOCR
    RAPIDOCR_AVAILABLE = True
except Exception:
    RapidOCR = Any  # type: ignore
    RAPIDOCR_AVAILABLE = False

from bubble_role import classify_role_bgr

BUBBLE_CLASS_NAMES = frozenset({"chat_bubble", "bubble"})
CHAT_OBJECT_CLASS_NAMES = frozenset({"chat_object", "name"})


def _class_key(det: dict[str, Any]) -> str:
    return str(det.get("class_name", "")).strip().lower()


def is_bubble_det(det: dict[str, Any]) -> bool:
    return _class_key(det) in BUBBLE_CLASS_NAMES


def is_chat_object_det(det: dict[str, Any]) -> bool:
    return _class_key(det) in CHAT_OBJECT_CLASS_NAMES


def setup_windows_cuda_dll_paths() -> None:
    if os.name != "nt":
        return
    site_dir = Path(sys.prefix) / "Lib" / "site-packages"
    nvidia_dir = site_dir / "nvidia"
    candidate_dirs: list[Path] = []
    if nvidia_dir.is_dir():
        candidate_dirs.extend(p for p in nvidia_dir.glob("*/bin") if p.is_dir())
    candidate_dirs.extend(
        [
            site_dir / "nvidia" / "cudnn" / "bin",
            site_dir / "nvidia" / "cublas" / "bin",
            site_dir / "nvidia" / "cuda_runtime" / "bin",
            site_dir / "nvidia" / "cufft" / "bin",
        ]
    )
    existed = [str(p) for p in candidate_dirs if p.is_dir()]
    if not existed:
        return
    current_path = os.environ.get("PATH", "")
    missing = [p for p in existed if p.lower() not in current_path.lower()]
    if missing:
        os.environ["PATH"] = ";".join(missing + [current_path]) if current_path else ";".join(missing)
    add_dll_dir = getattr(os, "add_dll_directory", None)
    if add_dll_dir is not None:
        for p in existed:
            try:
                add_dll_dir(p)
            except OSError:
                pass


def imread(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is not None:
        return img
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError:
        return None


def imwrite(path: Path, image: np.ndarray) -> bool:
    ok = cv2.imwrite(str(path), image)
    if ok:
        return True
    ext = path.suffix or ".png"
    ok_enc, buf = cv2.imencode(ext, image)
    if not ok_enc:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        buf.tofile(str(path))
        return True
    except OSError:
        return False


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clamp_box(xyxy: list[float], w: int, h: int) -> tuple[int, int, int, int] | None:
    if len(xyxy) != 4:
        return None
    x1, y1, x2, y2 = xyxy
    ix1 = max(0, min(w, int(round(min(x1, x2)))))
    iy1 = max(0, min(h, int(round(min(y1, y2)))))
    ix2 = max(0, min(w, int(round(max(x1, x2)))))
    iy2 = max(0, min(h, int(round(max(y1, y2)))))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1, iy1, ix2, iy2


def pick_source_image(record: dict[str, Any], json_path: Path, predict_dir: Path) -> Path | None:
    raw = record.get("image")
    if isinstance(raw, str) and raw.strip():
        p = Path(raw)
        if p.is_file():
            return p
    fallback = predict_dir / f"{json_path.stem}_vis.jpg"
    if fallback.is_file():
        return fallback
    return None


def extract_text_with_ocr(ocr: RapidOCR | None, image_bgr: np.ndarray) -> str:
    if ocr is None:
        return ""
    result, _ = ocr(image_bgr)
    if not result:
        return ""
    lines: list[str] = []
    for item in result:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        text = str(item[1]).strip()
        score = item[2]
        try:
            conf = float(score)
        except Exception:
            conf = 0.0
        if conf >= 0.1 and text:
            lines.append(text)
    return "\n".join(lines)


def build_ocr(device: str) -> RapidOCR | None:
    if not RAPIDOCR_AVAILABLE:
        return None
    use_cuda = device.lower().strip() == "gpu"
    try:
        return RapidOCR(
            det_use_cuda=use_cuda,
            cls_use_cuda=use_cuda,
            rec_use_cuda=use_cuda,
        )
    except Exception:
        return None


def resolve_speaker(det: dict[str, Any], full_img: np.ndarray, xyxy: list[float]) -> str:
    role = det.get("role")
    if role == "self":
        return "self"
    if role == "peer":
        return "peer"
    calc_role, _ = classify_role_bgr(full_img, xyxy)
    return "self" if calc_role == "self" else "peer"


def extract_chat_object(
    detections: list[Any],
    full_img: np.ndarray,
    w: int,
    h: int,
    ocr: RapidOCR | None,
) -> str:
    candidates: list[dict[str, Any]] = []
    for det in detections:
        if not isinstance(det, dict):
            continue
        if not is_chat_object_det(det):
            continue
        xyxy = det.get("xyxy")
        if not isinstance(xyxy, list):
            continue
        box = clamp_box([float(v) for v in xyxy[:4]], w, h)
        if box is None:
            continue
        conf_raw = det.get("conf", 0.0)
        try:
            conf = float(conf_raw)
        except Exception:
            conf = 0.0
        candidates.append({"box": box, "conf": conf})
    if not candidates:
        return "对方"
    candidates.sort(key=lambda x: x["conf"], reverse=True)
    x1, y1, x2, y2 = candidates[0]["box"]
    crop = full_img[y1:y2, x1:x2]
    if crop.size == 0:
        return "对方"
    text = extract_text_with_ocr(ocr, crop).strip()
    return text or "对方"


def process_one(
    json_path: Path,
    predict_dir: Path,
    img_out_dir: Path,
    json_out_dir: Path,
    ocr: RapidOCR | None,
    save_crop: bool,
) -> dict[str, Any] | None:
    record = load_json(json_path)
    src_img_path = pick_source_image(record, json_path, predict_dir)
    if src_img_path is None:
        return None

    full_img = imread(src_img_path)
    if full_img is None:
        return None
    h, w = full_img.shape[:2]

    detections = record.get("detections")
    if not isinstance(detections, list):
        detections = []
    chat_object = extract_chat_object(detections, full_img, w, h, ocr)

    bubbles: list[dict[str, Any]] = []
    for det in detections:
        if not isinstance(det, dict):
            continue
        if not is_bubble_det(det):
            continue
        xyxy = det.get("xyxy")
        if not isinstance(xyxy, list):
            continue
        box = clamp_box([float(v) for v in xyxy[:4]], w, h)
        if box is None:
            continue
        bubbles.append({"det": det, "box": box})

    bubbles.sort(key=lambda x: (x["box"][1], x["box"][0]))

    per_image_crop_dir = img_out_dir / json_path.stem
    if save_crop:
        per_image_crop_dir.mkdir(parents=True, exist_ok=True)

    messages: list[dict[str, Any]] = []
    for idx, item in enumerate(bubbles, start=1):
        x1, y1, x2, y2 = item["box"]
        crop = full_img[y1:y2, x1:x2]
        if crop.size == 0:
            text = ""
        else:
            text = extract_text_with_ocr(ocr, crop)
            if save_crop:
                crop_name = f"msg_{idx:04d}.png"
                imwrite(per_image_crop_dir / crop_name, crop)
        det = item["det"]
        xyxy_raw = det.get("xyxy", [x1, y1, x2, y2])
        speaker = resolve_speaker(det, full_img, [float(v) for v in xyxy_raw[:4]])
        speaker_name = "self" if speaker == "self" else chat_object
        messages.append(
            {
                "index": idx,
                "bbox": [x1, y1, x2, y2],
                "speaker": speaker_name,
                "text": text,
            }
        )
    out = {
        "source_image": str(src_img_path),
        "source_detection_json": str(json_path),
        "chat_object": chat_object,
        "messages": messages,
    }
    out_path = json_out_dir / f"{json_path.stem}.json"
    json_out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    setup_windows_cuda_dll_paths()
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict-out", type=Path, default=root / "predict_out")
    ap.add_argument("--text-img", type=Path, default=root / "text_img")
    ap.add_argument("--text-json", type=Path, default=root / "text_json")
    ap.add_argument("--save-crop", action="store_true", default=True)
    ap.add_argument("--no-save-crop", action="store_true")
    ap.add_argument("--ocr-device", type=str, choices=("cpu", "gpu"), default="gpu")
    ap.add_argument("--sleep-ms", type=int, default=0)
    args = ap.parse_args()

    predict_dir = args.predict_out.resolve()
    if not predict_dir.is_dir():
        raise SystemExit(f"目录不存在: {predict_dir}")

    save_crop = args.save_crop and (not args.no_save_crop)
    img_out_dir = args.text_img.resolve()
    json_out_dir = args.text_json.resolve()
    json_out_dir.mkdir(parents=True, exist_ok=True)
    if save_crop:
        img_out_dir.mkdir(parents=True, exist_ok=True)

    ocr = build_ocr(args.ocr_device)
    if ocr is None:
        raise SystemExit(
            "RapidOCR 初始化失败。GPU 请安装 onnxruntime-gpu；CPU 请安装 onnxruntime。"
        )

    json_files = sorted(
        p for p in predict_dir.glob("*.json") if p.name.lower() != "summary.json"
    )

    all_outputs: list[dict[str, Any]] = []
    for json_path in json_files:
        one = process_one(
            json_path=json_path,
            predict_dir=predict_dir,
            img_out_dir=img_out_dir,
            json_out_dir=json_out_dir,
            ocr=ocr,
            save_crop=save_crop,
        )
        if one is not None:
            all_outputs.append(one)
        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    summary_path = json_out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(all_outputs, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
