"""
本文件的作用也是数据标注，不过是半自动数据标注，启动后，依旧是读取before_img的文件，但是会使用训练好的模型自动进行标注
你可以对标注的图片进行修改
_DEFAULT_WEIGHTS_PARTS = ("runs", "detect", "wechat_bubbles-3", "weights", "best.pt")
这里是设置模型的路径

按键说明：
- q / Esc：退出并保存当前图
- s：保存当前图标签
- n：保存并下一张
- p：保存并上一张
- r：重新用模型推理当前图
- 直接右键框的边缘（有点难点） / del / backspace：删除数据标注出来的框
- [ / ]：切换选中框类别（上一类 / 下一类）
- 0~9：设置新画框类别
- 鼠标左键拖拽：新增框；左键单击框：选中；右键单击框：删除框

保存时不会保存重复的图片，如果你按了两次s，那么会在原来的标注上覆盖

"""


# -*- coding: utf-8 -*-
"""
三类 YOLO：chat_bubble / chat_object / distraction（与 data.yaml 一致）。
对图片目录推理，OpenCV 交互改框，导出 YOLO txt；保存时默认同步原图到 img/。
"""
from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

DEFAULT_NAMES: dict[int, str] = {
    0: "chat_bubble",
    1: "chat_object",
    2: "distraction",
}

_DEFAULT_WEIGHTS_PARTS = ("runs", "detect", "wechat_bubbles-3", "weights", "best.pt")


def imread(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is None:
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except OSError:
            return None
    return img


def load_class_names(data_yaml: Path | None) -> dict[int, str]:
    if data_yaml is None or not data_yaml.is_file():
        return dict(DEFAULT_NAMES)
    try:
        import yaml
    except ImportError:
        return dict(DEFAULT_NAMES)
    with data_yaml.open("r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    names = d.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    return dict(DEFAULT_NAMES)


def bgr_for_class(cls: int, names: dict[int, str]) -> tuple[int, int, int]:
    key = str(names.get(cls, "")).lower()
    if cls == 0 or key == "chat_bubble":
        return (80, 200, 80)
    if cls == 1 or key == "chat_object":
        return (0, 140, 255)
    if cls == 2 or key == "distraction":
        return (200, 120, 220)
    return (160, 160, 160)


def read_yolo_txt(path: Path, w: int, h: int) -> list[tuple[int, float, float, float, float]]:
    if not path.is_file():
        return []
    out: list[tuple[int, float, float, float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        p = line.split()
        if len(p) < 5:
            continue
        cls = int(p[0])
        cx, cy, bw, bh = map(float, p[1:5])
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        out.append((cls, x1, y1, x2, y2))
    return out


def write_yolo_txt(
    path: Path,
    w: int,
    h: int,
    boxes: list[tuple[int, float, float, float, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for cls, x1, y1, x2, y2 in boxes:
        x1c = max(0.0, min(float(w), x1))
        y1c = max(0.0, min(float(h), y1))
        x2c = max(0.0, min(float(w), x2))
        y2c = max(0.0, min(float(h), y2))
        if x2c - x1c < 1 or y2c - y1c < 1:
            continue
        bw = (x2c - x1c) / w
        bh = (y2c - y1c) / h
        cx = (x1c + x2c) / 2 / w
        cy = (y1c + y2c) / 2 / h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def predict_boxes(
    model: YOLO,
    img_path: Path,
    imgsz: int,
    conf: float,
    device: str,
) -> list[tuple[int, float, float, float, float]]:
    res = model.predict(
        source=str(img_path),
        imgsz=imgsz,
        conf=conf,
        device=device,
        verbose=False,
    )
    r = res[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    out: list[tuple[int, float, float, float, float]] = []
    xyxy = r.boxes.xyxy.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy().astype(int)
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = map(float, xyxy[i])
        out.append((int(clss[i]), x1, y1, x2, y2))
    return out


def display_scale(w: int, h: int, max_side: int) -> float:
    m = max(w, h)
    if m <= max_side:
        return 1.0
    return max_side / m


def to_disp(x: float, y: float, scale: float) -> tuple[int, int]:
    return int(x * scale), int(y * scale)


def to_img(xd: int, yd: int, scale: float) -> tuple[float, float]:
    return xd / scale, yd / scale


def pick_box_at(
    boxes: list[tuple[int, float, float, float, float]],
    ix: float,
    iy: float,
) -> int:
    candidates: list[tuple[float, int]] = []
    for i, (_c, x1, y1, x2, y2) in enumerate(boxes):
        if x1 <= ix <= x2 and y1 <= iy <= y2:
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            candidates.append((area, i))
    if not candidates:
        return -1
    candidates.sort(key=lambda t: t[0])
    return candidates[0][1]


@dataclass
class App:
    model: YOLO
    images: list[Path]
    labels_dir: Path
    img_mirror_dir: Path | None
    names: dict[int, str]
    imgsz: int
    conf: float
    device: str
    max_side: int
    idx: int = 0
    boxes: list[tuple[int, float, float, float, float]] = field(default_factory=list)
    selected: int = -1
    cur_cls: int = 0
    drag_start: tuple[float, float] | None = None
    drag_cur: tuple[float, float] | None = None
    img: np.ndarray | None = None
    scale_disp: float = 1.0
    win: str = "label_review"

    def num_classes(self) -> int:
        if not self.names:
            return len(DEFAULT_NAMES)
        return max(self.names.keys()) + 1

    def label_path(self) -> Path:
        return self.labels_dir / f"{self.images[self.idx].stem}.txt"

    def load_boxes_for_image(self, prefer_file: bool) -> None:
        assert self.img is not None
        h, w = self.img.shape[:2]
        lp = self.label_path()
        if prefer_file and lp.is_file():
            self.boxes = read_yolo_txt(lp, w, h)
        else:
            self.boxes = predict_boxes(self.model, self.images[self.idx], self.imgsz, self.conf, self.device)
        self.selected = -1
        self.drag_start = None
        self.drag_cur = None

    def load_image_index(self, i: int, prefer_file: bool) -> None:
        self.idx = max(0, min(i, len(self.images) - 1))
        p = self.images[self.idx]
        self.img = imread(p)
        if self.img is None:
            self.boxes = []
            self.selected = -1
            return
        self.load_boxes_for_image(prefer_file)

    def save_current(self) -> None:
        if self.img is None:
            logging.warning("保存跳过：当前无图像")
            return
        h, w = self.img.shape[:2]
        out = self.label_path()
        write_yolo_txt(out, w, h, self.boxes)
        logging.info("已写入 %s | 框数量=%d", out.resolve(), len(self.boxes))
        if self.img_mirror_dir is not None:
            src = self.images[self.idx]
            self.img_mirror_dir.mkdir(parents=True, exist_ok=True)
            dest = self.img_mirror_dir / src.name
            shutil.copy2(src, dest)
            logging.info("已同步图片 %s", dest.resolve())

    def render(self) -> np.ndarray:
        assert self.img is not None
        canvas = self.img.copy()
        h, w = canvas.shape[:2]
        for i, (cls, x1, y1, x2, y2) in enumerate(self.boxes):
            bc = bgr_for_class(cls, self.names)
            c = (0, 255, 255) if i == self.selected else bc
            thick = 4 if i == self.selected else 2
            x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
            cv2.rectangle(canvas, (x1i, y1i), (x2i, y2i), c, thick)
            name = self.names.get(cls, str(cls))
            cv2.putText(
                canvas,
                f"{cls}:{name}",
                (x1i, max(12, y1i - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                c,
                1,
                cv2.LINE_AA,
            )
        if self.drag_start and self.drag_cur:
            x1, y1 = map(int, self.drag_start)
            x2, y2 = map(int, self.drag_cur)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 128, 0), 1)
        ncls = self.num_classes()
        cur_name = self.names.get(self.cur_cls, str(self.cur_cls))
        bar = f"{self.images[self.idx].name}  |  {self.idx+1}/{len(self.images)}  |  new_cls={self.cur_cls} {cur_name}"
        cv2.rectangle(canvas, (0, h - 28), (w, h), (40, 40, 40), -1)
        cv2.putText(canvas, bar, (6, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        help1 = "L-drag L-click R-click Del [/] 0-2 cls  n/p s r q"
        cv2.putText(canvas, help1, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        legend = " ".join(
            f"{i}:{self.names.get(i, str(i))}" for i in range(min(ncls, 10))
        )
        cv2.putText(canvas, legend[: min(len(legend), 120)], (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 200, 220), 1, cv2.LINE_AA)
        sc = display_scale(w, h, self.max_side)
        self.scale_disp = sc
        if sc < 1.0:
            canvas = cv2.resize(canvas, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
        return canvas

    def on_mouse(self, event: int, xd: int, yd: int, _flags: int, _param: object) -> None:
        if self.img is None:
            return
        h, w = self.img.shape[:2]
        sc = self.scale_disp
        ix, iy = to_img(xd, yd, sc)

        if event == cv2.EVENT_LBUTTONDOWN:
            bi = pick_box_at(self.boxes, ix, iy)
            if bi >= 0:
                self.selected = bi
                self.drag_start = None
                self.drag_cur = None
            else:
                self.selected = -1
                self.drag_start = (ix, iy)
                self.drag_cur = (ix, iy)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_cur = (ix, iy)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            x1, y1 = self.drag_start
            x2, y2 = self.drag_cur if self.drag_cur else self.drag_start
            self.drag_start = None
            self.drag_cur = None
            rx1, rx2 = sorted((x1, x2))
            ry1, ry2 = sorted((y1, y2))
            if rx2 - rx1 >= 5 and ry2 - ry1 >= 5:
                rx1 = max(0.0, min(float(w), rx1))
                rx2 = max(0.0, min(float(w), rx2))
                ry1 = max(0.0, min(float(h), ry1))
                ry2 = max(0.0, min(float(h), ry2))
                self.boxes.append((self.cur_cls, rx1, ry1, rx2, ry2))
                self.selected = len(self.boxes) - 1
        elif event == cv2.EVENT_RBUTTONDOWN:
            bi = pick_box_at(self.boxes, ix, iy)
            if bi >= 0:
                del self.boxes[bi]
                self.selected = -1

    def run(self) -> None:
        if not self.images:
            raise SystemExit("没有图片")
        cv2.namedWindow(self.win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.win, self.on_mouse)

        self.load_image_index(0, prefer_file=True)
        while True:
            if self.img is None:
                frame = np.zeros((240, 640, 3), dtype=np.uint8)
                cv2.putText(
                    frame,
                    f"read fail: {self.images[self.idx]}",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                )
            else:
                frame = self.render()
            cv2.imshow(self.win, frame)
            key = cv2.waitKey(16) & 0xFF
            if key in (ord("q"), 27):
                kname = "Esc" if key == 27 else "q"
                logging.info("按键 %s：退出并保存当前图", kname)
                self.save_current()
                break
            if key == ord("s"):
                logging.info("按键 s：保存当前图标签")
                self.save_current()
            elif key == ord("n"):
                logging.info("按键 n：保存并下一张")
                self.save_current()
                self.load_image_index(self.idx + 1, prefer_file=True)
            elif key == ord("p"):
                logging.info("按键 p：保存并上一张")
                self.save_current()
                self.load_image_index(self.idx - 1, prefer_file=True)
            elif key == ord("r"):
                if self.img is None:
                    logging.warning("按键 r：当前无图像，已忽略")
                else:
                    cur = self.images[self.idx]
                    logging.info("按键 r：重新推理 | %s", cur.name)
                    self.boxes = predict_boxes(
                        self.model, cur, self.imgsz, self.conf, self.device
                    )
                    self.selected = -1
                    logging.info("推理结束 | 框数量=%d（需 s 或切图才会写入磁盘）", len(self.boxes))
            elif key in (8, 127):
                if self.selected >= 0:
                    logging.info("按键 Del/Backspace：删除选中框 index=%d", self.selected)
                    del self.boxes[self.selected]
                    self.selected = -1
            elif key == ord("["):
                if self.selected >= 0:
                    c, x1, y1, x2, y2 = self.boxes[self.selected]
                    n = self.num_classes()
                    nc = (c - 1) % n
                    self.boxes[self.selected] = (nc, x1, y1, x2, y2)
                    logging.info("按键 [：选中框类别 %d -> %d", c, nc)
            elif key == ord("]"):
                if self.selected >= 0:
                    c, x1, y1, x2, y2 = self.boxes[self.selected]
                    n = self.num_classes()
                    nc = (c + 1) % n
                    self.boxes[self.selected] = (nc, x1, y1, x2, y2)
                    logging.info("按键 ]：选中框类别 %d -> %d", c, nc)
            elif ord("0") <= key <= ord("9"):
                d = key - ord("0")
                if d < self.num_classes():
                    self.cur_cls = d
                    logging.info("按键 %c：新框类别设为 %d", key, self.cur_cls)

        cv2.destroyAllWindows()


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="YOLO 预测 + 交互改框 + 导出 txt")
    ap.add_argument(
        "--weights",
        type=Path,
        default=root.joinpath(*_DEFAULT_WEIGHTS_PARTS),
        help="best.pt；默认见本文件 _DEFAULT_WEIGHTS_PARTS，可用命令行覆盖",
    )
    ap.add_argument(
        "--images",
        type=Path,
        default=root / "before_img",
        help="图片目录，默认项目根下 before_img",
    )
    ap.add_argument(
        "--labels-out",
        type=Path,
        default=root / "labels",
        help="YOLO 标签输出目录，默认项目根下 labels",
    )
    ap.add_argument(
        "--img-mirror",
        type=Path,
        default=root / "img",
        help="保存标签时同步复制原图到此目录，默认项目根下 img（对接 split_dataset --img-dir）",
    )
    ap.add_argument(
        "--no-img-mirror",
        action="store_true",
        help="不向 img-mirror 复制图片",
    )
    ap.add_argument("--data", type=Path, default=root / "data.yaml", help="读取 names 显示用")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--max-side", type=int, default=1280, help="窗口最大边，大图缩小显示")
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="终端日志级别",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [label_review] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    weights_path = args.weights.resolve()
    if not weights_path.is_file():
        raise SystemExit(
            f"找不到权重: {weights_path}\n请指定: --weights 路径\\to\\best.pt"
        )

    img_dir = args.images.resolve()
    if not img_dir.is_dir():
        raise SystemExit(f"不是目录: {img_dir}")

    images = sorted(
        [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT],
        key=lambda p: p.name.lower(),
    )
    names = load_class_names(args.data.resolve() if args.data else None)
    model = YOLO(str(weights_path))
    mirror = None if args.no_img_mirror else args.img_mirror.resolve()
    app = App(
        model=model,
        images=images,
        labels_dir=args.labels_out.resolve(),
        img_mirror_dir=mirror,
        names=names,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        max_side=args.max_side,
    )
    app.run()


if __name__ == "__main__":
    main()
