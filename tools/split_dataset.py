"""
这个文件用于将已标注的数据按比例拆分为训练集和验证集。

输入是项目根目录下的 img 和 labels（按同名 stem 配对），

输出到 dataset/images/{train,val} 与 dataset/labels/{train,val}，
用于后续 train_bubbles.py 训练时由 data.yaml 引用。
"""
import argparse
import os
import random
import shutil
from pathlib import Path

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _stem_paths(img_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in img_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            out[p.stem] = p
    return out


def split_dataset(
    img_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    seed: int | None = 42,
) -> None:
    images = _stem_paths(img_dir)
    if not images:
        raise SystemExit(f"未在 {img_dir} 找到图片")

    paired: list[str] = []
    for stem, img_path in images.items():
        label_path = labels_dir / f"{stem}.txt"
        if label_path.is_file():
            paired.append(stem)

    if not paired:
        raise SystemExit(
            f"没有同时存在图片与标签的样本：检查 {img_dir} 与 {labels_dir} 文件名是否一致"
        )

    if seed is not None:
        random.seed(seed)
    random.shuffle(paired)

    # 前 train_ratio 为训练集，余下为验证集（与 int 截断一致，可能 val 多 1 个样本）
    split_point = int(len(paired) * train_ratio)
    train_stems = paired[:split_point]
    val_stems = paired[split_point:]

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    def copy_split(stems: list[str], split: str) -> None:
        for stem in stems:
            src_img = images[stem]
            shutil.copy2(src_img, output_dir / "images" / split / src_img.name)
            shutil.copy2(
                labels_dir / f"{stem}.txt",
                output_dir / "labels" / split / f"{stem}.txt",
            )

    copy_split(train_stems, "train")
    copy_split(val_stems, "val")

    print(f"训练集: {len(train_stems)} 验证集: {len(val_stems)} 输出: {output_dir}")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", type=Path, default=root / "img")
    ap.add_argument("--labels-dir", type=Path, default=root / "labels")
    ap.add_argument("--output-dir", type=Path, default=root / "dataset")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    split_dataset(
        args.img_dir.resolve(),
        args.labels_dir.resolve(),
        args.output_dir.resolve(),
        train_ratio=args.train_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
