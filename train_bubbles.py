"""
这个文件是用来训练模型的，你可以使用之前的模型进行二次训练
本文件会读取 img 和 labels 文件夹，读取这两个文件夹的数据标注，然后进行训练


微信界面 YOLO 检测：聊天气泡、聊天对象、干扰项。基于 Ultralytics 在本地标注数据上训练，可选验证集可视化推理。
data.yaml 指向 YOLO 格式目录（images/train|val + labels/train|val）。
"""
import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset


def main() -> None:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="训练 YOLO：聊天气泡 / 聊天对象 / 干扰项")
    ap.add_argument("--data", type=Path, default=root / "data.yaml")
    ap.add_argument("--model", type=str, default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader 进程数；Windows 低内存建议 0，避免多进程重复加载 CUDA 动态库",
    )
    ap.add_argument(
        "--half",
        action="store_true",
        help="FP16 训练，降低显存占用",
    )
    ap.add_argument(
        "--cache",
        type=str,
        default="false",
        choices=("false", "ram", "disk"),
        help="数据集缓存：false 最省内存，disk 次之，ram 最快但占内存",
    )
    ap.add_argument(
        "--no-plots",
        action="store_true",
        help="关闭训练过程绘图，降低内存与磁盘占用",
    )
    ap.add_argument(
        "--save-val-vis",
        action="store_true",
        help="训练结束后用 best.pt 对验证集整目录推理并保存带检测框的图片（写入本次 run 目录下的 val_vis）",
    )
    ap.add_argument("--project", type=Path, default=root / "runs" / "detect")
    ap.add_argument("--name", type=str, default="wechat_bubbles")
    args = ap.parse_args()

    data_yaml = args.data.resolve()
    if not data_yaml.is_file():
        raise SystemExit(f"找不到 data 配置: {data_yaml}")

    # ultralytics：False 不缓存；True 整集进内存；disk 落盘索引以省 RAM
    cache_val: bool | str
    if args.cache == "false":
        cache_val = False
    elif args.cache == "ram":
        cache_val = True
    else:
        cache_val = "disk"

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        half=args.half,
        cache=cache_val,
        plots=not args.no_plots,
        project=str(args.project.resolve()),
        name=args.name,
    )

    if args.save_val_vis:
        # 与 train 的 project/name 一致；若同名 run 已存在，ultralytics 可能加后缀，此处需与实际目录一致
        run_dir = args.project.resolve() / args.name
        best = run_dir / "weights" / "best.pt"
        if not best.is_file():
            raise SystemExit(f"未找到权重，无法导出验证集可视化: {best}")
        data = check_det_dataset(str(data_yaml))
        val_src = data["val"]
        vis_model = YOLO(str(best))
        vis_model.predict(
            source=val_src,
            imgsz=args.imgsz,
            device=args.device,
            half=args.half,
            workers=args.workers,
            save=True,
            project=str(run_dir),
            name="val_vis",
            exist_ok=True,
            verbose=False,
        )


if __name__ == "__main__":
    main()
