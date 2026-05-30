
<div align = "center">
<h1 align="center">
微信气泡检测文字提取 · WeChat Bubble YOLO
</h1>


[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![uv](https://img.shields.io/badge/uv-package_manager-DE5FE9?style=flat-square)](https://github.com/astral-sh/uv)
[![Platform](https://img.shields.io/badge/Platform-Windows-0078D6?style=flat-square&logo=windows&logoColor=white)](https://www.microsoft.com/windows)
[![QQ](https://img.shields.io/badge/QQ-1743355601-12B7F5?style=flat-square)](https://im.qq.com/)

</div>

#### 有问题可以加我qq:`174335601`
### 环境

```bash
uv sync
```

文本提取需额外安装 OCR（按需）：

```bash
uv pip install rapidocr-onnxruntime onnxruntime
# GPU：uv pip install onnxruntime-gpu
```

### 0 · 截图

```bash
uv run python tools/wechat_window_loop_ocr.py --interval-ms 1000
```

### 1 · 标注（二选一）

**网页标注** — 启动后打开 `http://127.0.0.1:8810/`

```bash
uv run python tools/fount/server.py
```

**半自动标注**

```bash
uv run python tools/label_review.py --weights runs/detect/wechat_bubbles-7/weights/best.pt
```

### 2 · 划分训练 / 验证集

```bash
uv run python tools/split_dataset.py --train-ratio 0.8
```

### 3 · 训练（或二次训练）

```bash
uv run python train_bubbles.py --model runs/detect/wechat_bubbles-7/weights/best.pt
```

不指定 `--model` 则从头训练新模型。

### 4 · 推理

```bash
uv run python predict_bubbles.py --weights runs/detect/wechat_bubbles-7/weights/best.pt --source test
```

### 5 · 文本提取

```bash
uv run python text_process.py
```

---

## 输出目录

| 目录 | 说明 |
|:---|:---|
| `before_img/` | 截图原图 |
| `img/`、`labels/` | 标注图片与 YOLO 标签 |
| `dataset/` | 训练 / 验证集 |
| `runs/detect/.../weights/` | 模型权重 |
| `predict_out/` | 推理结果 |
| `text_json/`、`text_img/` | OCR 文本输出 |

---

## 技术栈

| 类别 | 技术 |
|:---|:---|
| 语言 / 环境 | Python 3.11+、uv |
| 深度学习 | PyTorch、TorchVision、Ultralytics YOLOv8 |
| 图像 | OpenCV、Pillow、NumPy |
| OCR | RapidOCR + ONNX Runtime |
| 标注前端 | HTML / CSS / Canvas + `http.server` |
| 截图 | Win32 API（ctypes）+ PIL |
