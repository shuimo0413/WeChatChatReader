
"""
本文件是快速截图的脚本，原理是使用进程名搜索对应进程被操作系统分配的 PID 来精准定位进程，从而截图


Windows：枚举 WeChat/Weixin 进程与主窗口，按间隔截取客户区到 ../before_img，
供人工标注后走 split_dataset → YOLO 训练。DPI 感知 + PrintWindow 兜底避免全黑截屏。
"""
import argparse
import ctypes
import datetime
import logging
import os
import sys
import time
from ctypes import wintypes

from PIL import Image, ImageGrab

TH32CS_SNAPPROCESS = 0x00000002
PW_RENDERFULLCONTENT = 0x00000002
PW_CLIENTONLY = 0x00000001
DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = (
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", wintypes.WCHAR * 260),
    )


class BITMAP(ctypes.Structure):
    _fields_ = (
        ("bmType", ctypes.c_long),
        ("bmWidth", ctypes.c_long),
        ("bmHeight", ctypes.c_long),
        ("bmWidthBytes", ctypes.c_long),
        ("bmPlanes", wintypes.WORD),
        ("bmBitsPixel", wintypes.WORD),
        ("bmBits", ctypes.c_void_p),
    )


def _dpi_aware() -> None:
    user32 = ctypes.windll.user32
    fn = getattr(user32, "SetProcessDpiAwarenessContext", None)
    if fn:
        fn(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
    else:
        user32.SetProcessDPIAware()


def _pids_for_exe_basenames(names_lower: set[str]) -> set[int]:
    kernel32 = ctypes.windll.kernel32
    snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snap in (-1, 0xFFFFFFFF):
        return set()
    pe = PROCESSENTRY32W()
    pe.dwSize = ctypes.sizeof(PROCESSENTRY32W)
    out: set[int] = set()
    if kernel32.Process32FirstW(snap, ctypes.byref(pe)):
        while True:
            base = os.path.basename(pe.szExeFile).lower()
            if base in names_lower:
                out.add(pe.th32ProcessID)
            if not kernel32.Process32NextW(snap, ctypes.byref(pe)):
                break
    kernel32.CloseHandle(snap)
    return out


def _client_rect_screen(hwnd) -> tuple[int, int, int, int] | None:
    user32 = ctypes.windll.user32
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    tl = wintypes.POINT(0, 0)
    br = wintypes.POINT(rect.right, rect.bottom)
    if not user32.ClientToScreen(hwnd, ctypes.byref(tl)):
        return None
    if not user32.ClientToScreen(hwnd, ctypes.byref(br)):
        return None
    return tl.x, tl.y, br.x, br.y


def _pick_wechat_hwnd(target_pids: set[int]):
    user32 = ctypes.windll.user32
    acc = []

    def _callback(hwnd, _lp):
        if not user32.IsWindowVisible(hwnd):
            return True
        if user32.IsIconic(hwnd):
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value not in target_pids:
            return True
        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return True
        w = rect.right - rect.left
        h = rect.bottom - rect.top
        if w <= 0 or h <= 0:
            return True
        buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, buf, 256)
        acc.append((hwnd, w * h, buf.value))
        return True

    CMPFUNC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    cb = CMPFUNC(_callback)
    user32.EnumWindows(cb, 0)
    if not acc:
        return None
    for hwnd, area, cls in acc:
        if cls == "WeChatMainWndForPC":
            return hwnd
    return max(acc, key=lambda x: x[1])[0]


def _mostly_black(im: Image.Image | None, thresh: int = 8, frac: float = 0.98) -> bool:
    if im is None:
        return True
    g = im.convert("L")
    hist = g.histogram()
    dark = sum(hist[:thresh])
    return dark >= frac * (im.width * im.height)


def _capture_printwindow_bgrx(hwnd, w: int, h: int) -> Image.Image | None:
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    hdc = user32.GetWindowDC(hwnd)
    if not hdc:
        return None
    hdc_mem = gdi32.CreateCompatibleDC(hdc)
    hbmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
    old = gdi32.SelectObject(hdc_mem, hbmp)
    ok = user32.PrintWindow(hwnd, hdc_mem, PW_RENDERFULLCONTENT | PW_CLIENTONLY)
    if not ok:
        user32.PrintWindow(hwnd, hdc_mem, PW_RENDERFULLCONTENT)
    bmp = BITMAP()
    gdi32.GetObjectW(hbmp, ctypes.sizeof(BITMAP), ctypes.byref(bmp))
    bw = bmp.bmWidth
    bh = abs(bmp.bmHeight)
    nbytes = bmp.bmWidthBytes * bh
    buf = ctypes.create_string_buffer(nbytes)
    gdi32.GetBitmapBits(hbmp, nbytes, buf)
    gdi32.SelectObject(hdc_mem, old)
    gdi32.DeleteObject(hbmp)
    gdi32.DeleteDC(hdc_mem)
    user32.ReleaseDC(hwnd, hdc)
    try:
        return Image.frombytes("RGB", (bw, bh), buf.raw, "raw", "BGRX", bmp.bmWidthBytes, 1)
    except Exception:
        return None


def capture_wechat_client(hwnd) -> Image.Image | None:
    box = _client_rect_screen(hwnd)
    if not box:
        return None
    left, top, right, bottom = box
    w = right - left
    h = bottom - top
    if w <= 0 or h <= 0:
        return None
    im = ImageGrab.grab(bbox=(left, top, right, bottom))
    if _mostly_black(im):
        im2 = _capture_printwindow_bgrx(hwnd, w, h)
        if im2 is not None:
            im = im2
    return im


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    _dpi_aware()
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-ms", type=int, default=1000)
    parser.add_argument(
        "--exe",
        action="append",
        default=None,
        help="进程 exe 基名，可多次指定，默认 WeChat.exe Weixin.exe",
    )
    args = parser.parse_args()
    names = {x.lower() for x in args.exe} if args.exe else {"wechat.exe", "weixin.exe"}
    before_dir = os.path.normpath(os.path.join(here, "..", "before_img"))
    os.makedirs(before_dir, exist_ok=True)

    interval = max(1, args.interval_ms) / 1000.0

    while True:
        try:
            pids = _pids_for_exe_basenames(names)
            if not pids:
                logging.warning("未找到微信进程")
                time.sleep(interval)
                continue
            hwnd = _pick_wechat_hwnd(pids)
            if not hwnd:
                logging.warning("未找到微信主窗口")
                time.sleep(interval)
                continue
            im = capture_wechat_client(hwnd)
            if im is None:
                logging.warning("截屏失败")
                time.sleep(interval)
                continue
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            img_path = os.path.join(before_dir, f"wechat_cap_{ts}.png")
            im.save(img_path)
            logging.info("已保存 %s", img_path)
        except KeyboardInterrupt:
            raise
        except Exception:
            logging.exception("本轮截图异常")
        time.sleep(interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
