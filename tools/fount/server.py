"""
这是数据标注的前端，本文件读取before_img文件夹，读取后你可以对数据进行手动标注

标注后的数据
图片被存储在 img 文件夹
标注等相关信息存储在 labels 文件夹
"""
import json
import os
import shutil
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent.parent
FOUNT = Path(__file__).resolve().parent
BEFORE = ROOT / "before_img"
IMG = ROOT / "img"
LABELS = ROOT / "labels"
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _ensure_dirs() -> None:
    BEFORE.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    LABELS.mkdir(parents=True, exist_ok=True)


def _safe_image_name(name: str) -> str | None:
    base = Path(name).name
    if not base or base != name.strip("/\\"):
        return None
    if Path(base).suffix.lower() not in IMG_EXT:
        return None
    return base


class AnnotateHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FOUNT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/list-before":
            self._list_before()
            return
        if path == "/api/image/before":
            qs = parse_qs(parsed.query)
            raw = (qs.get("name") or [None])[0]
            if raw:
                self._serve_before(unquote(raw))
            else:
                self.send_error(400, "missing name")
            return

        if path in ("/", ""):
            path = "/index.html"

        rel = path.lstrip("/").replace("\\", "/")
        if ".." in rel.split("/"):
            self.send_error(403)
            return
        target = FOUNT / rel
        if not target.is_file():
            self.send_error(404)
            return
        self._send_path(target)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/save":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(400, "invalid json")
            return
        fn = payload.get("filename")
        labels = payload.get("labels") or ""
        safe = _safe_image_name(str(fn))
        if not safe:
            self.send_error(400, "bad filename")
            return
        src = BEFORE / safe
        if not src.is_file():
            self.send_error(400, "not in before_img")
            return
        _ensure_dirs()
        shutil.copy2(src, IMG / safe)
        stem = Path(safe).stem
        lbl_path = LABELS / f"{stem}.txt"
        text = labels.rstrip()
        lbl_path.write_text(text + "\n" if text else "", encoding="utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": True, "image": safe}).encode("utf-8"))

    def _list_before(self) -> None:
        _ensure_dirs()
        names = sorted(
            p.name
            for p in BEFORE.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXT
        )
        body = json.dumps(names, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(body)

    def _serve_before(self, name: str) -> None:
        safe = _safe_image_name(name)
        if not safe:
            self.send_error(400)
            return
        path = BEFORE / safe
        if not path.is_file():
            self.send_error(404)
            return
        self._send_path(path)

    def _send_path(self, path: Path) -> None:
        ext = path.suffix.lower()
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }.get(ext, "application/octet-stream")
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args) -> None:
        print("%s - %s" % (self.address_string(), fmt % args))


def main() -> None:
    _ensure_dirs()
    port = int(os.environ.get("ANNOTATE_PORT", "8810"))
    host = os.environ.get("ANNOTATE_HOST", "127.0.0.1")
    httpd = HTTPServer((host, port), AnnotateHandler)
    print("open http://%s:%s/" % (host, port))
    httpd.serve_forever()


if __name__ == "__main__":
    main()
