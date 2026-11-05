import json
import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ocr  # noqa: E402
from vlm import VLM  # noqa: E402

SOCK = os.environ.get("OCR_SOCK", "/tmp/mc2-ocr.sock")


def read_image(vlm, path, deadline=None):
    im = vlm.load_image(path)
    # Small crops are upscaled so a plate is at least ~900 px wide; huge photos are capped by the processor's max_pixels.
    w, h = im.size
    if max(w, h) < 900:
        s = 900 / max(w, h)
        im = im.resize((round(w * s), round(h * s)))
    raw = vlm.generate(ocr.PROMPT, [im], max_new=80, max_time=deadline - time.time() - 0.5 if deadline else None)
    return ocr.clean(raw), raw


def handle(conn, vlm, lock):
    with conn:
        data = b""
        while not data.endswith(b"\n"):
            chunk = conn.recv(65536)
            if not chunk:
                break
            data += chunk
        try:
            req = json.loads(data.decode())
            if req.get("op") == "ping":
                res = {"ok": True, "model": vlm.ok}
            elif not vlm.ok:
                res = {"text": "", "confidence": 0.0, "error": "model not loaded"}
            else:
                deadline = req.get("deadline")
                with lock:
                    # A request that waited behind a slow one and whose caller already gave up is skipped,
                    # so one slow image can never make every later image time out.
                    if deadline and time.time() > deadline - 1:
                        text, raw = "", ""
                    else:
                        text, raw = read_image(vlm, req["image"], deadline)
                res = {"text": text, "confidence": 0.9 if text else 0.0, "raw": raw}
        except Exception as e:
            res = {"text": "", "confidence": 0.0, "error": repr(e)}
        try:
            conn.sendall((json.dumps(res, ensure_ascii=False) + "\n").encode())
        except OSError:
            pass


def main():
    with open(SOCK + ".pid", "w") as f:
        f.write(str(os.getpid()))
    vlm = VLM()
    print(f"[server] model loaded: {vlm.ok}", flush=True)
    tmp = SOCK + ".tmp"
    for p in (tmp, SOCK):
        if os.path.exists(p):
            os.remove(p)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(tmp)
    s.listen(16)
    os.replace(tmp, SOCK)
    lock = threading.Lock()
    while True:
        conn, _ = s.accept()
        threading.Thread(target=handle, args=(conn, vlm, lock), daemon=True).start()


if __name__ == "__main__":
    main()
