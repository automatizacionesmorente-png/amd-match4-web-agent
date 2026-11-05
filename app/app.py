import argparse
import json
import os
import socket
import subprocess
import sys
import time

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
SOCK = os.environ.get("OCR_SOCK", "/tmp/mc2-ocr.sock")
BUDGET = float(os.environ.get("IMAGE_BUDGET", "27"))


def write(stem, text="", conf=0.0):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{stem}_output.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"text": str(text or ""), "confidence": float(conf)}, f, ensure_ascii=False)
    os.replace(tmp, path)


def ask(req, timeout):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(max(1.0, timeout))
    s.connect(SOCK)
    s.sendall((json.dumps(req) + "\n").encode())
    data = b""
    while not data.endswith(b"\n"):
        chunk = s.recv(65536)
        if not chunk:
            break
        data += chunk
    s.close()
    return json.loads(data.decode())


def server_running():
    try:
        with open(SOCK + ".pid") as f:
            os.kill(int(f.read().strip()), 0)
        return True
    except (OSError, ValueError):
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-image", "--input_image", "--image", dest="image", required=True)
    a, _ = p.parse_known_args()
    stem = os.path.splitext(os.path.basename(a.image))[0]
    write(stem)  # a valid file is on disk before anything can go wrong

    if not os.path.exists(SOCK) and not server_running():
        # The container's CMD normally starts the server; this covers a harness that overrides CMD.
        subprocess.Popen([sys.executable, os.path.join(HERE, "server.py")], start_new_session=True,
                         stdin=subprocess.DEVNULL, stdout=open("/tmp/mc2-server.log", "ab"), stderr=subprocess.STDOUT)
    while not os.path.exists(SOCK) and time.time() - T0 < BUDGET - 3:
        time.sleep(0.25)
    try:
        res = ask({"image": os.path.abspath(a.image)}, BUDGET - (time.time() - T0))
        write(stem, res.get("text"), res.get("confidence") or 0.0)
        if res.get("error"):
            print(f"[app] {res['error']}", file=sys.stderr)
    except Exception as e:
        print(f"[app] server unavailable: {e!r}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
