import argparse
import json
import os
import socket
import subprocess
import sys
import time

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("MC2_OUTPUT_DIR", "/app/output")
SOCK = os.environ.get("RAG_SOCK", "/tmp/mc3-rag.sock")
QUERY_BUDGET = float(os.environ.get("QUERY_BUDGET", "26"))
INDEX_BUDGET = float(os.environ.get("INDEX_BUDGET", "540"))


def write(qid, ans="", cites=(), conf=0.0):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{qid}_output.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"answer": str(ans or ""), "citations": [str(c) for c in cites], "confidence": float(conf)},
                  f, ensure_ascii=False)
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


def alive():
    try:
        return ask({"op": "ping"}, 3).get("ok", False)
    except Exception:
        return False


def server_running():
    try:
        with open(SOCK + ".pid") as f:
            os.kill(int(f.read().strip()), 0)
        return True
    except (OSError, ValueError):
        return False


def start_server():
    # The container's CMD normally starts the server; this covers a harness that overrides CMD.
    # A server that is still loading the model has written its pid but not yet its socket.
    if server_running():
        return
    subprocess.Popen([sys.executable, os.path.join(HERE, "server.py")], start_new_session=True,
                     stdin=subprocess.DEVNULL, stdout=open("/tmp/mc3-server.log", "ab"), stderr=subprocess.STDOUT)


def wait_for_server(until):
    while time.time() < until:
        if os.path.exists(SOCK) and alive():
            return True
        time.sleep(0.5)
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index")
    p.add_argument("--corpus")
    p.add_argument("--query-id", "--query_id", dest="qid")
    p.add_argument("--query", "--question", dest="query")
    a, _ = p.parse_known_args()

    if a.index:
        end = T0 + INDEX_BUDGET
        if not (os.path.exists(SOCK) and alive()):
            time.sleep(2)
            start_server()
        if wait_for_server(end - 60):
            try:
                res = ask({"op": "index", "corpus": a.index, "deadline": end - 15}, end - time.time())
                print(f"[app] indexed {res.get('files')} files; skipped: {res.get('skipped')}", flush=True)
            except Exception as e:
                print(f"[app] index failed: {e!r}", file=sys.stderr)
        return 0

    qid = a.qid or "query"
    write(qid)  # a valid refusal is on disk before anything can go wrong
    if not a.query or not a.corpus:
        return 0
    end = T0 + QUERY_BUDGET
    if not os.path.exists(SOCK):
        start_server()
    if not wait_for_server(end - 3):
        return 0
    try:
        res = ask({"op": "query", "corpus": a.corpus, "query": a.query, "deadline": end - 1.5}, end - time.time())
        write(qid, res.get("answer"), res.get("citations") or [], res.get("confidence") or 0.0)
    except Exception as e:
        print(f"[app] query failed: {e!r}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
