import json
import os
import socket
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent  # noqa: E402

SOCK = os.environ.get("AGENT_SOCK", "/tmp/web-agent.sock")


def handle(conn, llm, lock):
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
                res = {"ok": True, "model": llm.ok}
            else:
                with lock:
                    res = agent.solve(llm, req["query"], req.get("url"), req.get("corpus"), float(req.get("budget", 22)))
        except Exception as e:
            res = {"answer": "", "citations": [], "confidence": 0.0, "error": repr(e)}
        conn.sendall((json.dumps(res) + "\n").encode())


def main():
    llm = agent.LLM()
    print(f"[server] model loaded: {llm.ok}", flush=True)
    if os.path.exists(SOCK):
        os.remove(SOCK)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(SOCK)
    s.listen(16)
    lock = threading.Lock()
    while True:
        conn, _ = s.accept()
        threading.Thread(target=handle, args=(conn, llm, lock), daemon=True).start()


if __name__ == "__main__":
    main()
