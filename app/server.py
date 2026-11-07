import json
import os
import socket
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent  # noqa: E402

SOCK = os.environ.get("AGENT_SOCK", "/tmp/web-agent.sock")


class Loading:
    """Stands in for the model while it loads: questions are answered by the rules from the first second."""
    ok = False


state = {"llm": Loading()}


def load_model():
    llm = agent.LLM()
    state["llm"] = llm
    print(f"[server] model loaded: {llm.ok}", flush=True)


def handle(conn):
    llm = state["llm"]
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
                # No global lock: the model guards itself, so a late answer never blocks the next question.
                res = agent.solve(llm, req["query"], req.get("url"), req.get("corpus"), float(req.get("budget", 22)))
        except Exception as e:
            res = {"answer": "", "citations": [], "confidence": 0.0, "error": repr(e)}
        conn.sendall((json.dumps(res) + "\n").encode())


def main():
    # Listen before loading: a large model takes a while to load, and no question should wait for it.
    if os.path.exists(SOCK):
        os.remove(SOCK)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(SOCK)
    s.listen(16)
    threading.Thread(target=load_model, daemon=True).start()
    while True:
        conn, _ = s.accept()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    main()
