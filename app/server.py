import json
import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rag  # noqa: E402
from vlm import VLM  # noqa: E402

SOCK = os.environ.get("RAG_SOCK", "/tmp/mc3-rag.sock")
INDEX_DIR = os.environ.get("MC3_INDEX_DIR", "/app/index")


class State:
    def __init__(self, vlm):
        self.vlm = vlm
        self.idx = None
        self.searcher = None
        self.lock = threading.Lock()

    def ensure(self, corpus, deadline):
        if self.idx is not None and self.idx["corpus"] == os.path.abspath(corpus):
            return
        idx = rag.load_index(INDEX_DIR, corpus)
        if idx is None:
            # Never indexed (or a different corpus): build a text-only index now so the question still gets answered.
            idx = rag.build_index(corpus, None, deadline)
        self.idx, self.searcher = idx, rag.Searcher(idx)

    def index(self, corpus, deadline):
        idx = rag.build_index(corpus, self.vlm, deadline)
        rag.save_index(idx, INDEX_DIR)
        self.idx, self.searcher = idx, rag.Searcher(idx)
        return {"ok": True, "files": len(idx["files"]), "skipped": idx["skipped"]}

    def query(self, corpus, query, deadline):
        self.ensure(corpus, deadline)
        ans, cites, conf = rag.answer(self.idx, self.searcher, self.vlm, query, deadline)
        return {"answer": ans, "citations": cites, "confidence": conf}


def handle(conn, st):
    with conn:
        data = b""
        while not data.endswith(b"\n"):
            chunk = conn.recv(65536)
            if not chunk:
                break
            data += chunk
        try:
            req = json.loads(data.decode())
            deadline = float(req.get("deadline") or time.time() + 25)
            if req.get("op") == "ping":
                res = {"ok": True, "model": st.vlm.ok}
            else:
                with st.lock:
                    if req.get("op") == "index":
                        res = st.index(req["corpus"], deadline)
                    else:
                        res = st.query(req["corpus"], req["query"], deadline)
        except Exception as e:
            res = {"answer": "", "citations": [], "confidence": 0.0, "error": repr(e)}
        try:
            conn.sendall((json.dumps(res) + "\n").encode())
        except OSError:
            pass


def main():
    with open(SOCK + ".pid", "w") as f:
        f.write(str(os.getpid()))
    st = State(VLM())
    print(f"[server] model loaded: {st.vlm.ok}", flush=True)
    tmp = SOCK + ".tmp"
    for p in (tmp, SOCK):
        if os.path.exists(p):
            os.remove(p)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(tmp)
    s.listen(16)
    # The socket only appears under its real name once it is accepting, so a client never connects too early.
    os.replace(tmp, SOCK)
    while True:
        conn, _ = s.accept()
        threading.Thread(target=handle, args=(conn, st), daemon=True).start()


if __name__ == "__main__":
    main()
