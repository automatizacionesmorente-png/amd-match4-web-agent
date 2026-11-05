import argparse
import json
import os
import socket
import sys
import time

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
SOCK = os.environ.get("AGENT_SOCK", "/tmp/web-agent.sock")
BUDGET = float(os.environ.get("QUERY_BUDGET", "24"))


def write(qid, res):
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {"answer": str(res.get("answer") or ""), "citations": list(res.get("citations") or []),
           "confidence": float(res.get("confidence") or 0.0)}
    url = out["citations"][0] if out["citations"] else ""
    out.update({"sources": out["citations"], "source_url": url})
    path = os.path.join(OUT_DIR, f"{qid}_output.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    os.replace(tmp, path)


def ask_server(req, timeout):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--query", "--question", "-q", dest="query")
    p.add_argument("--query-id", "--query_id", "--id", dest="qid")
    p.add_argument("--url", "--website", "--start-url", "--site", dest="url")
    p.add_argument("--corpus", "--pages", dest="corpus")
    p.add_argument("--index", dest="index")
    p.add_argument("--input", dest="input")
    a, _ = p.parse_known_args()

    if a.index:
        return
    if a.input and os.path.isfile(a.input):
        with open(a.input, encoding="utf-8") as f:
            d = json.load(f)
        a.query = a.query or d.get("query") or d.get("question")
        a.url = a.url or d.get("url") or d.get("website") or d.get("start_url")
        a.qid = a.qid or d.get("query_id") or d.get("id") or os.path.splitext(os.path.basename(a.input))[0]
    qid = a.qid or "query"
    write(qid, {})
    if not a.query:
        return

    req = {"query": a.query, "url": a.url, "corpus": a.corpus or (a.url if a.url and os.path.isdir(a.url) else None)}
    if req["corpus"]:
        req["url"] = None
    remaining = BUDGET - (time.time() - T0)
    try:
        req["budget"] = remaining - 2.5
        res = ask_server(req, remaining - 0.5)
    except Exception as e:
        print(f"[app] server unavailable ({e!r}); running rules-only", file=sys.stderr)
        sys.path.insert(0, HERE)
        import agent
        res = agent.solve(None, req["query"], req["url"], req["corpus"], BUDGET - (time.time() - T0) - 2)
    write(qid, res)


if __name__ == "__main__":
    main()
