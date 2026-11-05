import json
import math
import os
import re
import time

from parse import parse_file, walk

STOP = set("""a an the of to in on for and or is are was were be by with from at as what which who whom whose when where
why how does do did this that these those it its their there here into about can could should would please give tell
me show find value number""".split())
ID_RE = re.compile(r"\b[A-Z][A-Z0-9]*(?:[-_][A-Z0-9]+)+#?|\b[A-Z]{1,3}\d{3,}\b")
STALE_RE = re.compile(r"\b(withdrawn|superseded|obsolete|deprecated|do not use)\b", re.I)
REV_RE = re.compile(r"^(?P<stem>.*?)[\s_\-]*(?:r|rev|v|version)[\s_\-]?(?P<n>\d+)(?P<tail>.*)$", re.I)
UNIT_RE = re.compile(r"^(?P<num>[-+]?\d[\d,]*(?:\.\d+)?)\s*(?:°\s*[CF]|degrees?(?:\s+[CF]|\s+celsius)?|seconds?|secs?|s|ms|"
                     r"minutes?|mins?|hours?|hrs?|h|days?|[kMG]?Hz|[mk]?V|[mk]?W|[mk]?A|mm|cm|m|kg|g|[KMGT]i?B|%|units?|pcs)\.?$", re.I)
TRANSCRIBE = ("Transcribe every piece of text in this image exactly as written. For diagrams, tables, labels and forms, "
              "write each label next to the value or label it belongs to or is connected to, one pair per line, e.g. "
              "'B14: THERM_ALERT#' or 'BOARD REVISION: REV-C2'. Then add one line describing anything else important. "
              "No other commentary.")
SYSTEM = "You answer questions strictly from a company's internal documents. The products are fictional: never use outside knowledge."
PROMPT = """Question: {q}

Documents:
{ctx}
{img_note}
Rules:
1. Answer with the value only: a number, part number, version, code, quarter or name. No sentence, no explanation, no units.
2. Give the complete value as written, keeping qualifiers that identify it (for example "Q3 FY27", not "Q3"; "REV-C2", not "C2").
3. Use only the documents above. If they do not contain the answer, the answer is "".
4. Never use a document marked SUPERSEDED when a current document covers the same thing.
5. In source code, the answer is the value actually assigned in the code, not an old value mentioned in a comment.
6. "files" lists ONLY the files you needed: a file belongs there only if the answer would be impossible without it. If one file gave you an identifier (a ticket, error code, part number) and another file gave the value for that identifier, list both. Usually it is one file.
7. "evidence" is the exact short line from the documents that contains the answer.

Reply with JSON only: {{"answer": "...", "files": ["path"], "evidence": "..."}}"""


def tokens(s):
    out = []
    for t in re.findall(r"[a-z0-9][a-z0-9_#\-.]*[a-z0-9#]|[a-z0-9]", s.lower()):
        if t in STOP:
            continue
        out.append(t)
        if re.search(r"[-_.]", t):
            out += [p for p in re.split(r"[-_.]", t) if p and p not in STOP]
    return out


def alnum(s):
    return re.sub(r"[^0-9a-z]", "", s.lower())


def _flex(ans):
    chars = [re.escape(c) for c in ans if c.isalnum()]
    if not chars:
        return None
    return re.compile(r"(?<![0-9A-Za-z])" + r"[\s\-._·:°/]*".join(chars) + r"(?![0-9A-Za-z])", re.I)


def contains(ans, text):
    rx = _flex(ans)
    return bool(rx and rx.search(text))


def lines_with(ans, text):
    rx = _flex(ans)
    return [l for l in text.splitlines() if rx and rx.search(l)] if rx else []


# ---------------------------------------------------------------- index

def _mark_stale(files):
    for f in files:
        name = os.path.basename(f["path"])
        # Only a document's own title lines can withdraw it: "supersedes revision 1 ... Revision 1 is withdrawn"
        # in a current datasheet is about the other file.
        head = [l for l in f["text"].splitlines() if l.strip() and not l.startswith("[")][:4]
        own = any(STALE_RE.search(l) and not re.search(r"\bsupersedes\b|\b(r|rev|revision)\s*\d+ (is|was) ", l, re.I)
                  for l in head)
        if STALE_RE.search(name) or own:
            f["stale"] = "WITHDRAWN/SUPERSEDED - do not use"
    groups = {}
    for i, f in enumerate(files):
        base, ext = os.path.splitext(os.path.basename(f["path"]))
        m = REV_RE.match(re.sub(STALE_RE, "", base))
        if m:
            key = (os.path.dirname(f["path"]), m.group("stem").lower().strip("_- "), ext.lower())
            groups.setdefault(key, []).append((int(m.group("n")), i))
    for members in groups.values():
        if len(members) < 2:
            continue
        top_n, top_i = max(members)
        for n, i in members:
            if n < top_n and "stale" not in files[i]:
                files[i]["stale"] = f"OLDER REVISION - SUPERSEDED by {files[top_i]['path']}"


def _chunks(f, size=900):
    lines = [l for l in f["text"].splitlines() if l.strip()]
    if f["kind"] in ("csv", "xlsx"):
        header = lines[0] if lines else ""
        return [header + "\n" + l if l != header else l for l in lines] or [""]
    out, cur = [], ""
    for l in lines:
        if cur and len(cur) + len(l) > size:
            out.append(cur)
            cur = cur.splitlines()[-1] + "\n" if cur.splitlines() else ""
        cur += l + "\n"
    if cur.strip() or not out:
        out.append(cur)
    return out


def build_index(corpus, vlm=None, deadline=None):
    deadline = deadline or time.time() + 480
    files, skipped = [], []
    for rel, path in walk(corpus):
        try:
            kind, text, embedded = parse_file(path)
        except Exception as e:
            skipped.append(f"{rel}: {e!r}"[:200])
            continue
        if vlm is not None and vlm.ok and time.time() < deadline - 20:
            try:
                if kind == "image":
                    text = vlm.generate(TRANSCRIBE, [vlm.load_image(path)], max_new=512)
                for data in embedded[:4]:
                    if time.time() > deadline - 20:
                        break
                    text += "\n[embedded image]\n" + vlm.generate(TRANSCRIBE, [vlm.load_image(data)], max_new=400)
            except Exception as e:
                skipped.append(f"{rel}: vision {e!r}"[:200])
        files.append({"path": rel, "abs": path, "kind": kind, "text": text})
    _mark_stale(files)
    return {"corpus": os.path.abspath(corpus), "files": files, "skipped": skipped}


def save_index(idx, index_dir):
    os.makedirs(index_dir, exist_ok=True)
    tmp = os.path.join(index_dir, "index.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(idx, fh, ensure_ascii=False)
    os.replace(tmp, os.path.join(index_dir, "index.json"))


def load_index(index_dir, corpus):
    try:
        with open(os.path.join(index_dir, "index.json"), encoding="utf-8") as fh:
            idx = json.load(fh)
        if idx.get("corpus") == os.path.abspath(corpus):
            return idx
    except Exception:
        pass
    return None


class Searcher:
    def __init__(self, idx):
        self.files = idx["files"]
        self.chunks = []
        for fi, f in enumerate(self.files):
            path_toks = tokens(f["path"].replace("/", " "))
            for c in _chunks(f):
                self.chunks.append((fi, c, tokens(c) + path_toks))
        self.df = {}
        for _, _, ts in self.chunks:
            for t in set(ts):
                self.df[t] = self.df.get(t, 0) + 1
        self.n = len(self.chunks) or 1
        self.avg = sum(len(ts) for _, _, ts in self.chunks) / self.n or 1

    def score(self, q_toks, ts):
        tf = {}
        for t in ts:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for q in set(q_toks):
            f = tf.get(q, 0)
            if f:
                idf = math.log(1 + (self.n - self.df[q] + 0.5) / (self.df[q] + 0.5))
                s += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * len(ts) / self.avg))
        return s

    def search(self, query, k=10):
        q = tokens(query)
        scored = [(self.score(q, ts), ci) for ci, (_, _, ts) in enumerate(self.chunks)]
        scored = [x for x in scored if x[0] > 0]
        scored.sort(reverse=True)
        return scored[:k]


# ---------------------------------------------------------------- answer

def _clean(ans):
    ans = ("" if ans is None else str(ans)).strip().strip('"\'`').strip().rstrip(".")
    if re.fullmatch(r"(?i)(n/?a|none|null|unknown|not (found|available|stated|provided|specified).*|"
                    r"(i )?(do not|don't) know.*|no answer.*|the documents? do(es)? not.*|-)", ans):
        return ""
    m = UNIT_RE.match(ans)
    return m.group("num") if m else ans


def _parse(raw):
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        try:
            d = json.loads(re.sub(r",\s*}", "}", m.group(0)))
        except Exception:
            return None
    if not isinstance(d, dict):
        return None
    files = d.get("files") or d.get("citations") or d.get("sources") or []
    if isinstance(files, str):
        files = [files]
    return _clean(d.get("answer")), [str(x) for x in files], str(d.get("evidence") or "")


def _bridge_ids(lines, query, ans):
    ids = set()
    for l in lines:
        for m in ID_RE.findall(l):
            if alnum(m) and alnum(m) not in alnum(query) and alnum(m) not in alnum(ans) and alnum(ans) not in alnum(m):
                ids.add(m)
    return ids


def answer(idx, searcher, vlm, query, deadline):
    files = idx["files"]
    by_path = {f["path"]: i for i, f in enumerate(files)}
    hits = searcher.search(query, k=12)
    if not hits:
        return "", [], 0.0
    chosen = [ci for _, ci in hits]
    # Second hop: identifiers found next to the query's terms lead to the files that hold their values.
    q_alnum = alnum(query)
    hop_ids = []
    for _, ci in hits[:5]:
        for m in ID_RE.findall(searcher.chunks[ci][1]):
            if alnum(m) not in q_alnum and m not in hop_ids:
                hop_ids.append(m)
    for ident in hop_ids[:8]:
        rx = _flex(ident)
        extra = [ci for ci, (fi, c, _) in enumerate(searcher.chunks) if ci not in chosen and rx.search(c)]
        chosen += extra[:3]

    file_order, file_chunks = [], {}
    for ci in chosen:
        fi = searcher.chunks[ci][0]
        if fi not in file_chunks:
            file_order.append(fi)
            file_chunks[fi] = []
        file_chunks[fi].append(searcher.chunks[ci][1])
    rank1 = file_order[0]
    file_order = file_order[:8]

    parts, budget = [], 18000
    for fi in file_order:
        f = files[fi]
        label = f" [{f['stale']}]" if f.get("stale") else ""
        kind = "image, transcribed" if f["kind"] == "image" else f["kind"]
        body = f["text"] if len(f["text"]) <= 4000 else "\n...\n".join(dict.fromkeys(file_chunks[fi]))
        block = f"=== FILE: {f['path']} ({kind}){label}\n{body.strip()}\n"
        if len(block) > budget:
            block = block[:budget]
        parts.append(block)
        budget -= len(block)
        if budget <= 200:
            break
    if vlm is None or not vlm.ok or time.time() > deadline - 4:
        return "", [], 0.0
    images, img_paths = [], []
    for fi in file_order:
        if files[fi]["kind"] == "image" and len(images) < 2 and time.time() < deadline - 12:
            try:
                images.append(vlm.load_image(files[fi]["abs"]))
                img_paths.append(files[fi]["path"])
            except Exception:
                pass
    img_note = ("\nThe attached images are, in order: " + ", ".join(img_paths) + ". Read them directly.\n") if img_paths else ""
    prompt = PROMPT.format(q=query, ctx="\n".join(parts), img_note=img_note)
    try:
        parsed = _parse(vlm.generate(prompt, images, max_new=200, system=SYSTEM))
    except Exception as e:
        print(f"[rag] generation failed: {e!r}", flush=True)
        parsed = None
    if parsed is None:
        return "", [], 0.0
    ans, cited, _evidence = parsed
    if not ans:
        return "", [], 0.1

    def norm_path(p):
        p = p.strip().replace("\\", "/")
        root = idx["corpus"].rstrip("/") + "/"
        if p.startswith(root):
            p = p[len(root):]
        p = re.sub(r"^(/app/corpus/|\./)", "", p)
        if p in by_path:
            return p
        cand = [q for q in by_path if q.endswith("/" + p) or os.path.basename(q) == os.path.basename(p)]
        return cand[0] if len(cand) == 1 else None

    cited = [p for p in dict.fromkeys(norm_path(c) for c in cited) if p]
    holds = lambda p: contains(ans, files[by_path[p]]["text"]) or (files[by_path[p]]["kind"] == "image" and p in img_paths)
    with_ans = [p for p in cited if holds(p)]
    if not with_ans:
        pool = [files[fi]["path"] for fi in file_order if holds(files[fi]["path"])]
        fresh = [p for p in pool if not files[by_path[p]].get("stale")]
        if not (fresh or pool):
            return "", [], 0.05
        with_ans = [(fresh or pool)[0]]
    fresh = [p for p in with_ans if not files[by_path[p]].get("stale")]
    with_ans = fresh or with_ans

    support = []
    for p in with_ans:
        support += lines_with(ans, files[by_path[p]]["text"])
    ids = _bridge_ids(support, query, ans)
    bridges = []
    for p in cited:
        if p not in with_ans and any(contains(i, files[by_path[p]]["text"]) for i in ids):
            bridges.append(p)
    top = files[rank1]
    if top["path"] not in with_ans and not top.get("stale") and any(contains(i, top["text"]) for i in ids):
        bridges.append(top["path"])
    cites = list(dict.fromkeys(with_ans + bridges))[:3]
    return ans, cites, 0.8
