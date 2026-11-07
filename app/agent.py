import json
import os
import re
import threading
import time

import retrieval
from extract import answer_pattern, expand_date, grounded, rule_extract

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/qwen3-1.7b")
# Used instead when a GPU has room for it: reading and choosing the exact value is where a bigger model pays off.
BIG_MODEL_DIR = os.environ.get("BIG_MODEL_DIR", "/models/qwen3-14b")
BIG_MODEL_MIN_FREE = float(os.environ.get("BIG_MODEL_MIN_FREE_GB", "36")) * 1e9
# Crawling never needs more than this; a large test budget then goes to the (slow, CPU) model instead.
CRAWL_MAX = float(os.environ.get("CRAWL_MAX", "15"))
# Passages shown to the model. On the GPU a few thousand more prompt tokens cost well under a second.
TOPK = int(os.environ.get("TOPK", "12"))

PROMPT = """You extract one exact value from web pages.

Question: {q}

Sources:
{src}

Rules:
- Answer with the value only (a number, name, version, price, date, phone, email, code). No sentence, no units the page does not attach, no explanation.
- Give the COMPLETE value as written on the page (keep prefixes, currency, qualifiers like "Q3 FY27" or "REV-C2").
- A date is the full date as written (day, month and year), unless the question asks only for the year.
- The page heading after "|" in a source's title line is part of that page; a version or name there counts.
- When sources disagree (an old release page against the current one), trust the start page.
- Use only the sources. If they do not contain the answer, answer "".
- List in "sources" only the source ids you needed; usually one.

Reply with JSON only: {{"answer": "...", "sources": ["S1"]}}"""


class LLM:
    def __init__(self):
        self.ok = False
        try:
            import torch
            self.torch = torch
            self.dev = "cuda" if torch.cuda.is_available() else "cpu"
            for path in self._candidates():
                try:
                    self._load(path)
                    break
                except Exception as e:
                    print(f"[agent] could not load {path}: {e!r}", flush=True)
                    self.model = None
                    if self.dev == "cuda":
                        torch.cuda.empty_cache()
            else:
                return
            self.busy = threading.Lock()
            self.ok = True
            self.generate("Say OK.", 4)
            self._calibrate()
        except Exception as e:
            self.ok = False
            print(f"[agent] model unavailable, rules only: {e!r}", flush=True)

    def _candidates(self):
        paths = [MODEL_DIR]
        if self.dev == "cuda" and os.path.isdir(BIG_MODEL_DIR) and BIG_MODEL_DIR != MODEL_DIR:
            free = self.torch.cuda.mem_get_info()[0]
            print(f"[agent] GPU free memory: {free / 1e9:.0f} GB", flush=True)
            if free >= BIG_MODEL_MIN_FREE:
                paths.insert(0, BIG_MODEL_DIR)
        return paths

    def _load(self, path):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(path, dtype=self.torch.bfloat16)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=self.torch.bfloat16)
        self.model.to(self.dev).eval()
        self.name = os.path.basename(path.rstrip("/"))
        print(f"[agent] using {self.name} on {self.dev}", flush=True)

    def _calibrate(self):
        # Measured once at start-up so each question can size its prompt to the time it has left,
        # whatever the hardware: a few thousand tokens are nothing on the GPU and minutes on a CPU.
        t = time.time()
        self.generate("Repeat: " + "the quick brown fox jumps over the lazy dog. " * 50, 1)
        self.prefill = max(1.0, 520 / max(1e-3, time.time() - t))
        t = time.time()
        self.generate("Count from 1 to 30 separated by spaces.", 16)
        self.per_token = max(1e-3, (time.time() - t) / 16)
        print(f"[agent] speed: {self.prefill:.0f} prompt tok/s, {self.per_token * 1000:.0f} ms/token", flush=True)

    def seconds_for(self, prompt, max_new=80):
        return len(self.tok(prompt)["input_ids"]) / getattr(self, "prefill", 1e9) + max_new * getattr(self, "per_token", 0.0)

    def generate(self, prompt, max_new=80, max_time=None):
        msgs = [{"role": "user", "content": prompt}]
        # Qwen3-14B thinks out loud by default; this task needs the value straight away.
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = self.tok(text, return_tensors="pt").to(self.dev)
        extra = {"max_time": max(0.5, max_time)} if max_time else {}
        with self.torch.inference_mode():
            out = self.model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id, **extra)
        text = self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        return re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S).strip()


def _parse(raw):
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None, []
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None, []
    if not isinstance(d, dict):
        return None, []
    ans = d.get("answer", "")
    ans = "" if ans is None else str(ans).strip().strip('"').strip()
    if re.fullmatch(r"(?i)(n/?a|none|null|unknown|not (found|available|stated).*|i don'?t know.*)", ans):
        ans = ""
    srcs = d.get("sources") or d.get("source") or []
    if isinstance(srcs, str):
        srcs = [srcs]
    return ans, [str(s) for s in srcs]


def _urls_in(text):
    return [u.rstrip(").,;'\"") for u in re.findall(r"https?://[^\s<>\"']+", text)]


def solve(llm, query, url=None, corpus=None, budget=22.0):
    t0 = time.time()
    deadline = t0 + budget
    if corpus and os.path.isdir(corpus):
        pages = retrieval.load_corpus(corpus)
    else:
        starts = [url] if url else _urls_in(query)
        if not starts:
            starts = retrieval.search_web(query, t0 + 7)
        pages = retrieval.crawl(starts, query, t0 + min(budget * 0.65, CRAWL_MAX)) if starts else []
    ranked = retrieval.passages(pages, query, k=TOPK, want=answer_pattern(query))
    if not ranked:
        return {"answer": "", "citations": [], "confidence": 0.0}

    prompt = None
    if llm is not None and llm.ok:
        # Drop the weakest passages until the model can answer before the deadline; below 3 the rules answer.
        for k in range(len(ranked), 2, -1):
            src = "\n\n".join(f"[S{i + 1}] {p.url}{' (start page)' if p.depth == 0 else ''}\n{p.title}\n{c}"
                              for i, (p, c, _) in enumerate(ranked[:k]))
            # Half the time left: on a CPU, prefill of a long prompt is slower than the short calibration.
            if llm.seconds_for(PROMPT.format(q=query, src=src)) < (deadline - time.time() - 1.5) * 0.5:
                prompt = PROMPT.format(q=query, src=src)
                break
    # The prompt's prefill cannot be interrupted, so the model runs in its own thread under a hard deadline;
    # if it is late, or still busy with an earlier late question, the rules answer instead.
    ans, ids = None, []
    if prompt is not None and llm.busy.acquire(blocking=False):
        box = {}

        def run():
            try:
                box["raw"] = llm.generate(prompt, max_time=deadline - time.time() - 1)
            except Exception as e:
                print(f"[agent] generation failed: {e!r}", flush=True)
            finally:
                llm.busy.release()

        th = threading.Thread(target=run, daemon=True)
        th.start()
        th.join(max(0.0, deadline - time.time() - 0.3))
        if "raw" in box:
            ans, ids = _parse(box["raw"])
        else:
            print("[agent] model late; rules answer", flush=True)
    if ans is not None:
        picked = []
        for s in ids:
            m = re.search(r"\d+", s)
            if m and 1 <= int(m.group(0)) <= len(ranked):
                picked.append(ranked[int(m.group(0)) - 1])
        if ans:
            support = picked or [r for r in ranked if grounded(ans, r[1])][:1]
            support = [r for r in support if grounded(ans, r[1])] or [r for r in ranked if grounded(ans, r[1])][:1]
            if support:
                ans = expand_date(query, ans, [c for _, c, _ in support])
                cites = list(dict.fromkeys(p.url for p, _, _ in support))
                return {"answer": ans, "citations": cites, "confidence": 0.8}
        else:
            return {"answer": "", "citations": [], "confidence": 0.1}

    ans, src_url, conf = rule_extract(query, ranked)
    if ans:
        return {"answer": ans, "citations": [src_url], "confidence": conf}
    return {"answer": "", "citations": [], "confidence": 0.0}
