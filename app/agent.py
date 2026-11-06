import json
import os
import re
import time

import retrieval
from extract import answer_pattern, expand_date, grounded, rule_extract

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/qwen3-4b")
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
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.torch = torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.tok = AutoTokenizer.from_pretrained(MODEL_DIR)
            try:
                self.model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
            except TypeError:
                self.model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
            self.model.to(dev).eval()
            self.dev = dev
            self.ok = True
            self.generate("Say OK.", 4)
        except Exception as e:
            print(f"[agent] model unavailable, rules only: {e!r}", flush=True)

    def generate(self, prompt, max_new=80):
        msgs = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt").to(self.dev)
        with self.torch.inference_mode():
            out = self.model.generate(**ids, max_new_tokens=max_new, do_sample=False, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


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

    if llm is not None and llm.ok and time.time() < deadline - 3:
        src = "\n\n".join(f"[S{i + 1}] {p.url}{' (start page)' if p.depth == 0 else ''}\n{p.title}\n{c}"
                          for i, (p, c, _) in enumerate(ranked))
        try:
            ans, ids = _parse(llm.generate(PROMPT.format(q=query, src=src)))
        except Exception as e:
            print(f"[agent] generation failed: {e!r}", flush=True)
            ans, ids = None, []
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
