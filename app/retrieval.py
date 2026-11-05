import concurrent.futures as cf
import gzip
import html
import math
import os
import re
import time
import urllib.parse
import urllib.request
import zlib
from html.parser import HTMLParser

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
STOP = set("""a an the of to in on for and or is are was were be by with from at as what which who whom whose
when where why how does do did this that these those it its their there here into about can could should would
please find tell me give show get i you we they he she my your our list value page site website current latest""".split())
SKIP_EXT = re.compile(r"\.(png|jpe?g|gif|svg|webp|ico|css|js|zip|gz|tar|mp4|mp3|woff2?|ttf|exe|dmg|pdf)(\?|$)", re.I)
BLOCK_TAGS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article",
              "header", "footer", "table", "ul", "ol", "dt", "dd", "pre", "blockquote", "form", "main"}
DROP_TAGS = {"script", "style", "noscript", "svg", "template", "iframe"}


def tokens(text):
    return [t for t in re.findall(r"[a-z0-9][a-z0-9.+#-]*", text.lower()) if t not in STOP and len(t) > 1]


class _PageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.links, self.title = [], [], ""
        self._drop, self._in_title, self._href, self._atext = 0, False, None, []

    def handle_starttag(self, tag, attrs):
        if tag in DROP_TAGS:
            self._drop += 1
            return
        a = dict(attrs)
        if tag == "title":
            self._in_title = True
        if tag == "a" and a.get("href"):
            self._href, self._atext = a["href"], [a.get("title") or "", a.get("aria-label") or ""]
        if tag in ("td", "th"):
            self.parts.append(" | ")
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")
        if tag == "meta" and (a.get("name") or a.get("property") or "").lower() in ("description", "og:description"):
            self.parts.append("\n" + (a.get("content") or "") + "\n")

    def handle_endtag(self, tag):
        if tag in DROP_TAGS:
            self._drop = max(0, self._drop - 1)
            return
        if tag == "title":
            self._in_title = False
        if tag == "a" and self._href is not None:
            self.links.append((self._href, " ".join(self._atext).strip()))
            self._href = None
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._drop:
            return
        if self._in_title:
            self.title += data
        if self._href is not None:
            self._atext.append(data)
        self.parts.append(data)


def parse_html(raw):
    p = _PageParser()
    try:
        p.feed(raw)
    except Exception:
        pass
    text = "".join(p.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", html.unescape(text))
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    return p.title.strip(), text, p.links


def fetch(url, timeout=6.0, max_bytes=3_000_000):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/html,*/*;q=0.8",
                                               "Accept-Encoding": "gzip, deflate", "Accept-Language": "en,*;q=0.5"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        ctype = r.headers.get("Content-Type", "")
        if ctype and not any(k in ctype for k in ("html", "text", "xml", "json")):
            return r.geturl(), ""
        body = r.read(max_bytes)
        enc = (r.headers.get("Content-Encoding") or "").lower()
        if enc == "gzip":
            body = gzip.decompress(body)
        elif enc == "deflate":
            body = zlib.decompress(body)
        charset = r.headers.get_content_charset() or "utf-8"
        return r.geturl(), body.decode(charset, errors="replace")


def _same_site(a, b):
    ha, hb = urllib.parse.urlparse(a).hostname or "", urllib.parse.urlparse(b).hostname or ""
    strip = lambda h: h[4:] if h.startswith("www.") else h
    return strip(ha) == strip(hb) or strip(ha).endswith("." + strip(hb)) or strip(hb).endswith("." + strip(ha))


def _link_score(qtok, url, text):
    words = set(tokens(text + " " + urllib.parse.unquote(urllib.parse.urlparse(url).path).replace("/", " ").replace("-", " ").replace("_", " ")))
    hit = sum(1 for t in qtok if t in words or any(w.startswith(t[:5]) for w in words if len(t) >= 5))
    generic = 0.3 if re.search(r"contact|about|pricing|price|download|release|news|support|faq|team|docs", url + " " + text, re.I) else 0
    return hit + generic


class Page:
    __slots__ = ("url", "title", "text", "depth")

    def __init__(self, url, title, text, depth):
        self.url, self.title, self.text, self.depth = url, title, text, depth


def crawl(start_urls, query, deadline, max_pages=18, max_depth=2, workers=8):
    qtok = tokens(query)
    seen, pages = set(), []
    frontier = [(10.0, u, 0) for u in start_urls]
    root = start_urls[0] if start_urls else ""
    with cf.ThreadPoolExecutor(workers) as ex:
        while frontier and len(pages) < max_pages and time.time() < deadline - 1.0:
            frontier.sort(key=lambda x: -x[0])
            batch, frontier = frontier[:workers], frontier[workers:]
            batch = [b for b in batch if b[1] not in seen]
            for b in batch:
                seen.add(b[1])
            remaining = max(0.5, deadline - time.time() - 0.5)
            futs = {ex.submit(fetch, u, min(6.0, remaining)): (s, u, d) for s, u, d in batch}
            try:
                for f in cf.as_completed(futs, timeout=remaining):
                    s, u, d = futs[f]
                    try:
                        final, raw = f.result()
                    except Exception:
                        continue
                    if not raw:
                        continue
                    title, text, links = parse_html(raw)
                    pages.append(Page(final, title, text, d))
                    seen.add(final)
                    if d >= max_depth:
                        continue
                    for href, ltext in links:
                        nu = urllib.parse.urljoin(final, href.strip()).split("#")[0]
                        if not nu.startswith("http") or nu in seen or SKIP_EXT.search(nu) or not _same_site(nu, root or final):
                            continue
                        sc = _link_score(qtok, nu, ltext)
                        if sc > 0 or d == 0:
                            frontier.append((sc - 0.5 * d, nu, d + 1))
            except cf.TimeoutError:
                break
    return pages


CORPUS_EXT = {".html", ".htm", ".xhtml", ".txt", ".md", ".json", ".xml", ".csv"}


def load_corpus(folder):
    pages = []
    for dirpath, dirnames, filenames in os.walk(folder, onerror=lambda e: None):
        dirnames.sort()
        for fn in sorted(filenames):
            path = os.path.join(dirpath, fn)
            if os.path.splitext(fn)[1].lower() not in CORPUS_EXT:
                continue
            try:
                with open(path, "rb") as fh:
                    raw = fh.read(5_000_000).decode("utf-8", errors="replace")
            except Exception:
                continue
            rel = os.path.relpath(path, folder)
            if fn.lower().endswith((".html", ".htm", ".xhtml", ".xml")):
                title, text, _ = parse_html(raw)
            else:
                title, text = fn, raw
            pages.append(Page(rel, title, text, 0))
    return pages


def passages(pages, query, k=8, size=700, overlap=150):
    qtok = tokens(query)
    chunks = []
    for p in pages:
        t = p.text
        if not t:
            continue
        step = size - overlap
        for i in range(0, max(1, len(t) - overlap), step):
            chunks.append((p, t[i:i + size]))
    if not chunks or not qtok:
        return [(p, c, 0.0) for p, c in chunks[:k]]
    df = {}
    toks = []
    for p, c in chunks:
        ts = tokens(p.title + " " + c)
        toks.append(ts)
        for t in set(ts):
            df[t] = df.get(t, 0) + 1
    n = len(chunks)
    avg = sum(len(t) for t in toks) / n or 1
    scored = []
    for (p, c), ts in zip(chunks, toks):
        tf = {}
        for t in ts:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for q in qtok:
            f = tf.get(q, 0)
            if f:
                idf = math.log(1 + (n - df[q] + 0.5) / (df[q] + 0.5))
                s += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * len(ts) / avg))
        s -= 0.6 * p.depth
        scored.append((p, c, s))
    scored.sort(key=lambda x: -x[2])
    out, per_page = [], {}
    for p, c, s in scored:
        if per_page.get(p.url, 0) >= 3:
            continue
        per_page[p.url] = per_page.get(p.url, 0) + 1
        out.append((p, c, s))
        if len(out) >= k:
            break
    return out


def search_web(query, deadline):
    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
    try:
        _, raw = fetch(url, timeout=min(6.0, max(0.5, deadline - time.time() - 1)))
    except Exception:
        return []
    found = []
    for m in re.finditer(r'class="result__a"[^>]*href="([^"]+)"', raw):
        href = html.unescape(m.group(1))
        q = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
        real = q.get("uddg", [href])[0]
        if real.startswith("http") and "duckduckgo.com" not in real:
            found.append(real)
    return found[:4]
