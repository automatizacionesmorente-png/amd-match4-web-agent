import re

PATTERNS = [
    (r"\b(phone|telephone|tel|call|fax|mobile|hotline)\b",
     r"(?:\+\d{1,3}[\s.-]?)?(?:\(\d{1,4}\)[\s.-]?)?\d{2,4}(?:[\s.-]\d{2,4}){1,4}(?![\d\s.-]*\d)"),
    (r"\b(e-?mail|email address)\b", r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    (r"\b(version|release|firmware|build)\b", r"\bv?\d+(?:\.\d+){1,3}(?:[-.]?(?:rc|beta|alpha|post|dev)\d*)?\b"),
    (r"\b(price|cost|fee|how much|pricing|subscription)\b", r"(?:[$€£¥]\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s?(?:USD|EUR|GBP|€|\$))"),
    (r"\b(date|when|released on|founded|deadline|year)\b",
     r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}|(?:19|20)\d{2})\b"),
    (r"\b(how many|number of|count|percent|percentage)\b", r"\b\d[\d,]*(?:\.\d+)?%?"),
]


QSTOP = set("what which who when where how is are was the of a an to in on for and or does do this that site page website value latest current".split())


def rule_extract(query, ranked):
    q = query.lower()
    qwords = [w for w in re.findall(r"[a-z]{3,}", q) if w not in QSTOP]
    for trigger, pat in PATTERNS:
        if not re.search(trigger, q):
            continue
        near = re.compile(trigger + "|" + "|".join(map(re.escape, qwords)) if qwords else trigger, re.I)
        for page, chunk, score in ranked:
            if score <= 0:
                continue
            for m in re.finditer(pat, chunk, re.I):
                # Only trust a pattern hit when the question's words sit right next to it.
                if near.search(chunk[max(0, m.start() - 80):m.end() + 30]):
                    return m.group(0).strip(" ,;:"), page.url, 0.35
        # PATTERNS run most-specific first; "phone number of" must not fall through to the counting rule.
        break
    return "", None, 0.0


def norm(s):
    return re.sub(r"[\s\-.·_,]", "", s.upper())


def grounded(answer, text):
    a = norm(answer)
    if not a:
        return False
    if a in norm(text):
        return True
    digits = re.sub(r"\D", "", answer)
    return len(digits) >= 2 and digits in re.sub(r"\D", "", text)
