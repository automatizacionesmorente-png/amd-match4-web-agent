import json
import re

PROMPT = """Read this image. It shows a vehicle licence plate or a road sign.

If it is a LICENCE PLATE: give only the registration number, exactly as printed, in reading order.
Leave out everything printed around the number: state or country names (e.g. CALIFORNIA, NEW YORK),
slogans and mottos (e.g. EXCELSIOR, THE LONE STAR STATE), websites, month/year stickers and dealer frames.
On a Chinese plate the leading province character and letter ARE part of the number: 京A·12345 stays 京A·12345.

If it is a ROAD SIGN: give every word and number printed on the sign, top to bottom, left to right,
separated by single spaces (e.g. "SPEED LIMIT 65", "ROAD WORK AHEAD", "STOP"). Never add units or words
that are not printed (a plaque that shows only 35 is "35").

Read each character carefully; the image may be blurred, noisy, dark, glared or seen at an angle.
Reply with JSON only: {"type": "plate" or "sign", "text": "..."}"""

STATES = """ALABAMA ALASKA ARIZONA ARKANSAS CALIFORNIA COLORADO CONNECTICUT DELAWARE FLORIDA GEORGIA HAWAII IDAHO
ILLINOIS INDIANA IOWA KANSAS KENTUCKY LOUISIANA MAINE MARYLAND MASSACHUSETTS MICHIGAN MINNESOTA MISSISSIPPI MISSOURI
MONTANA NEBRASKA NEVADA NEW_HAMPSHIRE NEW_JERSEY NEW_MEXICO NEW_YORK NORTH_CAROLINA NORTH_DAKOTA OHIO OKLAHOMA OREGON
PENNSYLVANIA RHODE_ISLAND SOUTH_CAROLINA SOUTH_DAKOTA TENNESSEE TEXAS UTAH VERMONT VIRGINIA WASHINGTON WEST_VIRGINIA
WISCONSIN WYOMING DISTRICT_OF_COLUMBIA PUERTO_RICO USA""".split()
# Phrases that only ever appear around a plate number, never in it. Longest first so multi-word ones go whole.
BANNERS = sorted([s.replace("_", " ") for s in STATES] + [
    "EXCELSIOR", "THE EMPIRE STATE", "EMPIRE STATE", "THE LONE STAR STATE", "LONE STAR STATE", "SUNSHINE STATE",
    "THE GARDEN STATE", "GARDEN STATE", "LAND OF LINCOLN", "GREAT LAKES STATE", "PURE MICHIGAN", "THE KEYSTONE STATE",
    "KEYSTONE STATE", "LIVE FREE OR DIE", "FAMOUS POTATOES", "GRAND CANYON STATE", "THE GOLDEN STATE", "GOLDEN STATE",
    "THE OCEAN STATE", "OCEAN STATE", "FIRST IN FLIGHT", "FIRST IN FREEDOM", "THE BEEHIVE STATE", "LIFE ELEVATED",
    "DMV CA GOV", "DMV.CA.GOV", "TEXAS GOV", "VIRGINIA IS FOR LOVERS", "WILD WONDERFUL", "BIG SKY COUNTRY",
    "THE CONSTITUTION STATE", "CONSTITUTION STATE", "LAND OF ENCHANTMENT", "AMERICA'S DAIRYLAND", "SHOW ME STATE",
    "THE NATURAL STATE", "NATURAL STATE", "TREASURE STATE", "THE SILVER STATE", "SILVER STATE", "HEART OF DIXIE",
    "THE PEACH STATE", "PEACH STATE", "VACATIONLAND", "THE FIRST STATE", "MY OLD KENTUCKY HOME", "UNBRIDLED SPIRIT",
    "SPORTSMAN'S PARADISE", "COLORFUL COLORADO", "ALOHA STATE", "GREEN MOUNTAIN STATE", "EVERGREEN STATE",
    "PASSENGER", "COMMERCIAL", "LICENSE PLATE", "LICENCE PLATE",
], key=len, reverse=True)
# Words that mean the model read a sign even if it called it a plate; sign text is never trimmed.
SIGN_WORDS = (r"\b(SPEED|LIMIT|STOP|YIELD|AHEAD|ROAD|WORK|ZONE|MPH|KM/H|EXIT|DETOUR|CAUTION|SLOW|BRIDGE|ICY|ICE|"
              r"MERGE|LANE|CLOSED|ONE WAY|DO NOT|ENTER|WRONG WAY|SCHOOL|CURVE|TURN|NO|MINIMUM|MAXIMUM|END)\b")
PREFIX_RE =re.compile(r"^(?:the\s+)?(?:text|answer|plate(?:\s+number)?|registration(?:\s+number)?|sign|reads?|it\s+(?:says|reads))\s*[:=-]\s*", re.I)


def norm(s):
    return re.sub(r"[\s\-.·_]", "", s.upper())


def _parse(raw):
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            if isinstance(d, dict) and "text" in d:
                return str(d.get("type") or "").lower(), str(d.get("text") or "")
        except Exception:
            pass
        m2 = re.search(r'"text"\s*:\s*"([^"]*)', m.group(0))
        if m2:
            t = re.search(r'"type"\s*:\s*"(\w+)', m.group(0))
            return (t.group(1).lower() if t else ""), m2.group(1)
    return "", raw


def clean(raw):
    kind, text = _parse(raw)
    text = text.replace("\\n", "\n").strip().strip('"\'`').strip()
    text = PREFIX_RE.sub("", text)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    signish = re.search(SIGN_WORDS, text, re.I)
    if not signish and (kind == "plate" or (not kind and len(lines) <= 3 and any(re.search(r"\d", l) for l in lines))):
        # A plate's number is the line with digits (or Chinese characters); banner lines carry neither.
        keep = [l for l in lines if re.search(r"\d|[一-鿿]", l)] or lines
        text = " ".join(keep)
        up = text.upper()
        for b in BANNERS:
            up = re.sub(r"(?<![A-Z0-9])" + re.escape(b) + r"(?![A-Z0-9])", " ", up)
        # Only lowercase-insensitive removal; keep the original characters (Chinese) where nothing was removed.
        text = up if norm(up) != norm(text) else text
    else:
        text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip(" ,;:")
    return text
