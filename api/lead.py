import datetime, ipaddress, json, re, socket
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler
from urllib.parse import urljoin, urlparse
import requests
from google import genai
from google.genai import types

_client = genai.Client()  # reads GEMINI_API_KEY automatically


def ask(system, user, tokens=2000):
    r = _client.models.generate_content(
        model="gemini-3.8-flash", contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system, response_mime_type="application/json",
            max_output_tokens=tokens, thinking_config=types.ThinkingConfig(thinking_budget=0)))
    return json.loads(r.text)


UA = {"user-agent": "LeadResearchBot/1.0 (student project)"}
HINT = re.compile(r"(about|company|product|service|solution|platform|customer|case|contact|industr)", re.I)
SYSTEM = """You qualify B2B leads against an ideal customer profile (ICP), using only the page text supplied.
Rules:
- Never invent facts, people, names, job titles, emails, funding or technology. Use only supplied page text and supplied notes.
- what_they_do: 1 to 2 plain sentences from the pages.
- fit: "Strong", "Possible", "Weak" or "Unclear". Use "Unclear" when pages do not give enough evidence.
- For each ICP criterion give status "match", "mismatch" or "unknown", the evidence_url (one of the supplied page urls) and a short exact quote (max 20 words) copied from that page. If there is no evidence, status must be "unknown" with empty quote.
- outreach_angle: one sentence linking something the business actually does to the seller's offer.
- draft_message: under 90 words, addressed to "[Name]" as a placeholder, references only supported facts, no flattery about unverified claims.
- unknowns: what a human should check before sending.
Return ONLY JSON: {"what_they_do":"","fit":"","criteria":[{"criterion":"","status":"","evidence_url":"","quote":""}],"outreach_angle":"","draft_message":"","unknowns":[""]}"""


def is_public(url):
    try:
        u = urlparse(url)
        if u.scheme not in ("http", "https") or not u.hostname:
            return False
        return all(ipaddress.ip_address(i[4][0]).is_global for i in socket.getaddrinfo(u.hostname, None))
    except Exception:
        return False


def get(url):
    r = requests.get(url, headers=UA, timeout=6)
    if not is_public(r.url):
        raise ValueError("Redirected to a non-public address")
    return {"url": url, "final": r.url, "status": r.status_code, "html": r.text if "text/html" in r.headers.get("content-type", "") else ""}


def safe_get(u):
    try:
        return get(u)
    except Exception:
        return None


def disallowed(txt):
    rules, on = [], False
    for line in txt.splitlines():
        l = line.split("#")[0].strip()
        if re.match(r"user-agent:\s*\*", l, re.I):
            on = True
        elif re.match(r"user-agent:", l, re.I):
            on = False
        elif on and l.lower().startswith("disallow:"):
            v = l.split(":", 1)[1].strip()
            if v:
                rules.append(v)
    return rules


def parse(p):
    h = p["html"]
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>|<nav[\s\S]*?</nav>|<footer[\s\S]*?</footer>", " ", h, flags=re.I))).strip()
    desc = re.search(r'<meta[^>]+name=["\']description["\'][^>]*content=["\']([^"\']*)', h, re.I)
    t = re.search(r"<title[^>]*>([\s\S]*?)</title>", h, re.I)
    return {"url": p["final"], "title": t.group(1).strip() if t else "", "desc": desc.group(1) if desc else "", "text": text[:2500]}


def research(lead):
    url = lead["url"]
    if not is_public(url):
        raise ValueError("Invalid or non-public URL")
    start = urlparse(url)
    blocked = []
    try:
        blocked = disallowed(requests.get(f"{start.scheme}://{start.netloc}/robots.txt", headers=UA, timeout=4).text)
    except Exception:
        pass
    ok = lambda u: not any(urlparse(u).path.startswith(x) for x in blocked)
    if not ok(url):
        raise ValueError("robots.txt disallows this page")
    first = get(url)
    hrefs = []
    for m in re.finditer(r'<a\s[^>]*href=["\']([^"\'#]+)["\']', first["html"], re.I):
        u = urlparse(urljoin(first["final"], m.group(1)))
        h = u._replace(fragment="").geturl()
        if u.hostname == start.hostname and HINT.search(u.path) and ok(h) and h not in hrefs:
            hrefs.append(h)
    extra = [safe_get(u) for u in hrefs[:2]]
    return [parse(p) for p in [first] + extra if p and p["status"] < 400 and p["html"]]


def process(lead, icp, offer):
    try:
        pages = research(lead)
        if not pages:
            raise ValueError("No readable pages found")
        user = json.dumps({"icp": icp, "sellerOffer": offer, "company": lead.get("name"), "notes": lead.get("notes", ""), "pages": pages})
        ai = ask(SYSTEM, user, 1800)
        urls = {p["url"] for p in pages}
        crit = []
        for c in ai.get("criteria", []):
            page = next((p for p in pages if p["url"] == c.get("evidence_url")), None)
            q = " ".join((c.get("quote") or "").lower().split())
            if page and q and q in page["text"].lower():
                crit.append({**c, "verified": True})
            else:  # unverified evidence is removed, never shown as fact
                crit.append({**c, "evidence_url": c.get("evidence_url") if c.get("evidence_url") in urls else "", "quote": "", "status": "unknown", "verified": False})
        ai["criteria"] = crit
        return {"name": lead.get("name"), "url": lead["url"], "pages": [p["url"] for p in pages], **ai}
    except Exception as e:
        return {"name": lead.get("name"), "url": lead.get("url"), "error": str(e)}


class handler(BaseHTTPRequestHandler):
    def send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        try:
            d = json.loads(self.rfile.read(int(self.headers.get("content-length", 0) or 0)) or b"{}")
        except ValueError:
            return self.send(400, {"error": "Invalid JSON"})
        leads = d.get("leads")
        if not d.get("consent"):
            return self.send(400, {"error": "Confirm the lead details were supplied or are public."})
        if not d.get("icp") or not isinstance(leads, list) or not leads:
            return self.send(400, {"error": "Add an ICP and at least one lead."})
        with ThreadPoolExecutor(5) as ex:
            results = list(ex.map(lambda l: process(l, d["icp"], d.get("offer", "")), leads[:5]))
        self.send(200, {"date": datetime.date.today().isoformat(), "results": results})