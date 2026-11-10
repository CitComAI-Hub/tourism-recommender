"""Temporary web tool to audit a random sample of the catalog by hand.

Stratified sample: half from the 'sourced' subset (Wikidata / GVA inventory, the
one that would be published first) and half from the rest. For each POI the
reviewer answers whether it is correct and, if not, which fields are wrong.
Answers are saved after every click in data/audit/audit.json; /report gives the
error rate per stratum with a 95% Wilson interval.

    python research/audit_app.py [--n 50] [--port 8790]
"""
from __future__ import annotations

import argparse
import html
import json
import math
import random
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from pmv_turismo.publish import is_sourced  # noqa: E402

CATALOG = REPO / "data" / "processed" / "poi.jsonld"
OUT = REPO / "data" / "audit" / "audit.json"
FIELDS = ["name", "location", "category", "description", "image", "openingHours", "exists"]
LABELS = {"name": "Nombre", "location": "Ubicación", "category": "Categoría", "description": "Descripción",
          "image": "Imagen", "openingHours": "Horario", "exists": "No es un lugar turístico / no existe"}


def sample(n: int, seed: int = 20261110) -> list[str]:
    pois = json.loads(CATALOG.read_text(encoding="utf-8"))
    rng = random.Random(seed)
    sourced = [p["id"] for p in pois if is_sourced(p)]
    rest = [p["id"] for p in pois if not is_sourced(p)]
    pick = rng.sample(sourced, n // 2) + rng.sample(rest, n - n // 2)
    rng.shuffle(pick)
    return pick


def load_state(n: int) -> dict:
    if OUT.exists():
        return json.loads(OUT.read_text(encoding="utf-8"))
    return {"sample": sample(n), "answers": {}}


def save(state: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def wilson(k: int, n: int) -> tuple[float, float]:
    if not n:
        return 0.0, 1.0
    z, p = 1.96, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


PAGE = """<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Auditoría POIs</title>
<style>
:root{{--bg:#fafaf7;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2e1da;--ok:#1f7a4d;--ko:#b3261e}}
body{{margin:0;font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg)}}
main{{max-width:1100px;margin:0 auto;padding:16px}}
header{{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} @media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}
iframe{{width:100%;height:320px;border:1px solid var(--line);border-radius:8px}}
img{{max-width:100%;max-height:320px;border-radius:8px;border:1px solid var(--line)}}
dl{{display:grid;grid-template-columns:140px 1fr;gap:4px 12px;margin:0}} dt{{color:var(--mut)}} dd{{margin:0;overflow-wrap:anywhere}}
.tag{{display:inline-block;padding:0 6px;border-radius:4px;background:#efe9d8;font-size:12px}}
form{{margin-top:16px;padding:12px;border:1px solid var(--line);border-radius:8px;background:#fff}}
button{{font:inherit;padding:10px 22px;border-radius:999px;border:0;color:#fff;cursor:pointer}}
.ok{{background:var(--ok)}} .ko{{background:var(--ko)}} label{{margin-right:14px;white-space:nowrap}}
textarea{{width:100%;box-sizing:border-box;font:inherit;margin:8px 0}} a{{color:#1a5fb4}}
nav a{{margin-left:12px}}
</style></head><body><main>{body}</main>
<script>document.addEventListener('keydown',e=>{{if(e.target.tagName==='TEXTAREA')return;
if(e.key==='y'){{document.getElementById('yes')?.click()}}}});</script></body></html>"""


def poi_page(state: dict, idx: int) -> str:
    pois = {p["id"]: p for p in json.loads(CATALOG.read_text(encoding="utf-8"))}
    pid = state["sample"][idx]
    p = pois[pid]
    lon, lat = p["location"]["coordinates"]
    d = 0.003
    osm = (f"https://www.openstreetmap.org/export/embed.html?bbox={lon-d},{lat-d},{lon+d},{lat+d}"
           f"&layer=mapnik&marker={lat},{lon}")
    e = lambda x: html.escape(str(x)) if x not in (None, "") else "<span class=tag>vacío</span>"
    links = "".join(f'<a href="{html.escape(u)}" target="_blank" rel="noopener">{html.escape(u)}</a><br>'
                    for u in (p.get("refSeeAlso") or []))
    prev = state["answers"].get(pid, {})
    checks = "".join(
        f'<label><input type="checkbox" name="wrong" value="{f}"{" checked" if f in prev.get("wrong", []) else ""}> {LABELS[f]}</label>'
        for f in FIELDS)
    hours = p.get("openingHours")
    done = len(state["answers"])
    body = f"""
<header><h1 style="margin:0">{e(p.get('name'))}</h1>
<span>POI {idx + 1} de {len(state['sample'])} · revisados {done} <nav><a href="/report">informe</a>
{f'<a href="/poi/{idx - 1}">← anterior</a>' if idx else ''}<a href="/poi/{min(idx + 1, len(state['sample']) - 1)}">siguiente →</a></nav></span></header>
<p><span class=tag>{'con fuente (se publicaría)' if is_sourced(p) else 'sin fuente fuerte'}</span>
{' <span class=tag>descripción generada</span>' if p.get('descriptionGenerated') else ''}
{' <span class=tag>' + e(p.get('heritageDesignation')) + '</span>' if p.get('heritageDesignation') else ''}</p>
<div class=grid>
<div><iframe src="{osm}" loading="lazy" title="mapa"></iframe>
<p><a href="https://www.google.com/maps/search/?api=1&query={lat},{lon}" target="_blank" rel="noopener">Ver en Google Maps / Street View</a> ·
<a href="{html.escape(p.get('source') or '#')}" target="_blank" rel="noopener">objeto OSM</a></p></div>
<div>{f'<img src="{html.escape(p["image"])}" alt="">' if p.get('image') else '<p class=tag>sin imagen</p>'}</div>
</div>
<dl>
<dt>Categoría</dt><dd>{e(', '.join(p.get('category') or []))} ({e(p.get('subcategory'))})</dd>
<dt>Descripción</dt><dd>{e(p.get('description'))}</dd>
<dt>Dirección</dt><dd>{e((p.get('address') or {}).get('streetAddress'))}</dd>
<dt>Horario</dt><dd>{e(hours)}{' <span class=tag>estimado</span>' if p.get('openingHoursEstimated') else ''}</dd>
<dt>Duración</dt><dd>{e(p.get('duration'))} min{' <span class=tag>estimada</span>' if p.get('durationEstimated') else ''}</dd>
<dt>Enlaces</dt><dd>{links or e(None)}</dd>
<dt>Fuentes</dt><dd>{e(p.get('dataProvider'))}</dd>
</dl>
<form method="post" action="/answer/{idx}">
<p style="margin-top:0"><b>¿Es correcto?</b> Si no, marca qué falla (los campos marcados como estimados no cuentan como error).</p>
<div>{checks}</div>
<textarea name="comment" rows="2" placeholder="Comentario (opcional)">{html.escape(prev.get('comment', ''))}</textarea>
<button id="yes" class=ok name="ok" value="yes">Sí, correcto (y)</button>
<button class=ko name="ok" value="no">No, hay errores</button>
{f' <span>Respuesta guardada: <b>{"sí" if prev.get("ok") else "no"}</b></span>' if prev else ''}
</form>"""
    return PAGE.format(body=body)


def report(state: dict) -> str:
    pois = {p["id"]: p for p in json.loads(CATALOG.read_text(encoding="utf-8"))}
    rows = []
    for label, pred in (("Con fuente (se publicaría)", is_sourced), ("Sin fuente fuerte", lambda p: not is_sourced(p)),
                        ("Todos", lambda p: True)):
        ans = [a for pid, a in state["answers"].items() if pred(pois[pid])]
        bad = sum(1 for a in ans if not a["ok"])
        lo, hi = wilson(bad, len(ans))
        rows.append(f"<tr><td>{label}</td><td>{len(ans)}</td><td>{bad}</td>"
                    f"<td>{(bad / len(ans) if ans else 0):.0%}</td><td>{lo:.0%} – {hi:.0%}</td></tr>")
    fields = {f: sum(1 for a in state["answers"].values() if f in a.get("wrong", [])) for f in FIELDS}
    pending = [i for i, pid in enumerate(state["sample"]) if pid not in state["answers"]]
    body = f"""<header><h1 style="margin:0">Informe</h1><nav><a href="/poi/{pending[0] if pending else 0}">volver</a></nav></header>
<p>Revisados {len(state['answers'])} de {len(state['sample'])}.</p>
<table cellpadding=6><tr><th align=left>Estrato</th><th>Revisados</th><th>Con errores</th><th>Tasa</th><th>IC 95% (Wilson)</th></tr>{''.join(rows)}</table>
<h2>Errores por campo</h2><ul>{''.join(f'<li>{LABELS[f]}: {n}</li>' for f, n in fields.items())}</ul>"""
    return PAGE.format(body=body)


class Handler(BaseHTTPRequestHandler):
    state: dict = {}

    def _send(self, code: int, body: str = "", location: str | None = None) -> None:
        self.send_response(code)
        if location:
            self.send_header("Location", location)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode())

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            pending = [i for i, pid in enumerate(self.state["sample"]) if pid not in self.state["answers"]]
            return self._send(302, location=f"/poi/{pending[0]}" if pending else "/report")
        if path == "/report":
            return self._send(200, report(self.state))
        if path.startswith("/poi/"):
            idx = int(path.rsplit("/", 1)[1])
            if 0 <= idx < len(self.state["sample"]):
                return self._send(200, poi_page(self.state, idx))
        self._send(404, "not found")

    def do_POST(self) -> None:
        idx = int(urlparse(self.path).path.rsplit("/", 1)[1])
        form = parse_qs(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode())
        pid = self.state["sample"][idx]
        ok = form.get("ok", ["yes"])[0] == "yes"
        self.state["answers"][pid] = {"ok": ok, "wrong": [] if ok else form.get("wrong", []),
                                      "comment": form.get("comment", [""])[0]}
        save(self.state)
        nxt = next((i for i in range(idx + 1, len(self.state["sample"]))
                    if self.state["sample"][i] not in self.state["answers"]), None)
        self._send(303, location=f"/poi/{nxt}" if nxt is not None else "/report")

    def log_message(self, *args) -> None:  # quiet
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--port", type=int, default=8790)
    args = ap.parse_args()
    Handler.state = load_state(args.n)
    save(Handler.state)
    print(f"Audit: http://127.0.0.1:{args.port}/  ({len(Handler.state['answers'])}/{len(Handler.state['sample'])} done)")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
