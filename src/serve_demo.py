"""
serve_demo - a tiny live web demo of Perseus Vault as an agent's memory layer.

Runs the CPU memory engine for real (store -> recall with a visible BM25+recency
trace -> decay) and, when a Fireworks API key is present, routes the grounded
prompt to an open-weight model served on AMD Instinct via the Fireworks AI API.
Without a key it degrades honestly to a memory-grounded composition (clearly
labelled -- it never fabricates an LLM answer), so the demo is always live.

Stdlib only (http.server) so it runs in the existing container with no new deps.
Env: PORT (default 8823), FIREWORKS_API_KEY (optional), FIREWORKS_MODEL (optional),
PERSEUS_VAULT_DB (optional; default in-memory).
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from perseus_vault_store import open_store  # noqa: E402
import agent_memory_demo as demo  # noqa: E402

PORT = int(os.environ.get("PORT", "8823"))
HAS_KEY = bool(os.environ.get("FIREWORKS_API_KEY"))
WARNING = ("Published-spec estimates for MI300X/ROCm economics. Real MI300X data "
           "pending AMD hardware access. The memory engine below runs live on CPU.")

SEED = [
    ("project", "Perseus Vault runs inference on AMD Instinct MI300X via ROCm and Fireworks AI."),
    ("architecture", "Perseus Vault keeps agent memory on the host CPU, so 100% of MI300X HBM3 serves tokens."),
    ("preference", "Always label GPU numbers as measured, published-spec, or projection - never fake a measurement."),
    ("economics", "One MI300X fits Llama-3.1-70B on a single card and serves ~20 concurrent agents at ~$0.13/agent-hr."),
    ("fact", "Perseus Vault recall is SQLite + FTS5 hybrid search - no embeddings, no external vector database."),
    ("contact", "The AMD Developer Cloud region for this project is amd-cloud-1."),
    ("decision", "We chose the Fireworks AI API to serve open-weight Llama-3.1-70B on AMD hardware."),
    ("security", "Perseus Vault encrypts memory at rest with AES-256-GCM in one portable file per agent."),
]

STORE = open_store(os.environ.get("PERSEUS_VAULT_DB", ":memory:"))
for _cat, _txt in SEED:
    STORE.remember(_cat, _txt)


def answer_for(message: str):
    hits = STORE.recall(message, k=4)
    recall = [{"text": h.memory.text, "category": h.memory.category,
               "rank": round(h.rank, 3)} for h in hits]
    grounding = "\n".join(f"- {h.memory.text}" for h in hits) or "(no relevant memories yet)"

    if HAS_KEY:
        prompt = (f"Using only these remembered facts:\n{grounding}\n\n"
                  f"Answer briefly and concretely: {message}")
        text = demo.infer(prompt)
        backend = f"AMD Instinct via Fireworks AI - {demo.FIREWORKS_MODEL} (live)"
    else:
        if hits:
            text = ("Grounded in memory: " + " ".join(h.memory.text for h in hits[:2]))
        else:
            text = ("I have no memory matching that yet - add one on the right, then ask again.")
        backend = ("offline memory-grounded composition (no LLM call) - set FIREWORKS_API_KEY "
                   "to route this same grounding to Llama-3.1-70B on AMD Instinct")

    return {"answer": text, "recall": recall, "backend": backend,
            "count": STORE.count(), "has_key": HAS_KEY}


PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Perseus Vault x AMD Instinct - live demo</title><style>
:root{--bg:#0d1117;--panel:#161b22;--edge:#30363d;--tx:#e6edf3;--mut:#8b949e;
--red:#ed1b2f;--cy:#38bdf8;--gr:#3fb950;--am:#f5b43c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);
font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
header{padding:18px 22px;border-bottom:1px solid var(--edge)}
h1{margin:0;font-size:20px}h1 b{color:var(--cy)}h1 i{color:var(--red);font-style:normal}
.warn{margin-top:8px;font-size:12px;color:var(--am)}
.wrap{display:grid;grid-template-columns:1.3fr 1fr;gap:16px;padding:16px 22px;max-width:1100px}
@media(max-width:820px){.wrap{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--edge);border-radius:12px;padding:14px}
.card h2{margin:0 0 10px;font-size:13px;color:var(--mut);text-transform:uppercase;letter-spacing:.5px}
#log{min-height:280px;max-height:52vh;overflow:auto;display:flex;flex-direction:column;gap:10px}
.msg{padding:9px 12px;border-radius:10px;max-width:90%}
.me{align-self:flex-end;background:#1f6feb22;border:1px solid #1f6feb55}
.ai{align-self:flex-start;background:#0f2b34;border:1px solid var(--cy)}
.ai .bk{display:block;margin-top:6px;font-size:11px;color:var(--mut)}
.row{display:flex;gap:8px;margin-top:12px}
input,button{font:inherit}
input[type=text]{flex:1;background:#0b0f14;border:1px solid var(--edge);color:var(--tx);
border-radius:8px;padding:9px 11px}
button{background:var(--cy);color:#04222e;border:0;border-radius:8px;padding:9px 14px;
font-weight:600;cursor:pointer}button:hover{filter:brightness(1.1)}
.trace{display:flex;flex-direction:column;gap:8px}
.hit{background:#0b0f14;border:1px solid var(--edge);border-left:3px solid var(--cy);
border-radius:6px;padding:8px 10px;font-size:13px}
.hit .m{color:var(--mut);font-size:11px;margin-top:3px}
.stat{font-size:13px;color:var(--mut);margin-bottom:8px}
.stat b{color:var(--tx)}
.pill{display:inline-block;font-size:11px;padding:2px 8px;border-radius:20px;margin-left:6px}
.pill.on{background:#0f3320;color:var(--gr);border:1px solid var(--gr)}
.pill.off{background:#3a1f10;color:var(--am);border:1px solid var(--am)}
.mini{display:flex;gap:6px;margin-top:10px}.mini input{padding:6px 8px;font-size:13px}
.mini button{padding:6px 10px;font-size:13px;background:var(--gr);color:#052}
a{color:var(--cy)}.foot{padding:6px 22px 22px;font-size:12px;color:var(--mut)}
</style></head><body>
<header><h1><b>Perseus Vault</b> <i>x</i> AMD Instinct - live memory demo</h1>
<div class=warn>WARNING: __WARN__</div></header>
<div class=wrap>
 <div class=card>
  <h2>Agent chat (recall -> infer)</h2>
  <div id=log></div>
  <div class=row>
   <input id=q type=text placeholder="Ask e.g. 'Where does inference run and how are GPU numbers labelled?'"
     onkeydown="if(event.key==='Enter')send()">
   <button onclick=send()>Send</button>
  </div>
 </div>
 <div class=card>
  <h2>Perseus Vault - recall trace</h2>
  <div class=stat>Memories in store: <b id=cnt>-</b>
   <span id=bk class="pill off">inference: ...</span></div>
  <div id=trace class=trace><div class=hit>Ask something - the memories Perseus Vault
   pulls (BM25 + recency, on CPU) appear here, ranked.</div></div>
  <div class=mini>
   <input id=nm type=text placeholder="teach the agent a new fact...">
   <button onclick=remember()>Remember</button>
  </div>
 </div>
</div>
<div class=foot>Memory engine runs live on CPU (SQLite/FTS5, 0 GPU HBM). Inference runs on
 AMD Instinct via Fireworks AI when a key is configured. Repo:
 <a href="https://github.com/tcconnally/perseus-amd-act-ii">tcconnally/perseus-amd-act-ii</a></div>
<script>
const log=document.getElementById('log'),trace=document.getElementById('trace'),
 cnt=document.getElementById('cnt'),bk=document.getElementById('bk');
function add(cls,html){const d=document.createElement('div');d.className='msg '+cls;
 d.innerHTML=html;log.appendChild(d);log.scrollTop=log.scrollHeight;}
function setbk(t,on){bk.textContent='inference: '+t;bk.className='pill '+(on?'on':'off');}
async function send(){const q=document.getElementById('q');const m=q.value.trim();if(!m)return;
 q.value='';add('me',esc(m));add('ai','<i>...recalling + answering...</i>');
 const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({message:m})}).then(r=>r.json());
 log.lastChild.innerHTML=esc(r.answer)+'<span class=bk>'+esc(r.backend)+'</span>';
 cnt.textContent=r.count;setbk(r.has_key?'AMD Instinct (Fireworks, live)':'offline (no key)',r.has_key);
 trace.innerHTML=r.recall.length?r.recall.map(h=>'<div class=hit>'+esc(h.text)+
  '<div class=m>['+esc(h.category)+']  fused rank '+h.rank+'</div></div>').join(''):
  '<div class=hit>no matching memories</div>';}
async function remember(){const n=document.getElementById('nm');const t=n.value.trim();if(!t)return;
 n.value='';const r=await fetch('/api/remember',{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify({category:'user',text:t})})
  .then(r=>r.json());cnt.textContent=r.count;add('ai','<i>remembered.</i> store now has '+r.count+' memories.');}
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
fetch('/api/state').then(r=>r.json()).then(s=>{cnt.textContent=s.count;
 setbk(s.has_key?'AMD Instinct (Fireworks, live)':'offline (no key)',s.has_key);});
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _body(self):
        n = int(self.headers.get("Content-Length", "0") or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, "ok", "text/plain")
        if self.path == "/api/state":
            return self._send(200, json.dumps({"count": STORE.count(), "has_key": HAS_KEY}))
        if self.path in ("/", "/index.html"):
            return self._send(200, PAGE.replace("__WARN__", WARNING), "text/html; charset=utf-8")
        return self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path == "/api/chat":
            msg = str(self._body().get("message", ""))[:500]
            return self._send(200, json.dumps(answer_for(msg)))
        if self.path == "/api/remember":
            d = self._body()
            STORE.remember(str(d.get("category", "user"))[:40], str(d.get("text", ""))[:500])
            return self._send(200, json.dumps({"count": STORE.count()}))
        return self._send(404, "not found", "text/plain")


if __name__ == "__main__":
    print(f"Perseus Vault x AMD demo on :{PORT}  (Fireworks key: "
          f"{'present -> live AMD inference' if HAS_KEY else 'absent -> offline composition'})")
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
