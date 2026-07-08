"""Generate assets/thumbnail.png - dark-themed architecture diagram, 3:2.

Reproducible: `python assets/generate_thumbnail.py`. Requires Pillow.
"""
from __future__ import annotations

import os
from PIL import Image, ImageDraw, ImageFont

W, H = 1500, 1000  # 3:2
BG = (13, 17, 23)          # GitHub dark
PANEL = (22, 27, 34)
AMD_RED = (237, 27, 47)
GPU_BOX = (40, 20, 22)
GPU_EDGE = (237, 27, 47)
PV_BOX = (16, 30, 40)
PV_EDGE = (56, 189, 248)   # cyan
AGENT_BOX = (30, 27, 16)
AGENT_EDGE = (245, 180, 60)
TEXT = (230, 237, 243)
MUTED = (139, 148, 158)
GREEN = (63, 185, 80)


def _font(size: int, bold: bool = False):
    candidates = [
        ("arialbd.ttf" if bold else "arial.ttf"),
        os.path.join("C:\\", "Windows", "Fonts", "arialbd.ttf" if bold else "arial.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            continue
    return ImageFont.load_default()


def rbox(d, xy, radius, fill, outline, width=3):
    d.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def center(d, text, cx, y, font, fill):
    w = d.textbbox((0, 0), text, font=font)[2]
    d.text((cx - w / 2, y), text, font=font, fill=fill)


def main():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    f_title = _font(54, bold=True)
    f_sub = _font(28)
    f_box = _font(34, bold=True)
    f_body = _font(24)
    f_small = _font(21)
    f_tag = _font(20, bold=True)

    # Title
    center(d, "Perseus Vault  x  AMD Instinct", W / 2, 40, f_title, TEXT)
    center(d, "Encrypted, local-first memory for AI agents - kept OFF the GPU",
           W / 2, 108, f_sub, MUTED)

    cx = W / 2

    # --- GPU box (top) ---
    rbox(d, (300, 175, 1200, 320), 18, GPU_BOX, GPU_EDGE)
    center(d, "AMD Instinct MI300X", cx, 192, f_box, (255, 120, 130))
    center(d, "192 GB HBM3  -  LLM weights + KV cache  -  served via ROCm / Fireworks AI",
           cx, 244, f_body, TEXT)
    center(d, "100% of HBM serves tokens", cx, 280, f_small, GREEN)

    # --- Agent box (middle) ---
    rbox(d, (500, 420, 1000, 545), 18, AGENT_BOX, AGENT_EDGE)
    center(d, "Agent loop", cx, 438, f_box, (250, 200, 110))
    center(d, "recall(query)  ->  ground  ->  infer", cx, 492, f_body, TEXT)

    # --- Perseus Vault box (bottom) ---
    rbox(d, (300, 645, 1200, 830), 18, PV_BOX, PV_EDGE)
    center(d, "Perseus Vault  -  single Rust binary", cx, 662, f_box, (120, 210, 250))
    center(d, "SQLite + FTS5 (BM25) hybrid recall  -  no embeddings, no vector DB",
           cx, 716, f_body, TEXT)
    center(d, "AES-256-GCM  -  one portable file per agent  -  runs on host CPU",
           cx, 752, f_small, MUTED)
    center(d, "0 bytes of GPU HBM   -   ~85 MB RAM / agent (measured)", cx, 790,
           f_small, GREEN)

    # --- Arrows ---
    # GPU <-> Agent
    d.line((cx - 60, 320, cx - 60, 420), fill=AGENT_EDGE, width=4)
    d.polygon([(cx - 70, 415), (cx - 50, 415), (cx - 60, 428)], fill=AGENT_EDGE)  # down to agent
    d.line((cx + 60, 420, cx + 60, 320), fill=GPU_EDGE, width=4)
    d.polygon([(cx + 50, 185), (cx + 70, 185), (cx + 60, 172)], fill=GPU_EDGE)    # up to gpu
    d.text((cx - 235, 355), "grounding", font=f_small, fill=MUTED)
    d.text((cx + 80, 355), "tokens", font=f_small, fill=MUTED)

    # Agent <-> Perseus Vault
    d.line((cx, 545, cx, 645), fill=PV_EDGE, width=4)
    d.polygon([(cx - 10, 640), (cx + 10, 640), (cx, 653)], fill=PV_EDGE)
    d.text((cx + 20, 578), "remember() / recall() / decay()  (MCP, CPU)",
           font=f_small, fill=MUTED)

    # --- Economics callout (bottom strip) ---
    rbox(d, (300, 875, 1200, 960), 14, PANEL, (48, 54, 61), width=2)
    center(d, "One MI300X fits Llama-3.1-70B on ONE card  ->  ~$0.13 / agent-hr",
           cx, 888, f_body, TEXT)
    center(d, "~7.8x cheaper than H100 for this workload   [projection - published-spec inputs]",
           cx, 924, f_small, (255, 170, 120))

    # data-source legend tags (top-right)
    d.text((30, 40), "MIT", font=f_tag, fill=MUTED)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thumbnail.png")
    img.save(out, "PNG")
    print("wrote", out, img.size)


if __name__ == "__main__":
    main()
