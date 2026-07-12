"""Generate assets/thumbnail.png - dark-themed architecture diagram, 3:2.

Reproducible: `python assets/generate_thumbnail.py`. Requires Pillow.
"""
from __future__ import annotations

import os
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080  # 16:9
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

    f_title = _font(62, bold=True)
    f_sub = _font(30)
    f_box = _font(38, bold=True)
    f_body = _font(26)
    f_small = _font(22)
    f_tag = _font(22, bold=True)

    cx = W / 2

    # Title
    center(d, "Perseus Vault  x  AMD Instinct", cx, 46, f_title, TEXT)
    center(d, "Encrypted, local-first memory for AI agents - kept OFF the GPU",
           cx, 124, f_sub, MUTED)

    # --- GPU box (top) ---
    rbox(d, (360, 200, 1560, 352), 18, GPU_BOX, GPU_EDGE)
    center(d, "AMD Instinct MI300X", cx, 220, f_box, (255, 120, 130))
    center(d, "192 GB HBM3  -  LLM weights + KV cache  -  served via ROCm / Fireworks AI",
           cx, 276, f_body, TEXT)
    center(d, "100% of HBM serves tokens", cx, 314, f_small, GREEN)

    # --- Agent box (middle) ---
    rbox(d, (690, 456, 1230, 588), 18, AGENT_BOX, AGENT_EDGE)
    center(d, "Agent loop", cx, 474, f_box, (250, 200, 110))
    center(d, "recall(query)  ->  ground  ->  infer", cx, 532, f_body, TEXT)

    # --- Perseus Vault box (bottom) ---
    rbox(d, (360, 688, 1560, 878), 18, PV_BOX, PV_EDGE)
    center(d, "Perseus Vault  -  single Rust binary", cx, 706, f_box, (120, 210, 250))
    center(d, "SQLite + FTS5 (BM25) hybrid recall  -  no embeddings, no vector DB",
           cx, 762, f_body, TEXT)
    center(d, "AES-256-GCM  -  one portable file per agent  -  runs on host CPU",
           cx, 800, f_small, MUTED)
    center(d, "0 bytes of GPU HBM   -   ~85 MB RAM / agent (measured)", cx, 838,
           f_small, GREEN)

    # --- Arrows ---
    # GPU <-> Agent
    d.line((cx - 70, 352, cx - 70, 456), fill=AGENT_EDGE, width=4)
    d.polygon([(cx - 80, 451), (cx - 60, 451), (cx - 70, 466)], fill=AGENT_EDGE)  # down to agent
    d.line((cx + 70, 456, cx + 70, 352), fill=GPU_EDGE, width=4)
    d.polygon([(cx + 60, 214), (cx + 80, 214), (cx + 70, 200)], fill=GPU_EDGE)    # up to gpu
    d.text((cx - 250, 392), "grounding", font=f_small, fill=MUTED)
    d.text((cx + 90, 392), "tokens", font=f_small, fill=MUTED)

    # Agent <-> Perseus Vault
    d.line((cx, 588, cx, 688), fill=PV_EDGE, width=4)
    d.polygon([(cx - 10, 683), (cx + 10, 683), (cx, 696)], fill=PV_EDGE)
    d.text((cx + 20, 620), "remember() / recall() / decay()  (MCP, CPU)",
           font=f_small, fill=MUTED)

    # --- Economics callout (bottom strip) ---
    rbox(d, (360, 924, 1560, 1012), 14, PANEL, (48, 54, 61), width=2)
    center(d, "One MI300X holds Qwen2.5-72B on ONE card  ->  $0.143 / agent-hr",
           cx, 938, f_body, TEXT)
    center(d, "11.7x cheaper per agent than 2x H100   [MEASURED on real rented hardware]",
           cx, 976, f_small, GREEN)

    # data-source legend tags (top-left)
    d.text((34, 46), "MIT", font=f_tag, fill=MUTED)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thumbnail.png")
    img.save(out, "PNG")
    print("wrote", out, img.size)


if __name__ == "__main__":
    main()
