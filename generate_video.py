"""
Generate the AMD Act II demo video for Perseus Vault.

Fully automated, no human voice: narration is synthesized offline with
piper-tts (en_US-lessac-medium), then terminal-style PIL frames are rendered
and muxed with the voiceover via ffmpeg.

Adapted from the proven pipeline in
github.com/tcconnally/perseus-vault-hackathon/generate_video.py.

HONESTY: every benchmark/cost frame renders an on-screen data_source badge
(measured vs projection); the closing frame states the caveat. The tables here
match BENCHMARKS.md exactly.
"""
from PIL import Image, ImageDraw, ImageFont
import subprocess
import os
import shutil
import glob

from narration import NARRATION

# ---------------------------------------------------------------- config
WIDTH, HEIGHT = 1920, 1080
BG_COLOR = (18, 18, 18)              # #121212
PROMPT_COLOR = (0, 255, 100)        # green
OUTPUT_COLOR = (200, 200, 200)      # light gray
TITLE_COLOR = (100, 180, 255)       # blue
HIGHLIGHT_COLOR = (255, 200, 50)    # gold
DIM_COLOR = (120, 120, 120)
FPS = 24

HERE = os.path.dirname(os.path.abspath(__file__))
FRAME_DIR = os.path.join(HERE, "_frames")
AUDIO_DIR = os.path.join(HERE, "audio")
VOICEOVER_PATH = os.path.join(HERE, "voiceover.mp3")
VIDEO_PATH = os.path.join(HERE, "demo_video.mp4")
PIPER_VOICE = "en_US-lessac-medium"

TAIL_PAD = 0.9   # seconds of silence held on each frame after its narration

# ---------------------------------------------------------------- fonts
font_paths = [
    r"C:\Windows\Fonts\DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    r"C:\Windows\Fonts\consola.ttf",
]
_fp = next((p for p in font_paths if os.path.exists(p)), None)
if _fp:
    font = ImageFont.truetype(_fp, 32)
    font_small = ImageFont.truetype(_fp, 24)
    font_title = ImageFont.truetype(_fp, 48)
    font_badge = ImageFont.truetype(_fp, 26)
else:
    font = font_small = font_title = font_badge = ImageFont.load_default()


def make_frame(text_lines, y_offset=100, badge=None):
    """Render one terminal-style frame. `badge` (str) stamps a data_source footer."""
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # terminal window chrome
    draw.rectangle([20, 20, WIDTH - 20, HEIGHT - 20], outline=(60, 60, 60), width=2)
    draw.rectangle([20, 20, WIDTH - 20, 60], fill=(40, 40, 40))
    draw.text((40, 26), "● ● ●  perseus-vault — MI300X — 192×48",
              fill=(150, 150, 150), font=font_small)

    y = y_offset
    for line in text_lines:
        text, color = line if isinstance(line, tuple) else (line, OUTPUT_COLOR)
        if text.startswith("# "):
            draw.text((40, y), text[2:], fill=TITLE_COLOR, font=font_title)
            y += 60
        elif text.startswith("$ "):
            draw.text((40, y), text, fill=PROMPT_COLOR, font=font)
            y += 42
        elif text.startswith(">>> "):
            draw.text((40, y), text, fill=HIGHLIGHT_COLOR, font=font)
            y += 42
        else:
            draw.text((40, y), text, fill=color, font=font)
            y += 42

    if badge:
        # persistent on-frame honesty badge, bottom-centre (auto-sized to text)
        tb = draw.textbbox((0, 0), badge, font=font_badge)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad_x, pad_y = 28, 13
        bw, bh = tw + 2 * pad_x, th + 2 * pad_y
        bx = (WIDTH - bw) // 2
        by = HEIGHT - 95
        draw.rectangle([bx, by, bx + bw, by + bh], fill=(48, 40, 12), outline=HIGHLIGHT_COLOR, width=2)
        draw.text((bx + pad_x - tb[0], by + pad_y - tb[1]), badge, fill=HIGHLIGHT_COLOR, font=font_badge)

    return img


# ---------------------------------------------------------------- scenes
# Each scene: dict(lines=[...], badge=bool). Duration is driven by its narration
# segment length + TAIL_PAD so audio and visuals stay in sync.
scenes = [
    # 1 — title
    dict(lines=[
        ("", None), ("", None),
        ("# Perseus Vault", TITLE_COLOR),
        ("", None),
        ("  Encrypted Agent Memory — off the GPU, on AMD Instinct MI300X", HIGHLIGHT_COLOR),
    ]),
    # 2 — problem + what it is
    dict(lines=[
        ("# Agents forget when the session ends.", TITLE_COLOR),
        ("  A bolted-on vector DB eats the HBM you bought for inference.", DIM_COLOR),
        ("", None),
        ("# Perseus Vault", TITLE_COLOR),
        ("  • Single Rust binary · SQLite FTS5 recall · 56 MCP tools", OUTPUT_COLOR),
        ("  • Memory lives on the host CPU — 0 bytes of GPU HBM", OUTPUT_COLOR),
        ("", None),
        ("  Two markets: teams bleeding tokens, + regulated teams", HIGHLIGHT_COLOR),
        ("  who can't put memory in the cloud at all.", HIGHLIGHT_COLOR),
    ]),
    # 3 — store / recall / decay across a session boundary
    dict(lines=[
        ("$ perseus-vault store \"deploy target: MI300X, ROCm 7\"", PROMPT_COLOR),
        ("[store] entry a1b2 committed  (fts5 indexed)", OUTPUT_COLOR),
        ("$ perseus-vault decay --sweep    # age out unused entries", PROMPT_COLOR),
        ("[decay] 3 stale entries demoted", OUTPUT_COLOR),
        ("", None),
        ("--- New Session (context window reset) ---", TITLE_COLOR),
        ("$ perseus-vault recall \"where am I deploying?\"", PROMPT_COLOR),
        (">>> recalled across session boundary:", HIGHLIGHT_COLOR),
        (">>> \"deploy target: MI300X, ROCm 7\"", HIGHLIGHT_COLOR),
    ]),
    # 4 — MEASURED on a real MI300X (the hero) — docs/BENCHMARKS.md §3a
    dict(badge="data_source: measured (real AMD Instinct MI300X)", lines=[
        ("# Measured on a real MI300X: recall vs a saturated GPU", TITLE_COLOR),
        ("  Qwen2.5-72B bf16 on vLLM 0.19.1 / ROCm 7.13 — real serving load", DIM_COLOR),
        ("", None),
        ("  MI300X state                  Recall p50 (host CPU)", HIGHLIGHT_COLOR),
        ("  ------------                  --------------------", DIM_COLOR),
        ("  idle                          18.7 ms", OUTPUT_COLOR),
        ("  saturated serving the 72B     18.8 ms  (±0.6% median, 6 runs)", (120, 255, 160)),
        ("", None),
        ("  recall held flat while the GPU served flat-out", HIGHLIGHT_COLOR),
        ("  → CPU memory layer steals ~0 inference cycles", DIM_COLOR),
    ]),
    # 5 — recall throughput + footprint — docs/BENCHMARKS.md §1/§2 (AMD CPU reference)
    dict(badge="data_source: measured (AMD CPU reference)", lines=[
        ("# Recall throughput & footprint", TITLE_COLOR),
        ("", None),
        ("  Entries    Recall ops/s   p50 (ms)   DB file", HIGHLIGHT_COLOR),
        ("  --------   ------------   --------   -------", DIM_COLOR),
        ("    1,000           5,081       0.20   0.31 MB", OUTPUT_COLOR),
        ("   10,000             953       1.14   2.61 MB", OUTPUT_COLOR),
        ("  100,000              92      11.87  25.95 MB", OUTPUT_COLOR),
        ("", None),
        ("  BM25 + recency over SQLite FTS5 — on the HOST CPU,", DIM_COLOR),
        ("  0 bytes of GPU HBM.  Reproduce: python src/benchmark.py", DIM_COLOR),
    ]),
    # 6 — cost economics — MEASURED on BOTH vendors (§3a MI300X + §3b 2×H100)
    dict(badge="data_source: measured (MI300X + 2xH100, same model, same vLLM)", lines=[
        ("# Cost economics — measured on BOTH vendors", TITLE_COLOR),
        ("", None),
        ("  Qwen2.5-72B bf16 · vLLM 0.19.1 · 8K-token agents", DIM_COLOR),
        ("", None),
        ("  1x MI300X     holds it, 38 GiB spare   15.3 agents   $0.143/agent-hr", (120, 255, 160)),
        ("  1x H100       CANNOT LOAD THE MODEL      —              —", OUTPUT_COLOR),
        ("  2x H100       best case (eager, 97%)    5.0 agents   $1.68/agent-hr", OUTPUT_COLOR),
        ("", None),
        (">>> 11.7x cheaper per agent — measured, not projected", HIGHLIGHT_COLOR),
        ("  ($0.92 vs $3.42 per 1M output tokens · A100 row remains projection)", DIM_COLOR),
    ]),
    # 7 — closing
    dict(lines=[
        ("", None),
        ("# Perseus Vault × AMD Instinct MI300X", TITLE_COLOR),
        ("", None),
        ("  github.com/tcconnally/perseus-amd-act-ii", HIGHLIGHT_COLOR),
        ("", None),
        ("  Agents, $/agent-hr, recall-under-load, and the 2xH100", (255, 140, 90)),
        ("  comparison = ALL MEASURED.  Reproduce: src/amd_live_benchmark.py", (255, 140, 90)),
    ]),
]

assert len(scenes) == len(NARRATION), "scene/narration count mismatch"


def ffprobe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def main():
    for d in (FRAME_DIR, AUDIO_DIR):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)

    # 1) synthesize narration per scene, measure, pad, concat -> voiceover.mp3
    padded_segs = []
    durations = []
    for i, text in enumerate(NARRATION):
        raw = os.path.join(AUDIO_DIR, f"seg_{i}.wav")
        subprocess.run(["python", "-m", "piper", "--model", PIPER_VOICE,
                        "--output_file", raw], input=text, text=True,
                       check=True, capture_output=True)
        adur = ffprobe_duration(raw)
        scene_dur = round(adur + TAIL_PAD, 3)
        durations.append(scene_dur)
        padded = os.path.join(AUDIO_DIR, f"pad_{i}.wav")
        # pad silence at the tail so segment length == scene_dur
        subprocess.run(["ffmpeg", "-y", "-i", raw, "-af",
                        f"apad=whole_dur={scene_dur}", padded],
                       check=True, capture_output=True)
        padded_segs.append(padded)
        print(f"  scene {i+1}: narration {adur:.2f}s -> frame {scene_dur:.2f}s")

    concat_list = os.path.join(AUDIO_DIR, "concat.txt")
    with open(concat_list, "w") as fh:
        for p in padded_segs:
            fh.write(f"file '{p.replace(os.sep, '/')}'\n")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i",
                    concat_list, "-c:a", "libmp3lame", "-q:a", "2",
                    VOICEOVER_PATH], check=True, capture_output=True)
    voice_dur = ffprobe_duration(VOICEOVER_PATH)
    total = sum(durations)
    print(f"\nvoiceover.mp3: {voice_dur:.2f}s   (sum of scene frames: {total:.2f}s)")

    # 2) render frames per scene
    frame_count = 0
    for idx, (scene, dur) in enumerate(zip(scenes, durations)):
        img = make_frame(scene["lines"], badge=scene.get("badge", False))
        n = int(round(dur * FPS))
        for _ in range(n):
            img.save(os.path.join(FRAME_DIR, f"frame_{frame_count:06d}.png"))
            frame_count += 1
        print(f"  rendered scene {idx+1}: {n} frames")
    print(f"total frames: {frame_count}  (~{frame_count/FPS:.2f}s)")

    # 3) encode frames -> silent video
    novoice = os.path.join(HERE, "_novoice.mp4")
    subprocess.run(["ffmpeg", "-y", "-framerate", str(FPS), "-i",
                    os.path.join(FRAME_DIR, "frame_%06d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-preset", "medium", "-crf", "20", novoice],
                   check=True, capture_output=True)

    # 4) mux voiceover, +faststart
    subprocess.run(["ffmpeg", "-y", "-i", novoice, "-i", VOICEOVER_PATH,
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart", "-shortest",
                    "-map", "0:v:0", "-map", "1:a:0", VIDEO_PATH],
                   check=True, capture_output=True)

    # 5) cleanup intermediates
    os.remove(novoice)
    shutil.rmtree(FRAME_DIR, ignore_errors=True)
    for f in glob.glob(os.path.join(AUDIO_DIR, "*.wav")):
        os.remove(f)

    final_dur = ffprobe_duration(VIDEO_PATH)
    size = os.path.getsize(VIDEO_PATH) / 1024 / 1024
    print(f"\nDone -> {VIDEO_PATH}")
    print(f"  duration: {final_dur:.2f}s   size: {size:.1f} MB")


if __name__ == "__main__":
    main()
