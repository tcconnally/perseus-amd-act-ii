"""
Generate the AMD Act II demo video for Perseus Vault.

Fully automated, no human voice: narration is synthesized offline with
piper-tts (en_US-lessac-medium), then terminal-style PIL frames are rendered
and muxed with the voiceover via ffmpeg.

Adapted from the proven pipeline in
github.com/tcconnally/perseus-vault-hackathon/generate_video.py.

HONESTY: every benchmark/cost frame renders "data_source: published-spec
estimate"; the closing frame states the estimate caveat. The tables here match
BENCHMARKS.md exactly.
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
    # 1 — title (0-8s target)
    dict(lines=[
        ("", None), ("", None),
        ("# Perseus Vault", TITLE_COLOR),
        ("", None),
        ("  Agentic Memory on AMD Instinct MI300X", HIGHLIGHT_COLOR),
    ]),
    # 2 — what it is (8-20s)
    dict(lines=[
        ("# What it is", TITLE_COLOR),
        ("", None),
        ("  • Single self-contained Rust binary", OUTPUT_COLOR),
        ("  • Hybrid recall  —  SQLite FTS5 full-text + ranking", OUTPUT_COLOR),
        ("  • 27 tools over the Model Context Protocol (MCP)", OUTPUT_COLOR),
        ("  • Runs on a single AMD Instinct MI300X", OUTPUT_COLOR),
        ("", None),
        ("$ perseus-vault serve --mcp   # one binary, no sidecars", PROMPT_COLOR),
    ]),
    # 3 — store / recall / decay under load (20-34s)
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
    # 4 — recall throughput + footprint — matches docs/BENCHMARKS.md §1/§2
    #     (reference implementation, MEASURED on an AMD CPU)
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
    # 5 — cost economics — matches docs/BENCHMARKS.md §3
    #     (PROJECTION from published-spec datasheet + cloud-price inputs)
    dict(badge="data_source: projection (published-spec inputs)", lines=[
        ("# Cost economics  —  serve Llama-3.1-70B FP16", TITLE_COLOR),
        ("", None),
        ("  Accelerator   HBM      Agents   GPU $/agent-hr", HIGHLIGHT_COLOR),
        ("  -----------   ------   ------   --------------", DIM_COLOR),
        ("  MI300X        192 GB     20.4          $0.133", (120, 255, 160)),
        ("  A100 80GB      80 GB      7.6          $0.474", OUTPUT_COLOR),
        ("  H100 SXM       80 GB      7.6          $1.034", OUTPUT_COLOR),
        ("", None),
        ("  1 card fits 70B → most HBM left for agents.", DIM_COLOR),
        ("  ~8x cheaper/agent than H100.  python src/economics.py", DIM_COLOR),
    ]),
    # 6 — closing (74-88s)
    dict(lines=[
        ("", None),
        ("# Perseus Vault × AMD Instinct MI300X", TITLE_COLOR),
        ("", None),
        ("  github.com/tcconnally/perseus-amd-act-ii", HIGHLIGHT_COLOR),
        ("", None),
        ("  Recall = measured (AMD CPU).  Cost = projection", (255, 140, 90)),
        ("  (published-spec).  No measured MI300X numbers yet —", (255, 140, 90)),
        ("  real MI300X data pending cloud credits.", (255, 140, 90)),
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
