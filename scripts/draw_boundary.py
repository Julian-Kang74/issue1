"""Draw the two scheduling cases used by the boundary study.

Requires Pillow. Run: python3 scripts/draw_boundary.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"

W, H = 1800, 1310
BG = "#f7f9fc"
INK = "#17243b"
MUTED = "#55647b"
LINE = "#cbd5e1"
BLUE = "#d9eafe"
BLUE_INK = "#18599b"
PURPLE = "#e8defe"
PURPLE_INK = "#633ca3"
GREEN = "#d7f4e8"
GREEN_INK = "#176b49"
RED = "#ffe0df"
RED_INK = "#ac3336"
AMBER = "#fff0cb"
AMBER_INK = "#895b05"


def font(size):
    return ImageFont.truetype(FONT, size)


def text(x, y, label, size=28, color=INK):
    d.text((x, y), label, font=font(size), fill=color)


def box(x0, y0, x1, y1, fill, radius=18, outline=None, width=2):
    d.rounded_rectangle(
        (x0, y0, x1, y1), radius=radius, fill=fill, outline=outline, width=width
    )


def segment(x0, x1, y, fill, label, ink, h=64):
    box(x0, y, x1, y + h, fill, radius=10)
    tw = d.textbbox((0, 0), label, font=font(28))[2]
    text((x0 + x1 - tw) / 2, y + 10, label, 28, ink)


def bar(x0, x1, y, color, label, label_x=None):
    box(x0, y, x1, y + 24, color, radius=7)
    text(x0 if label_x is None else label_x, y + 31, label, 25, color)


def tick(x, y, label):
    d.line((x, y - 7, x, y + 77), fill=LINE, width=3)
    tw = d.textbbox((0, 0), label, font=font(25))[2]
    text(x - tw / 2, y + 79, label, 25, MUTED)


LABELS = {
    "en": {
        "title": "Why is changing < to <= not enough?",
        "subtitle": "The same condition is used for two different position relationships",
        "a_title": "A  Multimodal request: text and the MM item share one token axis",
        "budget": "Budget = maximum number of decoder-side tokens scheduled this step",
        "text": "Text: 400",
        "mm": "One complete MM item: 800",
        "round1": "Step 1",
        "round1_result": "Computed 0; budget 500 → schedule 400, stopping before the MM item",
        "round2": "Step 2",
        "round2_result": "Original: computed 400; budget 500 → schedule 500, stop at 900 (split)",
        "fixed": "Fixed",
        "fixed_wait": "Budget 500 → schedule 0 and wait; budget 800 → schedule the full item",
        "fixed_full": "Budget 800 → process [400, 1200)",
        "b_title": "B  Encoder–decoder: encoder input and decoder tokens use separate axes",
        "encoder": "Encoder input",
        "encoder_span": "0 → 200 encoder tokens",
        "separate": "Separate input",
        "decoder": "Decoder step",
        "decoder_span": "0 → 100 tokens",
        "valid": "Can schedule 100",
        "pr": "PR's <= check: 0 <= 0 and 100 < 200 → falsely sees a split; schedules 0",
        "candidate": "Scoped candidate: skip this boundary check for encoder–decoder; retain cache and budget checks",
        "footnote": "Values come from scheduler correctness tests; this is not a model-quality or performance result.",
    },
}


for language, labels in LABELS.items():
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)

    text(70, 45, labels["title"], 52)
    text(72, 119, labels["subtitle"], 27, MUTED)

    box(56, 184, 1744, 731, "#ffffff", radius=24, outline="#dce3ed")
    text(88, 205, labels["a_title"], 33)
    text(89, 260, labels["budget"], 23, MUTED)

    # One position axis: text [0, 400), one MM item [400, 1200).
    x0, x400, x900, x1200 = 148, 555, 1064, 1370
    segment(x0, x400, 322, BLUE, labels["text"], BLUE_INK)
    segment(x400, x1200, 322, PURPLE, labels["mm"], PURPLE_INK)
    for x, value in [(x0, "0"), (x400, "400"), (x900, "900"), (x1200, "1200")]:
        tick(x, 322, value)

    text(89, 398, labels["round1"], 27)
    bar(x0, x400, 434, GREEN_INK, labels["round1_result"])

    text(89, 486, labels["round2"], 27)
    bar(x400, x900, 521, RED_INK, labels["round2_result"])

    text(89, 573, labels["fixed"], 27)
    box(x400 - 5, 609, x400 + 15, 634, AMBER, radius=6)
    text(x400 + 27, 604, labels["fixed_wait"], 25, AMBER_INK)
    bar(x400, x1200, 663, GREEN_INK, labels["fixed_full"])

    box(56, 758, 1744, 1218, "#ffffff", radius=24, outline="#dce3ed")
    text(88, 778, labels["b_title"], 33)

    text(89, 866, labels["encoder"], 27, PURPLE_INK)
    segment(373, 1190, 865, PURPLE, labels["encoder_span"], PURPLE_INK, 55)
    text(1235, 874, labels["separate"], 24, MUTED)

    text(89, 961, labels["decoder"], 27, BLUE_INK)
    segment(373, 782, 959, BLUE, labels["decoder_span"], BLUE_INK, 55)
    text(824, 968, labels["valid"], 24, GREEN_INK)

    d.line((88, 1041, 1712, 1041), fill=LINE, width=2)
    text(89, 1063, labels["pr"], 26, RED_INK)
    text(89, 1120, labels["candidate"], 26, GREEN_INK)

    text(65, 1244, labels["footnote"], 22, MUTED)

    out = ROOT / "docs" / "scheduler-boundary-en.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out, optimize=True)
    print(out)
