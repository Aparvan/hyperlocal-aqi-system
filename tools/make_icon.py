"""
Generate the AeroSense AI Windows application icon (`assets/app_icon.ico` + `assets/app_icon.png`).
Creates a crisp multi-resolution `.ico` (16, 32, 48, 64, 128, 256 px) depicting a glowing
cyan/emerald hyperlocal geospatial radar grid with an atmospheric plume wave.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import math

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def make_master_icon(size: int = 512) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Rounded dark navy squircle background
    pad = int(size * 0.05)
    radius = int(size * 0.22)
    draw.rounded_rectangle(
        [pad, pad, size - pad, size - pad],
        radius=radius,
        fill=(11, 18, 33, 255),
        outline=(56, 189, 248, 220),
        width=int(size * 0.025),
    )

    # Subtle internal grid lines (hyperlocal 24x24 mesh motif)
    inner = int(size * 0.16)
    span = size - 2 * inner
    for i in range(5):
        pos = inner + int(span * i / 4)
        draw.line([(pos, inner), (pos, size - inner)], fill=(30, 58, 95, 130), width=3)
        draw.line([(inner, pos), (size - inner, pos)], fill=(30, 58, 95, 130), width=3)

    # Glowing pollution/clean-air radar rings
    cx, cy = size // 2, size // 2
    for r_frac, color, width in [
        (0.32, (16, 185, 129, 190), 8),   # Emerald outer ring
        (0.22, (56, 189, 248, 220), 9),   # Cyan middle ring
        (0.12, (245, 158, 11, 230), 10),  # Amber inner hotspot ring
    ]:
        r = int(size * r_frac)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=width)

    # Atmospheric wind-advection wave across center
    pts = []
    for x in range(inner, size - inner + 1, 6):
        t = (x - inner) / span
        y = cy + int(math.sin(t * 2 * math.pi) * size * 0.08)
        pts.append((x, y))
    if len(pts) > 1:
        draw.line(pts, fill=(125, 211, 252, 255), width=12)

    # Center sensor node pin
    pin_r = int(size * 0.048)
    draw.ellipse([cx - pin_r, cy - pin_r, cx + pin_r, cy + pin_r], fill=(255, 255, 255, 255))
    inner_r = int(size * 0.024)
    draw.ellipse([cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r], fill=(14, 165, 233, 255))

    return img


def main():
    master = make_master_icon(512)
    png_path = ASSETS_DIR / "app_icon.png"
    ico_path = ASSETS_DIR / "app_icon.ico"

    master.save(png_path, format="PNG")
    master.save(
        ico_path,
        format="ICO",
        sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    print(f"Generated {png_path} and {ico_path}")


if __name__ == "__main__":
    main()
