#!/usr/bin/env python3
"""
make_paper_gif.py — Convert an academic PDF to a scrolling preview GIF.

Single file:
    python make_paper_gif.py industry_papers/tue.pdf paper_gifs/tue.gif

Batch (all PDFs in a folder):
    python make_paper_gif.py --batch industry_papers/ paper_gifs/

Dependencies — install ONE of:
    pip install pymupdf           # preferred, no system deps
    pip install pdf2image pillow  # requires poppler on PATH

Options:
    --width    px   GIF width  (default: 260 — matches popup card)
    --height   px   Viewport window height (default: 110 — matches ppc-body)
    --fps      int  Frames per second      (default: 15)
    --duration sec  Total scroll time      (default: 8)
    --dpi      int  PDF render resolution  (default: 100)
    --pages    int  Max pages to include   (default: 5)
    --gap      px   White gap between pages (default: 6)
    --colors   int  GIF palette size       (default: 64)
    --batch         Process all PDFs in src folder → dst folder
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
from PIL import Image


# ── PDF → list[PIL.Image] ─────────────────────────────────────────────────────

def _render_with_pymupdf(pdf_path: str, dpi: int, max_pages: Optional[int]) -> List[Image.Image]:
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    n = min(len(doc), max_pages) if max_pages else len(doc)
    pages = []
    for i in range(n):
        pix = doc[i].get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        pages.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    doc.close()
    return pages


def _render_with_pdf2image(pdf_path: str, dpi: int, max_pages: Optional[int]) -> List[Image.Image]:
    from pdf2image import convert_from_path
    pages = convert_from_path(pdf_path, dpi=dpi)
    return pages[:max_pages] if max_pages else pages


def render_pages(pdf_path: str, dpi: int = 100, max_pages: Optional[int] = 5) -> List[Image.Image]:
    """Render PDF pages to RGB PIL Images. Tries PyMuPDF then pdf2image."""
    for fn in (_render_with_pymupdf, _render_with_pdf2image):
        try:
            return fn(pdf_path, dpi, max_pages)
        except ImportError:
            continue
    raise ImportError(
        "No PDF renderer found. Install PyMuPDF:\n"
        "    pip install pymupdf\n"
        "or pdf2image + poppler:\n"
        "    pip install pdf2image"
    )


# ── stitch pages vertically ───────────────────────────────────────────────────

def stitch(pages: List[Image.Image], target_w: int, gap: int = 6) -> Image.Image:
    """Scale all pages to target_w and stack them vertically with a gap."""
    scaled = []
    for p in pages:
        ratio = target_w / p.width
        scaled.append(p.resize((target_w, round(p.height * ratio)), Image.LANCZOS))

    total_h = sum(p.height for p in scaled) + gap * (len(scaled) - 1)
    canvas = Image.new("RGB", (target_w, total_h), (255, 255, 255))
    y = 0
    for i, p in enumerate(scaled):
        canvas.paste(p, (0, y))
        y += p.height
        if i < len(scaled) - 1:
            y += gap  # white gap — canvas already white
    return canvas


# ── build scroll frames ────────────────────────────────────────────────────────

def make_frames(
    full: Image.Image,
    viewport_h: int,
    fps: int,
    duration: float,
) -> List[Image.Image]:
    """Slide a viewport_h window from top to bottom and return cropped frames."""
    w = full.width
    scroll_dist = max(full.height - viewport_h, 0)
    n = round(fps * duration)

    frames = []
    for i in range(n):
        t = i / max(n - 1, 1)
        y = round(t * scroll_dist)
        frames.append(full.crop((0, y, w, y + viewport_h)))
    return frames


# ── save GIF ──────────────────────────────────────────────────────────────────

def save_gif(
    frames: List[Image.Image],
    out_path: str,
    fps: int,
    colors: int = 64,
) -> None:
    delay = round(1000 / fps)
    quantised = [f.convert("P", palette=Image.ADAPTIVE, colors=colors) for f in frames]
    quantised[0].save(
        out_path,
        save_all=True,
        append_images=quantised[1:],
        loop=0,
        duration=delay,
        optimize=True,
    )


# ── public entry point ────────────────────────────────────────────────────────

def pdf_to_gif(
    pdf_path: str,
    gif_path: str,
    width: int               = 260,
    height: int              = 110,
    fps: int                 = 12,
    duration: Optional[float] = None,   # None → auto: 1.5 s/page, min 8 s
    dpi: int                 = 80,
    max_pages: Optional[int] = None,    # None → all pages
    gap: int                 = 6,
    colors: int              = 48,
) -> None:
    """Convert *pdf_path* to a scrolling preview GIF saved at *gif_path*.

    With duration=None the scroll speed is ~1.5 s per page so short and
    long papers both feel natural. Pass an explicit value to override.
    """
    print(f"  ↳ rendering {pdf_path} …")
    pages = render_pages(pdf_path, dpi=dpi, max_pages=max_pages)
    full  = stitch(pages, target_w=width, gap=gap)

    # auto duration: 1.5 s per page, minimum 8 s
    dur = duration if duration is not None else max(8.0, len(pages) * 1.5)

    frames = make_frames(full, viewport_h=height, fps=fps, duration=dur)
    Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
    save_gif(frames, gif_path, fps=fps, colors=colors)
    size_kb = Path(gif_path).stat().st_size // 1024
    print(f"    saved {gif_path}  ({len(pages)} pages, {dur:.0f}s, {len(frames)} frames, {size_kb} KB)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert PDF(s) to scrolling preview GIF(s).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("src",  help="Input PDF file, or source folder when --batch")
    p.add_argument("dst",  help="Output GIF file, or destination folder when --batch")
    p.add_argument("--batch",    action="store_true", help="Process all PDFs in src/ → dst/")
    p.add_argument("--width",    type=int,   default=260,  metavar="px")
    p.add_argument("--height",   type=int,   default=110,  metavar="px")
    p.add_argument("--fps",      type=int,   default=12)
    p.add_argument("--duration", type=float, default=None, metavar="sec",
                   help="scroll duration in seconds (default: auto 1.5s/page)")
    p.add_argument("--dpi",      type=int,   default=80)
    p.add_argument("--pages",    type=int,   default=None, metavar="N", dest="max_pages",
                   help="max pages to include (default: all)")
    p.add_argument("--gap",      type=int,   default=6,    metavar="px")
    p.add_argument("--colors",   type=int,   default=48)
    return p


def main() -> None:
    args = build_parser().parse_args()

    kwargs = dict(
        width=args.width, height=args.height,
        fps=args.fps,     duration=args.duration,
        dpi=args.dpi,     max_pages=args.max_pages,
        gap=args.gap,     colors=args.colors,
    )

    if args.batch:
        src_dir = Path(args.src)
        dst_dir = Path(args.dst)
        pdfs = sorted(src_dir.glob("*.pdf"))
        if not pdfs:
            sys.exit(f"No PDF files found in {src_dir}")
        print(f"Batch: {len(pdfs)} PDF(s) → {dst_dir}/")
        for pdf in pdfs:
            pdf_to_gif(str(pdf), str(dst_dir / pdf.with_suffix(".gif").name), **kwargs)
    else:
        pdf_to_gif(args.src, args.dst, **kwargs)

    print("Done.")


if __name__ == "__main__":
    main()
