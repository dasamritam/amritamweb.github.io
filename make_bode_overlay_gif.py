#!/usr/bin/env python3
"""
Overlay Bode grid + animated SRG inset on gif-nonlinear.gif
→ saves gif-nonlinear-bode.gif, leaves the original untouched.

Bode grid calibration:
  x: log-freq  ω=0.1 → x=0,  ω=100 → x=420   (3 decades / 420 px)
  y: linear dB  +25 → y=0,   -35 → y=280        (60 dB / 280 px)

SRG inset (top-right, 112×112 px, equal-aspect complex plane):
  The SRG of the amplitude-dependent operator at gain level k(A) is the
  disk  { z ∈ ℂ : |z − k/2| ≤ k/2 }  centred on the real axis and
  tangent to the origin.  As amplitude A increases, k(A) decreases
  (nonlinear gain saturation) and the disk shrinks — giving the nested
  cardioid-like family seen in Figs. 5-7 of arXiv:2504.01585.
"""

import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

matplotlib.rcParams["mathtext.fontset"] = "cm"   # Computer Modern

# ── shared amplitude levels & colours ─────────────────────────────────────────
N_AMP  = 6
AMPS   = np.linspace(0.2, 3.5, N_AMP)
# Copper palette — warm gold → deep copper, matching the website accent
COLORS = ["#f2c898", "#e8a870", "#D08A4F", "#c07838", "#8B4A1C", "#6B3414"]

# ── system model (same as Bode animation) ─────────────────────────────────────
def wn(a):    return 9.0 / (1.0 + 0.45 * a)
def zeta(a):  return 0.10 + 0.09 * a

# frequency sweep for Nyquist curves (dense, covers 4 decades)
OMEGA_NY = np.logspace(-2, 3, 1200)


def nyquist(amp: float):
    """H(jω, A) for all OMEGA_NY → complex array."""
    w, z = wn(amp), zeta(amp)
    return w**2 / (-OMEGA_NY**2 + 2j * z * w * OMEGA_NY + w**2)

W, H = 420, 280


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — SRG inset  (one per animation stage)
# ══════════════════════════════════════════════════════════════════════════════
INSET_X, INSET_Y = 255, 5      # position in the 420×280 canvas
INSET_W, INSET_H = 160, 160

# The SRG of the dynamic operator H(jω, A) is the 2D region in ℂ swept
# by all Nyquist curves as amplitude varies.  Its boundary consists of the
# outermost Nyquist curve (low A, high gain) and the innermost one (high A).
# The shape is a crescent/lune — more complex than a circle.
#
# Axes: Re = (-0.4 … 1.2), Im = (-4.8 … 4.8)
# We intentionally do NOT enforce equal aspect so the crescent fits the
# square inset without wasting space (the Im axis is compressed visually,
# which is common in SRG plots in the literature).
SRG_RE_LIM = (-0.38, 1.20)
SRG_IM_LIM = (-2.20, 2.20)   # zoomed in: shows crescent for A=2…3.5 fully,
                               # clips deep peaks for low A (indicated by clipping)

THETA = np.linspace(0, 2 * np.pi, 360)


def render_srg_inset(n_visible: int) -> Image.Image:
    """
    SRG(H) as the filled 2-D region swept by Nyquist curves for
    amplitude levels AMPS[0 … n_visible-1].

    Outer boundary  = Nyquist(A_min)   [high gain, deep arc]
    Inner boundary  = Nyquist(A_cur)   [lower gain, shallower arc]
    Filled region   = crescent/lune in lower + upper half-plane
    Individual curves drawn with colour gradient blue → copper.
    """
    fig_w = INSET_W / 100.0
    fig_h = INSET_H / 100.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)

    BG = "#0a0b10"
    fig.patch.set_facecolor(BG);  ax.set_facecolor(BG)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.88, bottom=0.22)

    ax.set_xlim(*SRG_RE_LIM);  ax.set_ylim(*SRG_IM_LIM)
    # no equal-aspect: Im axis compressed to fit; still clearly shows lune shape

    # ── grid & reference axes ─────────────────────────────────────────────────
    ax.axhline(0, color="#3a2a1a", lw=0.55, zorder=1)
    ax.axvline(0, color="#3a2a1a", lw=0.55, zorder=1)
    ax.grid(True, color="#1e1510", lw=0.35, ls="-", zorder=0)

    # unit circle (reference)
    ax.plot(np.cos(THETA), np.sin(THETA),
            color="#6a4a30", lw=0.55, ls="--", alpha=0.70, zorder=2)
    ax.text(0.70, 1.20, r"$|z|{=}1$", color="#6a4a30", fontsize=3.8, alpha=0.80)

    # ── filled SRG crescent (between outer and inner Nyquist curves) ──────────
    if n_visible >= 2:
        H_out = nyquist(AMPS[0])            # outermost  (low A, high gain)
        H_in  = nyquist(AMPS[n_visible-1])  # innermost  (current high A)
        # lower half-plane polygon
        poly_re = np.concatenate([H_out.real, H_in.real[::-1]])
        poly_im = np.concatenate([H_out.imag, H_in.imag[::-1]])
        ax.fill(poly_re, poly_im,  color="#D08A4F", alpha=0.18, zorder=2)
        # upper half-plane mirror (negative frequencies)
        ax.fill(poly_re, -poly_im, color="#D08A4F", alpha=0.18, zorder=2)

    # ── individual Nyquist curves (colour gradient, both half-planes) ─────────
    for i in range(n_visible):
        H = nyquist(AMPS[i])
        is_cur = (i == n_visible - 1)
        lw    = 1.50 if is_cur else 0.75
        alpha = 1.00 if is_cur else 0.45

        ax.plot(H.real,  H.imag, color=COLORS[i], lw=lw, alpha=alpha,    zorder=3+i)
        ax.plot(H.real, -H.imag, color=COLORS[i], lw=lw*0.6,
                ls=":", alpha=alpha*0.50, zorder=3+i)  # upper half (mirror, fainter)

    # ── decoration ────────────────────────────────────────────────────────────
    TC = "#a07858"
    ax.set_xlabel(r"$\mathrm{Re}(z)$", color=TC, fontsize=5.5, labelpad=1)
    ax.set_ylabel(r"$\mathrm{Im}(z)$", color=TC, fontsize=5.5, labelpad=1)
    ax.set_title(r"$\mathrm{SRG}(H)$",  color="#c08850", fontsize=6.2, pad=2)
    ax.tick_params(colors="#7a5838", labelsize=4.2, length=2, width=0.4)
    for s in ax.spines.values():
        s.set_edgecolor("#3a2a1a");  s.set_linewidth(0.5);  s.set_visible(True)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=BG)
    plt.close(fig);  buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    if img.size != (INSET_W, INSET_H):
        img = img.resize((INSET_W, INSET_H), Image.LANCZOS)
    return img


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — Composite and save
# ══════════════════════════════════════════════════════════════════════════════

def apply_overlays(src_path: str, dst_path: str) -> None:
    src = Image.open(src_path)
    n   = src.n_frames
    print(f"Source: {src_path}  ({n} frames, {src.size})")

    print(f"Rendering {N_AMP} SRG insets …")
    srg_insets = []
    for k in range(1, N_AMP + 1):
        srg_insets.append(render_srg_inset(k))
        print(f"  {k}/{N_AMP}")

    # map frame index → SRG stage (0-indexed)
    def stage(i): return min(int(i * N_AMP / n), N_AMP - 1)

    print(f"Compositing {n} frames …")
    frames = [];  durations = []

    for i in range(n):
        src.seek(i)
        durations.append(src.info.get("duration", 67))

        base = src.convert("RGBA")
        comp = base.copy()
        comp.paste(srg_insets[stage(i)], (INSET_X, INSET_Y))
        frames.append(comp.convert("RGB"))

        if (i + 1) % 14 == 0 or i == n - 1:
            print(f"  {i+1}/{n}")

    q = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    q[0].save(dst_path, save_all=True, append_images=q[1:],
              loop=0, duration=durations, optimize=True)
    kb = Path(dst_path).stat().st_size // 1024
    print(f"Saved {dst_path}  ({n} frames, {kb} KB)")


if __name__ == "__main__":
    apply_overlays("gif-nonlinear.gif", "gif-nonlinear-bode.gif")
