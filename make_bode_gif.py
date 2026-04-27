#!/usr/bin/env python3
"""
Generate gif-nonlinear-bode.gif — animated amplitude-dependent Bode magnitude plot
for the Nonlinear Performance Shaping hero node.

Usage:
    python make_bode_gif.py

Output: gif-nonlinear-bode.gif  (420×260 px, dark background)
"""

import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from pathlib import Path

# ── output config ──────────────────────────────────────────────────────────────
OUT  = "gif-nonlinear-bode.gif"
W, H = 420, 260        # GIF pixel size
DPI  = 100
FPS  = 15
N_FRAMES = 80

# ── visual theme ───────────────────────────────────────────────────────────────
BG     = "#0d1117"
GRID_M = "#1c2535"    # major grid
GRID_m = "#161e2b"    # minor grid
SPINE  = "#2a3a4a"
TEXT   = "#7a93aa"
TICK   = "#5a7a8a"
DB3    = "#c0824a"    # -3 dB marker color (copper)

# amplitude levels low → high: blue → copper
N_AMP  = 6
AMPS   = np.linspace(0.2, 3.5, N_AMP)
COLORS = ["#4a8fd4", "#4fa8c0", "#52b89a", "#8aba5a", "#d4a030", "#D08A4F"]

# ── frequency-domain model ─────────────────────────────────────────────────────
# 2nd-order system with amplitude-dependent natural frequency and damping:
#   H(jω, A) = ωn²  /  (−ω² + 2jζωnω + ωn²)
# As A increases: ωn softens (frequency shift) and ζ grows (more damping) —
# this mimics how saturation/stiffening nonlinearities alter the Bode diagram.
OMEGA = np.logspace(-1, 2.2, 400)   # 0.1 … 158 rad/s

def mag_db(omega: np.ndarray, amp: float) -> np.ndarray:
    wn   = 9.0 / (1.0 + 0.45 * amp)   # natural freq softens with amplitude
    zeta = 0.10 + 0.09 * amp           # damping grows with amplitude
    H    = wn**2 / (-omega**2 + 2j * zeta * wn * omega + wn**2)
    return 20.0 * np.log10(np.abs(H))

CURVES = [mag_db(OMEGA, a) for a in AMPS]

# ── frame renderer ─────────────────────────────────────────────────────────────
FIG_W = W / DPI
FIG_H = H / DPI

def render_frame(t: float) -> Image.Image:
    """t ∈ [0, 1]: animation progress."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.set_xscale("log")
    ax.set_xlim(0.1, 158)
    ax.set_ylim(-46, 14)

    # grid
    ax.grid(True, which="major", color=GRID_M, linewidth=0.75, linestyle="-", zorder=0)
    ax.grid(True, which="minor", color=GRID_m, linewidth=0.30, linestyle="-", zorder=0)

    # -3 dB reference line
    ax.axhline(-3, color=DB3, linewidth=0.7, linestyle="--", alpha=0.55, zorder=1)
    ax.text(0.115, -1.5, "−3 dB", color=DB3, fontsize=5.2, alpha=0.75, va="bottom")

    # 0 dB reference line
    ax.axhline(0, color=SPINE, linewidth=0.5, linestyle="-", alpha=0.6, zorder=1)

    # draw curves with staggered reveal
    # curve i starts drawing at t_start_i and finishes at t_end_i
    slot = 0.75 / N_AMP        # fraction of t used per curve onset
    draw_span = slot * 1.6     # each curve takes this long to fully appear

    for i, (amp, curve, color) in enumerate(zip(AMPS, CURVES, COLORS)):
        t_start = i * slot
        if t < t_start:
            break
        progress  = min(1.0, (t - t_start) / draw_span)
        n_pts     = max(2, int(progress * len(OMEGA)))
        alpha     = 0.45 + 0.55 * progress
        lw        = 1.0 + 0.35 * (i / (N_AMP - 1))
        ax.plot(OMEGA[:n_pts], curve[:n_pts],
                color=color, linewidth=lw, alpha=alpha, zorder=2 + i)

    # axes labels & ticks
    ax.set_xlabel("ω  [rad/s]", color=TEXT, fontsize=6.5, labelpad=2)
    ax.set_ylabel("|H|  [dB]", color=TEXT, fontsize=6.5, labelpad=2)
    ax.tick_params(axis="both", colors=TICK, labelsize=5.5, length=3, width=0.6)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE)
        spine.set_linewidth(0.6)

    # legend (appears once first two curves are visible)
    if t > 1.8 * slot:
        handles = [
            Line2D([0], [0], color=COLORS[0],  lw=1.5, label=f"A = {AMPS[0]:.1f}"),
            Line2D([0], [0], color=COLORS[-1], lw=1.5, label=f"A = {AMPS[-1]:.1f}"),
        ]
        ax.legend(handles=handles, loc="lower left", fontsize=5.5,
                  facecolor="#111922", edgecolor=SPINE, labelcolor=TEXT,
                  framealpha=0.85, handlelength=1.6, borderpad=0.6)

    # title
    ax.set_title("Amplitude-Dependent Bode Magnitude", color=TEXT,
                 fontsize=6.5, pad=4, loc="left")

    fig.tight_layout(pad=0.55)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()

# ── build frames ───────────────────────────────────────────────────────────────
n_draw  = int(N_FRAMES * 0.72)   # drawing phase
n_hold  = N_FRAMES - n_draw      # hold-final phase

print(f"Rendering {N_FRAMES} frames at {W}×{H} px …")
frames = []
for i in range(n_draw):
    t = i / max(n_draw - 1, 1)
    frames.append(render_frame(t))
    if (i + 1) % 16 == 0:
        print(f"  {i+1}/{N_FRAMES}")

hold = render_frame(1.0)
for _ in range(n_hold):
    frames.append(hold.copy())
print(f"  {N_FRAMES}/{N_FRAMES}")

# ── save GIF ───────────────────────────────────────────────────────────────────
delay = round(1000 / FPS)
quantised = [f.convert("P", palette=Image.ADAPTIVE, colors=96) for f in frames]
quantised[0].save(
    OUT, save_all=True, append_images=quantised[1:],
    loop=0, duration=delay, optimize=True,
)
size_kb = Path(OUT).stat().st_size // 1024
print(f"Saved {OUT}  ({len(frames)} frames, {size_kb} KB)")
