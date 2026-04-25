"""
Generate four thematic GIFs for the research website.
Each GIF is 420×280, dark background, copper accent colour to match the site palette.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch, Circle
import matplotlib.patheffects as pe
from PIL import Image
import io, os

BG   = "#0A0B10"
CU   = "#D08A4F"        # copper
CU2  = "#B8743D"
DIM  = "#4A4C56"
HI   = "#ECE7DC"
LO   = "rgba(236,231,220,0.4)"
W, H = 420, 280
DPI  = 90
FRAMES = 48
DELAY  = 60            # ms between frames

OUT = os.path.dirname(os.path.abspath(__file__))


def save_gif(fig, update_fn, path, n=FRAMES):
    """Render n frames and save as an optimised GIF."""
    frames = []
    for i in range(n):
        update_fn(i)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=DPI,
                    facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0)
        frames.append(Image.open(buf).convert('P', palette=Image.ADAPTIVE, colors=128))
    frames[0].save(
        path, save_all=True, append_images=frames[1:],
        loop=0, duration=DELAY, optimize=True
    )
    plt.close(fig)
    print(f"  saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# GIF 1 — Control of PDEs  (1-D wave equation driven to rest by boundary feedback)
# ─────────────────────────────────────────────────────────────────────────────
def make_pde_gif():
    Nx = 120
    x  = np.linspace(0, 1, Nx)
    c  = 1.0
    dx = x[1] - x[0]
    dt = 0.85 * dx / c          # Courant number 0.85 — stable

    # initial Gaussian pluck
    u  = 0.9 * np.exp(-((x - 0.35) ** 2) / 0.006)
    u_prev = u.copy()

    # boundary damping coefficient
    alpha = 0.18

    fig, ax = plt.subplots(figsize=(W/DPI, H/DPI))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.05, 1.05)
    ax.axis('off')

    # decorative grid lines
    for yv in [-1, -0.5, 0, 0.5, 1]:
        ax.axhline(yv, color=DIM, lw=0.4, alpha=0.3)
    ax.axhline(0, color=DIM, lw=0.6, alpha=0.5)

    # x-axis line
    ax.plot([0,1],[0,0], color=DIM, lw=0.8, alpha=0.4)

    # label
    ax.text(0.5, 0.93, "PDE  ·  Boundary Feedback Control",
            transform=ax.transAxes, ha='center', va='top',
            color=CU, fontsize=7, fontfamily='monospace', alpha=0.85)

    line, = ax.plot(x, u, color=CU, lw=1.8, solid_capstyle='round')
    fill  = ax.fill_between(x, 0, u, alpha=0.12, color=CU)

    # energy readout
    etxt = ax.text(0.97, 0.08, "", transform=ax.transAxes,
                   ha='right', va='bottom', color=HI,
                   fontsize=6.5, fontfamily='monospace')

    state = {'u': u.copy(), 'up': u_prev.copy()}

    def update(i):
        u_  = state['u']
        up_ = state['up']
        # leapfrog step (×3 sub-steps per frame for speed)
        for _ in range(8):
            u_new = np.zeros_like(u_)
            u_new[1:-1] = (2*u_[1:-1] - up_[1:-1]
                           + r * (u_[2:] - 2*u_[1:-1] + u_[:-2]))
            # absorbing boundary at both ends
            u_new[0]  = u_new[1]  * (1 - alpha)
            u_new[-1] = u_new[-2] * (1 - alpha)
            up_[:] = u_[:]
            u_[:] = u_new

        line.set_ydata(u_)

        # redraw fill (remove old, add new)
        for coll in ax.collections[:]:
            coll.remove()
        ax.fill_between(x, 0, u_, alpha=0.12, color=CU)

        energy = np.sqrt(np.mean(u_ ** 2))
        etxt.set_text(f"E = {energy:.3f}")

    save_gif(fig, update, os.path.join(OUT, "gif-pde-control.gif"))


# ─────────────────────────────────────────────────────────────────────────────
# GIF 2 — Physics-Informed Learning  (surrogate fitting a nonlinear trajectory)
# ─────────────────────────────────────────────────────────────────────────────
def make_learning_gif():
    np.random.seed(42)
    T = np.linspace(0, 2 * np.pi, 120)
    # "true" state trajectory (van-der-Pol-like limit cycle in state space)
    x_true = np.sin(T)
    y_true = np.cos(T) * (1 + 0.3 * np.sin(2 * T))

    # noisy observations (sparse)
    idx = np.arange(0, 120, 5)
    xo = x_true[idx] + 0.07 * np.random.randn(len(idx))
    yo = y_true[idx] + 0.07 * np.random.randn(len(idx))

    fig, ax = plt.subplots(figsize=(W/DPI, H/DPI))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.axis('off')

    for v in [-1, 0, 1]:
        ax.axhline(v, color=DIM, lw=0.4, alpha=0.25)
        ax.axvline(v, color=DIM, lw=0.4, alpha=0.25)

    ax.text(0.5, 0.96, "Physics-Informed  ·  Surrogate Modeling",
            transform=ax.transAxes, ha='center', va='top',
            color=CU, fontsize=7, fontfamily='monospace', alpha=0.85)

    # axis labels
    ax.text(1.55, -0.07, "x₁", color=DIM, fontsize=7, ha='right')
    ax.text(-0.07, 1.55, "x₂", color=DIM, fontsize=7, va='top')

    # ghost true orbit (faint)
    ax.plot(x_true, y_true, color=DIM, lw=0.8, alpha=0.25, linestyle='--')

    # observations
    ax.scatter(xo, yo, s=12, color=HI, alpha=0.55, zorder=5, linewidths=0)

    # surrogate line (grows over frames)
    surr_line, = ax.plot([], [], color=CU, lw=2.0, alpha=0.92, zorder=6)

    # uncertainty band
    band = ax.fill_between([], [], [], alpha=0.0, color=CU)

    # "converging" label
    ctxt = ax.text(0.03, 0.06, "", transform=ax.transAxes,
                   ha='left', va='bottom', color=CU,
                   fontsize=6.5, fontfamily='monospace')

    def update(i):
        nonlocal band
        # surrogate "learns" more of the orbit each frame
        frac  = min(1.0, (i + 1) / (FRAMES * 0.75))
        n_pts = max(2, int(len(T) * frac))
        xs = x_true[:n_pts]
        ys = y_true[:n_pts]
        noise = (1 - frac) * 0.12
        surr_line.set_data(xs + noise * np.random.randn(n_pts),
                           ys + noise * np.random.randn(n_pts))
        for c in ax.collections:
            c.remove()
        ax.scatter(xo, yo, s=12, color=HI, alpha=0.55, zorder=5, linewidths=0)
        w = (1 - frac) * 0.18
        ax.fill_between(xs, ys - w, ys + w, alpha=0.10, color=CU, zorder=3)
        pct = int(frac * 100)
        ctxt.set_text(f"fit  {pct:3d}%")

    save_gif(fig, update, os.path.join(OUT, "gif-physics-learning.gif"))


# ─────────────────────────────────────────────────────────────────────────────
# GIF 3 — Nonlinear Performance Shaping  (amplitude-dependent Bode magnitude)
# ─────────────────────────────────────────────────────────────────────────────
def make_nonlinear_gif():
    freqs  = np.logspace(-1, 1.5, 300)
    amps   = np.linspace(0.05, 2.5, FRAMES)

    def bode_mag(f, A):
        # simple amplitude-dependent 2nd-order resonance
        omega_n = 1.0 + 0.35 * A          # resonance shifts with amplitude
        zeta    = 0.18 + 0.12 * A
        den     = np.sqrt((1 - (f/omega_n)**2)**2 + (2*zeta*f/omega_n)**2)
        return 1.0 / (den + 1e-9)

    fig, ax = plt.subplots(figsize=(W/DPI, H/DPI))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xscale('log')
    ax.set_xlim(freqs[0], freqs[-1])
    ax.set_ylim(-0.05, 4.2)
    ax.axis('off')

    # grid
    for fv in [0.1, 0.3, 1, 3, 10, 30]:
        ax.axvline(fv, color=DIM, lw=0.4, alpha=0.3)
    for yv in [0, 1, 2, 3, 4]:
        ax.axhline(yv, color=DIM, lw=0.4, alpha=0.3)

    ax.text(0.5, 0.96, "Nonlinear  ·  Amplitude-Dependent Bode",
            transform=ax.transAxes, ha='center', va='top',
            color=CU, fontsize=7, fontfamily='monospace', alpha=0.85)

    # frequency axis label
    ax.text(0.5, 0.02, "frequency  (rad/s)", transform=ax.transAxes,
            ha='center', va='bottom', color=DIM, fontsize=6.5)

    # ghost linear (A→0) response in dim copper
    mag0 = bode_mag(freqs, 0.01)
    ax.plot(freqs, mag0, color=DIM, lw=0.9, alpha=0.4, linestyle='--')

    main_line, = ax.plot([], [], color=CU, lw=2.2)
    fill_band  = ax.fill_between([], [], [], alpha=0.0)

    atxt = ax.text(0.97, 0.88, "", transform=ax.transAxes,
                   ha='right', va='top', color=HI,
                   fontsize=6.5, fontfamily='monospace')

    def update(i):
        A   = amps[i]
        mag = bode_mag(freqs, A)
        main_line.set_data(freqs, mag)
        for c in ax.collections:
            c.remove()
        ax.fill_between(freqs, 0, mag, alpha=0.10, color=CU)
        atxt.set_text(f"A = {A:.2f}")

    save_gif(fig, update, os.path.join(OUT, "gif-nonlinear.gif"))


# ─────────────────────────────────────────────────────────────────────────────
# GIF 4 — System-Level Proactive Diagnostics  (fault propagation & detection)
# ─────────────────────────────────────────────────────────────────────────────
def make_diagnostics_gif():
    # Fixed node positions for a small industrial network
    nodes = np.array([
        [0.50, 0.82],   # 0 — controller hub
        [0.20, 0.55],   # 1
        [0.50, 0.55],   # 2
        [0.80, 0.55],   # 3
        [0.12, 0.26],   # 4 — ← fault origin
        [0.38, 0.26],   # 5
        [0.62, 0.26],   # 6
        [0.88, 0.26],   # 7
    ])
    edges = [(0,1),(0,2),(0,3),(1,4),(1,5),(2,5),(2,6),(3,6),(3,7)]

    # BFS distance from fault node 4
    from collections import deque
    dist = {4: 0}
    q = deque([4])
    adj = {i: [] for i in range(8)}
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
    while q:
        v = q.popleft()
        for u in adj[v]:
            if u not in dist:
                dist[u] = dist[v] + 1
                q.append(u)
    max_dist = max(dist.values())

    fig, ax = plt.subplots(figsize=(W/DPI, H/DPI))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(0.1, 1.0)
    ax.axis('off')

    ax.text(0.5, 0.97, "System-Level  ·  Proactive Fault Diagnostics",
            transform=ax.transAxes, ha='center', va='top',
            color=CU, fontsize=7, fontfamily='monospace', alpha=0.85)

    labels = ["Hub", "N1", "N2", "N3", "F", "N5", "N6", "N7"]
    node_labels_art = []
    edge_arts = []

    # draw edges (static)
    for a, b in edges:
        l, = ax.plot([nodes[a,0], nodes[b,0]],
                     [nodes[a,1], nodes[b,1]],
                     color=DIM, lw=1.0, alpha=0.4, zorder=1)
        edge_arts.append((a, b, l))

    # draw nodes
    node_circles = []
    for i, (nx_, ny_) in enumerate(nodes):
        circ = Circle((nx_, ny_), 0.055, color=DIM, zorder=3,
                       transform=ax.transData)
        ax.add_patch(circ)
        node_circles.append(circ)
        txt = ax.text(nx_, ny_, labels[i], ha='center', va='center',
                      fontsize=6.5, fontfamily='monospace', color=BG,
                      fontweight='bold', zorder=4)
        node_labels_art.append(txt)

    status_txt = ax.text(0.5, 0.05, "", transform=ax.transAxes,
                         ha='center', va='bottom', color=CU,
                         fontsize=6.5, fontfamily='monospace')

    # animation phases:
    # 0..11   : idle (all nodes normal)
    # 12..19  : fault appears at node 4
    # 20..35  : propagates outward
    # 36..47  : detection alert, hub highlighted

    def update(i):
        phase_idle  = i < 12
        phase_fault = 12 <= i < 20
        phase_prop  = 20 <= i < 36
        phase_det   = i >= 36

        t_prop = max(0, (i - 20) / 16)   # 0→1 over propagation phase

        for ni in range(8):
            d = dist.get(ni, 99)
            if phase_idle:
                node_circles[ni].set_facecolor(DIM)
                node_circles[ni].set_edgecolor('none')
                node_circles[ni].set_linewidth(0)
            elif phase_fault and ni == 4:
                blink = 0.5 + 0.5 * np.sin(i * 1.8)
                node_circles[ni].set_facecolor(
                    plt.matplotlib.colors.to_rgba(CU, blink))
                node_circles[ni].set_edgecolor(CU)
                node_circles[ni].set_linewidth(1.5)
            elif (phase_prop or phase_det):
                # fault wave reaches node when t_prop > d/max_dist
                reached = t_prop > d / (max_dist + 0.5)
                if ni == 4:
                    node_circles[ni].set_facecolor(CU)
                    node_circles[ni].set_edgecolor(CU2)
                    node_circles[ni].set_linewidth(1.5)
                elif reached and phase_det and ni == 0:
                    # hub detects — flash alert
                    blink = 0.5 + 0.5 * np.sin(i * 2.5)
                    node_circles[ni].set_facecolor(
                        plt.matplotlib.colors.to_rgba("#E8C45A", blink))
                    node_circles[ni].set_edgecolor("#E8C45A")
                    node_circles[ni].set_linewidth(2.0)
                elif reached:
                    node_circles[ni].set_facecolor(CU2)
                    node_circles[ni].set_edgecolor(CU)
                    node_circles[ni].set_linewidth(1.0)
                else:
                    node_circles[ni].set_facecolor(DIM)
                    node_circles[ni].set_edgecolor('none')
                    node_circles[ni].set_linewidth(0)
            else:
                node_circles[ni].set_facecolor(DIM)
                node_circles[ni].set_edgecolor('none')
                node_circles[ni].set_linewidth(0)

        # edge colours follow fault wave
        for a, b, l in edge_arts:
            da, db = dist.get(a,99), dist.get(b,99)
            d_edge = min(da, db)
            if phase_prop or phase_det:
                reached = t_prop > d_edge / (max_dist + 0.5)
                l.set_color(CU if reached else DIM)
                l.set_alpha(0.75 if reached else 0.35)
            else:
                l.set_color(DIM); l.set_alpha(0.4)

        if phase_idle:
            status_txt.set_text("monitoring . . .")
            status_txt.set_color(DIM)
        elif phase_fault:
            status_txt.set_text("⚠  anomaly detected at N4")
            status_txt.set_color(CU)
        elif phase_prop:
            status_txt.set_text("propagation mapping . . .")
            status_txt.set_color(CU)
        elif phase_det:
            status_txt.set_text("✓  fault isolated  —  rerouting")
            status_txt.set_color("#E8C45A")

    save_gif(fig, update, os.path.join(OUT, "gif-diagnostics.gif"))


if __name__ == "__main__":
    print("Generating GIFs …")
    make_pde_gif()
    make_learning_gif()
    make_nonlinear_gif()
    make_diagnostics_gif()
    print("Done.")
