"""
Thematic GIFs styled to match the website:
  - Background  #0A0B10
  - Tron grid   rgba(208,138,79, 0.06) fine / 0.14 major
  - Vignette    radial dark overlay matching the hero CSS
  - Scanline    horizontal copper sweep
  - Copper      #D08A4F  |  Hi-text  #ECE7DC  |  Dim  #4A4C56
  - No axes, no tick marks, no text labels
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import io, os

# ── Palette (exact site values) ──────────────────────────────────────────────
BG  = "#0A0B10"
CU  = "#D08A4F"
CU2 = "#B8743D"
HI  = "#ECE7DC"
DIM = "#4A4C56"

CU_RGB  = np.array([208, 138,  79]) / 255
BG_RGB  = np.array([ 10,  11,  16]) / 255
DIM_RGB = np.array([ 74,  76,  86]) / 255
HI_RGB  = np.array([236, 231, 220]) / 255

W, H   = 420, 280      # pixel dimensions
DPI    = 90
FRAMES = 56
DELAY  = 55            # ms per frame  (~18 fps)
OUT    = os.path.dirname(os.path.abspath(__file__))


# ── Shared background helpers ─────────────────────────────────────────────────

def make_fig():
    fig, ax = plt.subplots(figsize=(W / DPI, H / DPI))
    fig.subplots_adjust(0, 0, 1, 1)          # zero margins
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return fig, ax


def draw_tron_grid(ax):
    """Faint copper grid matching the site's tron-lines CSS."""
    # fine grid every 54 px
    for x in np.arange(0, W, 54) / W:
        ax.axvline(x, color=CU_RGB, alpha=0.06, lw=0.5, zorder=1)
    for y in np.arange(0, H, 54) / H:
        ax.axhline(y, color=CU_RGB, alpha=0.06, lw=0.5, zorder=1)
    # major accent every 4 cells (216 px)
    for x in np.arange(0, W, 216) / W:
        ax.axvline(x, color=CU_RGB, alpha=0.14, lw=0.7, zorder=1)
    for y in np.arange(0, H, 216) / H:
        ax.axhline(y, color=CU_RGB, alpha=0.14, lw=0.7, zorder=1)


def make_vignette(W, H):
    """
    Matches CSS:
      radial-gradient(ellipse 72% 72% at 50% 50%,
        transparent 35%, rgba(10,11,16,0.78) 100%)
    Returns an RGBA uint8 array (H, W, 4).
    """
    xn = np.linspace(0, 1, W)
    yn = np.linspace(0, 1, H)
    X, Y = np.meshgrid(xn, yn)
    d = np.sqrt(((X - 0.5) / 0.72) ** 2 + ((Y - 0.5) / 0.72) ** 2)
    alpha = np.clip((d - 0.35) / 0.65, 0, 1) * 0.78
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    rgba[..., :3] = BG_RGB
    rgba[..., 3]  = alpha
    return rgba


VIGNETTE = make_vignette(W, H)


def add_vignette(ax):
    ax.imshow(VIGNETTE, extent=(0, 1, 0, 1), aspect='auto',
              zorder=15, interpolation='bilinear', origin='upper')


def scanline_y(frame, n_frames=FRAMES):
    """Y position (in axes coords 0-1, top→bottom) of the scan line."""
    t = frame / n_frames
    return 1.0 - t        # travels top to bottom


def add_scanline(ax, frame):
    y = scanline_y(frame)
    # centre bright, fade to transparent at sides
    ax.axhline(y, color=CU_RGB, alpha=0.22, lw=0.8, zorder=14,
               solid_capstyle='butt')
    # glow
    ax.axhline(y, color=CU_RGB, alpha=0.06, lw=4.0, zorder=13)


def save_gif(fig, update_fn, path, n=FRAMES):
    frames = []
    for i in range(n):
        update_fn(i)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=DPI,
                    facecolor=BG, edgecolor='none')
        buf.seek(0)
        img = Image.open(buf).copy()
        frames.append(img.convert('P', palette=Image.ADAPTIVE, colors=200))
    frames[0].save(
        path, save_all=True, append_images=frames[1:],
        loop=0, duration=DELAY, optimize=True
    )
    plt.close(fig)
    print(f"  saved {os.path.basename(path)}")


# ─────────────────────────────────────────────────────────────────────────────
# GIF 1 — Control of PDEs  ·  3-D surface, no box, chaotic → 4 synchronized peaks
# ─────────────────────────────────────────────────────────────────────────────
def make_pde_gif():
    from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
    from matplotlib.colors import LinearSegmentedColormap as LSC

    # ── Grid ─────────────────────────────────────────────────────────────────
    Nx = Ny = 150
    x  = np.linspace(0, 1, Nx)
    y  = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y)

    c  = 1.0
    dx = x[1] - x[0]
    dt = 0.36 * dx / np.sqrt(2)
    r  = (c * dt / dx) ** 2
    gamma = 0.8                    # damps chaotic physics; synced display takes over

    # Sponge at edges
    margin = 12
    sponge = np.ones((Ny, Nx))
    for k in range(margin):
        f = 1.0 - 0.06 * (margin - k) / margin
        sponge[k, :]    = np.minimum(sponge[k, :],    f)
        sponge[-1-k, :] = np.minimum(sponge[-1-k, :], f)
        sponge[:, k]    = np.minimum(sponge[:, k],    f)
        sponge[:, -1-k] = np.minimum(sponge[:, -1-k], f)

    # ── Chaotic initial state ─────────────────────────────────────────────────
    np.random.seed(7)
    u = np.zeros((Ny, Nx))
    for _ in range(28):
        cx  = np.random.uniform(0.06, 0.94)
        cy  = np.random.uniform(0.06, 0.94)
        amp = np.random.uniform(0.55, 1.0) * np.random.choice([-1, 1])
        sig = np.random.uniform(0.003, 0.006)
        u  += amp * np.exp(-((X - cx)**2 + (Y - cy)**2) / sig)
    u = np.clip(u, -1.0, 1.0)
    u_prev = u.copy()

    # ── Circular window — smooth cosine taper so the edge is perfectly round ─
    dist_c    = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
    R_max     = 0.49
    R_fade    = 0.022              # ~3 rendered cells at rcount=130 — smooths without visible fade
    R_inner   = R_max - R_fade
    circ_window = np.where(
        dist_c <= R_inner,
        1.0,
        np.where(
            dist_c <= R_max,
            0.5 * (1.0 + np.cos(np.pi * (dist_c - R_inner) / R_fade)),
            0.0
        )
    )
    circ_mask = dist_c > R_max     # hard NaN only beyond the taper

    # ── Synchronized template: bumps travel from centre toward corners ───────
    # Positions kept inside the disc (distance from centre ≤ 0.39 < 0.46)
    sync_start = [(0.44, 0.44), (0.56, 0.44), (0.44, 0.56), (0.56, 0.56)]
    sync_end   = [(0.30, 0.30), (0.70, 0.30), (0.30, 0.70), (0.70, 0.70)]
    omega_sync = 8.0               # rad/s — slow, majestic oscillation (~22 frames/cycle)

    # ── Colormap: site dark → warm dark → copper → cream ────────────────────
    cmap = LSC.from_list('site', [
        ( 10/255,  11/255,  16/255),
        ( 45/255,  28/255,  10/255),
        (208/255, 138/255,  79/255),
        (236/255, 231/255, 220/255),
    ], N=256)

    # ── Figure: dark background, 3-D axes with ALL box elements removed ──────
    fig = plt.figure(figsize=(W / DPI, H / DPI))
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=-0.18, right=1.18, bottom=-0.20, top=1.20)

    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(BG)
    ax.set_axis_off()
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_visible(False)
        axis.pane.set_edgecolor('none')
        axis.line.set_linewidth(0)
    ax.set_zlim(-1.1, 1.1)
    # Narrow xy limits to crop dead corners and zoom in on the disc
    ax.set_xlim(0.04, 0.96)
    ax.set_ylim(0.04, 0.96)
    ax.view_init(elev=38, azim=45)

    surf_h = [None]
    t_phys = [0.0]
    state  = {'u': u.copy(), 'up': u_prev.copy()}

    def update(i):
        u_  = state['u']
        up_ = state['up']

        # ── Physics: damped wave (chaos decays naturally) ─────────────────
        for _ in range(12):
            t_phys[0] += dt
            u_new = np.zeros_like(u_)
            denom = 1.0 + gamma * dt
            u_new[1:-1, 1:-1] = (
                (2 * u_[1:-1, 1:-1]
                 - up_[1:-1, 1:-1] * (1.0 - gamma * dt)
                 + r * (u_[2:,  1:-1] + u_[:-2, 1:-1]
                      + u_[1:-1, 2:  ] + u_[1:-1, :-2]
                      - 4 * u_[1:-1, 1:-1])
                ) / denom
            )
            u_new *= sponge
            up_[:] = u_[:]
            u_[:]  = u_new

        # ── Display: chaos slowly sinks, bumps travel toward corners ────────
        # alpha: power 2.2 → very gradual start, reaches 1 at frame 55
        alpha = float(np.clip(i / 55.0, 0.0, 1.0)) ** 2.2

        # Smoothstep for position travel (independent of alpha blend weight)
        t_pos = float(np.clip(i / 55.0, 0.0, 1.0))
        t_smooth = t_pos * t_pos * (3.0 - 2.0 * t_pos)   # smoothstep 0→1

        # Build animated Gaussian template: centres move, sigma grows
        sigma = 0.005 + 0.007 * t_smooth          # tight at start, broad at corners
        G_anim = np.zeros_like(X)
        for (sx, sy), (ex, ey) in zip(sync_start, sync_end):
            px = sx + (ex - sx) * t_smooth
            py = sy + (ey - sy) * t_smooth
            G_anim += np.exp(-((X - px)**2 + (Y - py)**2) / sigma)
        G_anim /= G_anim.max()

        u_sync    = G_anim * np.sin(omega_sync * t_phys[0]) * 0.88
        u_display = (1.0 - alpha) ** 1.4 * np.clip(u_, -1.0, 1.0) + alpha * u_sync

        # Smooth circular boundary: cosine taper to zero, then NaN outside
        Z_plot = (u_display * circ_window).astype(float)
        Z_plot[circ_mask] = np.nan

        if surf_h[0] is not None:
            surf_h[0].remove()

        surf_h[0] = ax.plot_surface(
            X, Y, Z_plot,
            cmap=cmap, vmin=-1.0, vmax=1.0,
            rcount=130, ccount=130,
            linewidth=0, antialiased=True,
        )

    save_gif(fig, update, os.path.join(OUT, 'gif-pde-control.gif'), n=80)


# ─────────────────────────────────────────────────────────────────────────────
# GIF 2 — Physics-Informed Learning  ·  surrogate converging on a nonlinear orbit
# ─────────────────────────────────────────────────────────────────────────────
def make_learning_gif():
    np.random.seed(7)
    T = np.linspace(0, 2 * np.pi, 200)
    # true orbit (limit-cycle-ish, in [-1,1])
    xt = 0.72 * np.sin(T)
    yt = 0.62 * np.cos(T) * (1 + 0.30 * np.sin(2 * T))

    # map to axes [0.08, 0.92]
    def to_ax(v):  return 0.50 + v * 0.40

    xt_ax = to_ax(xt);  yt_ax = to_ax(yt)

    # sparse noisy observations
    idx = np.arange(0, 200, 7)
    xo  = np.clip(xt_ax[idx] + 0.025 * np.random.randn(len(idx)), 0.05, 0.95)
    yo  = np.clip(yt_ax[idx] + 0.025 * np.random.randn(len(idx)), 0.05, 0.95)

    fig, ax = make_fig()
    draw_tron_grid(ax)

    # dim true orbit (always visible, behind everything)
    ax.plot(xt_ax, yt_ax, color=DIM_RGB, lw=0.8, alpha=0.30,
            linestyle='--', zorder=2)

    # observations (static)
    ax.scatter(xo, yo, s=7, color=HI_RGB, alpha=0.55, zorder=4,
               linewidths=0)

    surr_line, = ax.plot([], [], color=CU, lw=2.2, zorder=5,
                         solid_capstyle='round')

    add_vignette(ax)

    def update(i):
        frac  = min(1.0, (i + 1) / (FRAMES * 0.80))
        n_pts = max(3, int(len(T) * frac))
        noise = (1 - frac) * 0.022
        xs = xt_ax[:n_pts] + noise * np.random.randn(n_pts)
        ys = yt_ax[:n_pts] + noise * np.random.randn(n_pts)
        surr_line.set_data(xs, ys)

        for c_ in [a for a in ax.collections if a.zorder < 10]:
            c_.remove()
        # uncertainty band (shrinks as model converges)
        w = (1 - frac) * 0.028
        ax.fill_between(xs, ys - w, ys + w,
                        alpha=0.12, color=CU, zorder=3)
        # re-draw observations on top of fill
        ax.scatter(xo, yo, s=7, color=HI_RGB, alpha=0.55, zorder=4,
                   linewidths=0)
        add_scanline(ax, i)

    save_gif(fig, update, os.path.join(OUT, 'gif-physics-learning.gif'))


# ─────────────────────────────────────────────────────────────────────────────
# GIF 3 — Nonlinear Performance Shaping  ·  amplitude-dependent Bode
# ─────────────────────────────────────────────────────────────────────────────
def make_nonlinear_gif():
    # log-frequency axis, normalised to [0,1]
    log_f  = np.linspace(-1, 1.5, 300)         # log10(freq)
    f_norm = (log_f - (-1)) / (1.5 - (-1))     # → [0, 1] for axes x

    amps = np.linspace(0.05, 2.8, FRAMES)

    def mag(log_f_, A):
        f = 10 ** log_f_
        omega_n = 1.0 + 0.38 * A
        zeta    = 0.14 + 0.10 * A
        s = f / omega_n
        return 1.0 / np.sqrt((1 - s**2)**2 + (2*zeta*s)**2 + 1e-12)

    # linear (A→0) reference normalised to axes y
    mag0   = mag(log_f, 0.02)
    mag0_n = np.clip(mag0 / 5.0, 0, 1) * 0.75 + 0.05

    fig, ax = make_fig()
    draw_tron_grid(ax)

    # faint linear reference
    ax.plot(f_norm, mag0_n, color=DIM_RGB, lw=1.0, alpha=0.35,
            linestyle='--', zorder=2)

    main_line, = ax.plot([], [], color=CU, lw=2.2, zorder=5,
                         solid_capstyle='round')

    add_vignette(ax)

    def update(i):
        A   = amps[i]
        m   = mag(log_f, A)
        m_n = np.clip(m / 5.0, 0, 1) * 0.75 + 0.05
        main_line.set_data(f_norm, m_n)

        for c_ in [a for a in ax.collections if a.zorder < 10]:
            c_.remove()
        ax.fill_between(f_norm, 0.05, m_n,
                        alpha=0.11, color=CU, zorder=3)
        add_scanline(ax, i)

    save_gif(fig, update, os.path.join(OUT, 'gif-nonlinear.gif'))


# ─────────────────────────────────────────────────────────────────────────────
# GIF 4 — System-Level Diagnostics  ·  fault propagation & detection
# ─────────────────────────────────────────────────────────────────────────────
def make_diagnostics_gif():
    # node positions in axes coords
    nodes = np.array([
        [0.50, 0.84],   # 0 — hub
        [0.24, 0.58],   # 1
        [0.50, 0.58],   # 2
        [0.76, 0.58],   # 3
        [0.13, 0.26],   # 4  ← fault origin
        [0.37, 0.26],   # 5
        [0.63, 0.26],   # 6
        [0.87, 0.26],   # 7
    ])
    edges = [(0,1),(0,2),(0,3),(1,4),(1,5),(2,5),(2,6),(3,6),(3,7)]

    # BFS distance from fault node 4
    from collections import deque
    dist = {4: 0}
    adj  = {i: [] for i in range(8)}
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
    q = deque([4])
    while q:
        v = q.popleft()
        for u in adj[v]:
            if u not in dist:
                dist[u] = dist[v] + 1
                q.append(u)
    max_d = max(dist.values())

    # node radius in axes coords
    R = 0.042

    fig, ax = make_fig()
    draw_tron_grid(ax)

    # pre-draw static edges
    edge_lines = []
    for a, b in edges:
        l, = ax.plot([nodes[a,0], nodes[b,0]],
                     [nodes[a,1], nodes[b,1]],
                     color=DIM_RGB, lw=1.0, alpha=0.35, zorder=2)
        edge_lines.append(l)

    # node patches
    from matplotlib.patches import Circle
    circles = []
    for i, (nx, ny) in enumerate(nodes):
        circ = Circle((nx, ny), R, facecolor=DIM, edgecolor='none',
                      linewidth=0, zorder=5, transform=ax.transData)
        ax.add_patch(circ)
        circles.append(circ)

    add_vignette(ax)

    # phases: 0-10 idle | 11-20 fault appears | 21-38 propagates | 39+ detect
    def update(i):
        idle  = i < 11
        fault = 11 <= i < 21
        prop  = 21 <= i < 39
        det   = i >= 39

        t_prop = max(0, (i - 21) / 18.0)

        for ni in range(8):
            d = dist.get(ni, 99)

            if idle:
                circles[ni].set_facecolor(DIM)
                circles[ni].set_edgecolor('none')
                circles[ni].set_linewidth(0)
            elif fault:
                if ni == 4:
                    blink = 0.55 + 0.45 * np.sin(i * 1.6)
                    circles[ni].set_facecolor(
                        mcolors.to_rgba(CU, blink))
                    circles[ni].set_edgecolor(CU)
                    circles[ni].set_linewidth(1.5)
                else:
                    circles[ni].set_facecolor(DIM)
                    circles[ni].set_edgecolor('none')
            elif prop or det:
                reached = t_prop > d / (max_d + 0.6)
                if ni == 4:
                    circles[ni].set_facecolor(CU)
                    circles[ni].set_edgecolor(CU2)
                    circles[ni].set_linewidth(1.5)
                elif ni == 0 and det:
                    # hub alert — warm yellow flash
                    blink = 0.55 + 0.45 * np.sin(i * 2.2)
                    circles[ni].set_facecolor(
                        mcolors.to_rgba("#E8C45A", blink))
                    circles[ni].set_edgecolor("#E8C45A")
                    circles[ni].set_linewidth(2.0)
                elif reached:
                    circles[ni].set_facecolor(CU2)
                    circles[ni].set_edgecolor(CU)
                    circles[ni].set_linewidth(1.0)
                else:
                    circles[ni].set_facecolor(DIM)
                    circles[ni].set_edgecolor('none')
                    circles[ni].set_linewidth(0)

        for (a, b), l in zip(edges, edge_lines):
            d_edge = min(dist.get(a,99), dist.get(b,99))
            if (prop or det) and t_prop > d_edge / (max_d + 0.6):
                l.set_color(CU_RGB)
                l.set_alpha(0.65)
                l.set_linewidth(1.3)
            else:
                l.set_color(DIM_RGB)
                l.set_alpha(0.32)
                l.set_linewidth(1.0)

        add_scanline(ax, i)

    save_gif(fig, update, os.path.join(OUT, 'gif-diagnostics.gif'))


if __name__ == '__main__':
    print('Generating GIFs …')
    make_pde_gif()
    make_learning_gif()
    make_nonlinear_gif()
    make_diagnostics_gif()
    print('Done.')
