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

    surf_h       = [None]
    contour_arts = []
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

        # Remove previous contour lines
        for line in contour_arts:
            try: line.remove()
            except: pass
        contour_arts.clear()

        if surf_h[0] is not None:
            surf_h[0].remove()

        surf_h[0] = ax.plot_surface(
            X, Y, Z_plot,
            cmap=cmap, vmin=-1.0, vmax=1.0,
            rcount=130, ccount=130,
            linewidth=0, antialiased=True,
        )

        # ── Outer contour: single connected four-lobe outline ────────────────
        # Split compound paths at MOVETO codes → keep only the longest loop
        # (outer boundary); discards inner hole and seam segment.
        if alpha > 0.45:
            from matplotlib.path import Path as MplPath
            fig_tmp, ax_tmp = plt.subplots()
            cs = ax_tmp.contour(X, Y, G_anim * alpha, levels=[0.050])
            all_loops = []
            for coll in cs.collections:
                for path in coll.get_paths():
                    verts, codes = path.vertices, path.codes
                    if codes is None:
                        all_loops.append(verts.copy())
                    else:
                        cuts = [0] + [k for k in range(1, len(codes))
                                      if codes[k] == MplPath.MOVETO]
                        cuts.append(len(codes))
                        for a, b in zip(cuts, cuts[1:]):
                            seg = verts[a:b]
                            if len(seg) > 3:
                                all_loops.append(seg.copy())
            plt.close(fig_tmp)
            if all_loops:
                outer = max(all_loops, key=lambda v: len(v))
                contour_arts.extend(
                    ax.plot(outer[:, 0], outer[:, 1], 0.02,
                            color=CU, linewidth=3.5,
                            linestyle=(0, (3, 4)),   # dotted: 3 on, 4 off
                            zorder=6)
                )

    save_gif(fig, update, os.path.join(OUT, 'gif-pde-control.gif'), n=80)


# ─────────────────────────────────────────────────────────────────────────────
# GIF 5 — Control block diagram + synchronized PDE surface (same frequency)
# ─────────────────────────────────────────────────────────────────────────────
def make_control_diagram_gif():
    from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
    from matplotlib.colors import LinearSegmentedColormap as LSC
    from matplotlib.patches import FancyBboxPatch, Circle

    N_FRAMES    = 80     # same count as gif-pde-control.gif
    OMEGA       = 8.0   # rad/s — shared by arrows and PDE surface

    # ── Full PDE physics (mirrors gif-pde-control.gif) ────────────────────────
    Ns  = 100
    xs  = np.linspace(0, 1, Ns); ys = np.linspace(0, 1, Ns)
    Xs, Ys = np.meshgrid(xs, ys)

    c_s  = 1.0; dx_s = xs[1]-xs[0]
    dt_s = 0.36 * dx_s / np.sqrt(2); r_s = (c_s*dt_s/dx_s)**2
    gamma_s = 0.8

    margin_s = 12
    sponge_s = np.ones((Ns, Ns))
    for k in range(margin_s):
        f = 1.0 - 0.06*(margin_s-k)/margin_s
        sponge_s[k,:]=np.minimum(sponge_s[k,:],f); sponge_s[-1-k,:]=np.minimum(sponge_s[-1-k,:],f)
        sponge_s[:,k]=np.minimum(sponge_s[:,k],f); sponge_s[:,-1-k]=np.minimum(sponge_s[:,-1-k],f)

    np.random.seed(7)
    u_s = np.zeros((Ns, Ns))
    for _ in range(28):
        cx=np.random.uniform(0.06,0.94); cy=np.random.uniform(0.06,0.94)
        amp=np.random.uniform(0.55,1.0)*np.random.choice([-1,1])
        sig=np.random.uniform(0.003,0.006)
        u_s += amp*np.exp(-((Xs-cx)**2+(Ys-cy)**2)/sig)
    u_s = np.clip(u_s,-1.0,1.0); u_prev_s = u_s.copy()

    sync_start_s = [(0.44,0.44),(0.56,0.44),(0.44,0.56),(0.56,0.56)]
    sync_end_s   = [(0.30,0.30),(0.70,0.30),(0.30,0.70),(0.70,0.70)]

    dist_s = np.sqrt((Xs-0.5)**2+(Ys-0.5)**2)
    R_s, Rf_s = 0.49, 0.022; R_inner_s = R_s - Rf_s
    circ_win_s = np.where(dist_s<=R_inner_s, 1.0,
                 np.where(dist_s<=R_s,
                          0.5*(1+np.cos(np.pi*(dist_s-R_inner_s)/Rf_s)), 0.0))
    circ_mask_s = dist_s > R_s

    cmap = LSC.from_list('site', [
        (10/255,11/255,16/255),(45/255,28/255,10/255),
        (208/255,138/255,79/255),(236/255,231/255,220/255)], N=256)

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(W/DPI, H/DPI))
    fig.patch.set_facecolor(BG)

    # Left: block diagram (38 % of width — compact)
    ax_d = fig.add_axes([0.01, 0.0, 0.38, 1.0])
    ax_d.set_facecolor(BG); ax_d.set_xlim(0,1); ax_d.set_ylim(0,1)
    ax_d.axis('off')

    # Right: mini 3-D surface (62 % of width — larger)
    ax_s = fig.add_axes([0.37, 0.0, 0.63, 1.0], projection='3d')
    ax_s.set_facecolor(BG); ax_s.set_axis_off()
    for axis in (ax_s.xaxis, ax_s.yaxis, ax_s.zaxis):
        axis.pane.fill = False; axis.pane.set_visible(False)
        axis.pane.set_edgecolor('none'); axis.line.set_linewidth(0)
    ax_s.set_zlim(-1.1,1.1); ax_s.set_xlim(0.04,0.96); ax_s.set_ylim(0.04,0.96)
    ax_s.view_init(elev=40, azim=45)

    # ── Φ–P–K Generalized Plant (robust control LFT form) ────────────────────
    # Block extents (ax_d data coords 0–1 × 0–1)
    PX1,PY1,PX2,PY2 = 0.22, 0.36, 0.78, 0.64   # P
    KX1,KY1,KX2,KY2 = 0.32, 0.05, 0.68, 0.18   # K
    FX1,FY1,FX2,FY2 = 0.32, 0.82, 0.68, 0.95   # Φ

    VX, QX = 0.36, 0.64   # P top ports  (v←Φ,  q→Φ)
    UX, YX = 0.36, 0.64   # P bottom ports (u←K, y→K)
    WY, ZY = 0.50, 0.50   # P left/right exogenous ports

    def sline(xs, ys, ls='-', lw=0.8):
        ax_d.plot(xs, ys, color=CU, lw=lw, linestyle=ls, zorder=1)
    def sarrow(x, y, m):
        ax_d.plot([x],[y], m, color=CU, ms=4, zorder=2)
    def block(x1,y1,x2,y2, symbol, highlight=False):
        fc = CU if highlight else '#11131B'
        tc = '#0A0B10' if highlight else HI
        ax_d.add_patch(FancyBboxPatch(
            (x1,y1), x2-x1, y2-y1,
            boxstyle='round,pad=0.015',
            facecolor=fc, edgecolor=CU, lw=1.0, zorder=2))
        ax_d.text((x1+x2)/2, (y1+y2)/2, symbol,
                  ha='center', va='center', fontsize=15,
                  color=tc, zorder=3)
    def sig(x, y, symbol, ha='center', va='center'):
        ax_d.text(x, y, symbol, ha=ha, va=va, fontsize=10,
                  color=CU_RGB.tolist(), zorder=5)

    # ── Wires ─────────────────────────────────────────────────────────────────
    sline([0.01, PX1], [WY, WY]);        sarrow(PX1+0.005, WY,  '>')
    sline([PX2,  0.97],[ZY, ZY]);        sarrow(0.965,      ZY,  '>')
    sline([QX, QX], [PY2, FY1]);         sarrow(QX, FY1+0.005,  '^')
    sline([VX, VX], [FY1, PY2]);         sarrow(VX, PY2-0.005,  'v')
    sline([YX, YX], [PY1, KY2]);         sarrow(YX, KY2+0.005,  'v')
    sline([UX, UX], [KY2, PY1]);         sarrow(UX, PY1-0.005,  '^')

    # ── Blocks ────────────────────────────────────────────────────────────────
    block(PX1,PY1,PX2,PY2, r'$P_{\infty}$')
    block(KX1,KY1,KX2,KY2, r'$\mathcal{K}$', highlight=True)
    block(FX1,FY1,FX2,FY2, r'$\Phi_{\infty}$')

    # ── Signal symbols only (mathtext, no text labels) ────────────────────────
    sig(0.01,  WY + 0.08,            r'$w$',  ha='left')   # above the w wire
    sig(0.88,  ZY + 0.08,            r'$z$',  ha='center') # just before right edge
    sig(VX-0.03, (PY2+FY1)/2,       r'$v$',  ha='right')
    sig(QX+0.03, (PY2+FY1)/2,       r'$q$',  ha='left')
    sig(UX-0.03, (PY1+KY2)/2,       r'$u$',  ha='right')
    sig(YX+0.03, (PY1+KY2)/2,       r'$y$',  ha='left')

    # ── Traveling pulse dots — one per wire, all synchronised ────────────────
    def lerp_path(pts, t):
        segs = list(zip(pts[:-1], pts[1:]))
        lens = [np.hypot(b[0]-a[0], b[1]-a[1]) for a,b in segs]
        total = sum(lens)
        if total == 0: return pts[0]
        tgt = t*total; cum = 0.0
        for (a,b),l in zip(segs, lens):
            if cum+l >= tgt:
                f = (tgt-cum)/l if l>0 else 0.0
                return (a[0]+f*(b[0]-a[0]), a[1]+f*(b[1]-a[1]))
            cum += l
        return pts[-1]

    # Six wires — each dot stays on its own arrow, never enters a block
    wires = [
        [(0.01,  WY),  (PX1,   WY)],    # w  → P  (left)
        [(PX2,   ZY),  (0.97,  ZY)],    # P  → z  (right)
        [(YX,   PY1),  (YX,   KY2)],    # P  → K  (y, down)
        [(UX,   KY2),  (UX,   PY1)],    # K  → P  (u, up)
        [(QX,   PY2),  (QX,   FY1)],    # P  → Φ  (q, up)
        [(VX,   FY1),  (VX,   PY2)],    # Φ  → P  (v, down)
    ]

    # One dot + glow per wire
    wire_dots = []
    for _ in wires:
        dot,  = ax_d.plot([], [], 'o', color=CU, ms=5.5, zorder=8)
        glow, = ax_d.plot([], [], 'o', color=CU, ms=12,  zorder=7, alpha=0)
        wire_dots.append((dot, glow))

    surf_h       = [None]
    contour_arts_s = []
    t_phys  = [0.0]
    state_s = {'u': u_s.copy(), 'up': u_prev_s.copy()}

    def update(i):
        # ── PDE physics (identical to gif-pde-control.gif) ───────────────────
        u_  = state_s['u']; up_ = state_s['up']
        alpha = float(np.clip(i/55.0, 0.0, 1.0))**2.2
        t_pos = float(np.clip(i/55.0, 0.0, 1.0))
        t_sm  = t_pos*t_pos*(3.0-2.0*t_pos)

        sigma_s = 0.005 + 0.007*t_sm
        G_anim_s = np.zeros_like(Xs)
        for (sx,sy),(ex,ey) in zip(sync_start_s, sync_end_s):
            px=sx+(ex-sx)*t_sm; py=sy+(ey-sy)*t_sm
            G_anim_s += np.exp(-((Xs-px)**2+(Ys-py)**2)/sigma_s)
        G_anim_s /= G_anim_s.max()

        denom_s = 1.0 + gamma_s*dt_s
        for _ in range(12):
            t_phys[0] += dt_s
            u_new = np.zeros_like(u_)
            u_new[1:-1,1:-1] = (
                (2*u_[1:-1,1:-1] - up_[1:-1,1:-1]*(1.0-gamma_s*dt_s)
                 + r_s*(u_[2:,1:-1]+u_[:-2,1:-1]+u_[1:-1,2:]+u_[1:-1,:-2]
                        -4*u_[1:-1,1:-1])
                ) / denom_s)
            u_new *= sponge_s; up_[:]=u_[:]; u_[:]=u_new

        u_sync_s  = G_anim_s * np.sin(OMEGA*t_phys[0]) * 0.88
        u_display_s = (1.0-alpha)**1.4 * np.clip(u_,-1,1) + alpha*u_sync_s
        Z_p = (u_display_s * circ_win_s).astype(float); Z_p[circ_mask_s]=np.nan

        for c in contour_arts_s:
            try: c.remove()
            except: pass
        contour_arts_s.clear()

        if surf_h[0] is not None:
            surf_h[0].remove()
        surf_h[0] = ax_s.plot_surface(
            Xs, Ys, Z_p, cmap=cmap, vmin=-1.0, vmax=1.0,
            rcount=80, ccount=80, linewidth=0, antialiased=True)

        # Outer contour — same as gif-pde-control.gif
        if alpha > 0.45:
            from matplotlib.path import Path as MplPath
            fig_tmp, ax_tmp = plt.subplots()
            cs = ax_tmp.contour(Xs, Ys, G_anim_s * alpha, levels=[0.050])
            all_loops = []
            for coll in cs.collections:
                for path in coll.get_paths():
                    verts, codes = path.vertices, path.codes
                    if codes is None:
                        all_loops.append(verts.copy())
                    else:
                        cuts = [0] + [k for k in range(1, len(codes))
                                      if codes[k] == MplPath.MOVETO]
                        cuts.append(len(codes))
                        for a2, b2 in zip(cuts, cuts[1:]):
                            seg = verts[a2:b2]
                            if len(seg) > 3:
                                all_loops.append(seg.copy())
            plt.close(fig_tmp)
            if all_loops:
                outer = max(all_loops, key=lambda v: len(v))
                contour_arts_s.extend(
                    ax_s.plot(outer[:, 0], outer[:, 1], 0.02,
                              color=CU, linewidth=2.5,
                              linestyle=(0, (3, 4)), zorder=6)
                )

        # ── Arrow pulse dots — slower independent speed (1.5 loops / GIF) ──────
        phi_dot = i * 2.5 * 2 * np.pi / N_FRAMES
        phi = OMEGA * t_phys[0]   # PDE surface phase (unchanged)
        t_m  = (phi           % (2*np.pi)) / (2*np.pi)
        t_ff = ((phi+np.pi/2) % (2*np.pi)) / (2*np.pi)
        t_fb = ((phi-np.pi/2) % (2*np.pi)) / (2*np.pi)

        # Causal signal-flow cascade — each wire has a phase offset that
        # reflects when the signal actually propagates in a feedback loop:
        #   w→P first, then P emits y/q, then K/Φ process and return u/v,
        #   finally z exits P.
        #
        #  wire index:  0=w→P  1=P→z  2=P→y  3=K→u  4=P→q  5=Φ→v
        causal_offsets = [
            0,                      # 0  w → P   (input enters)
            2*np.pi * 3/5,          # 1  P → z   (output after full loop)
            2*np.pi * 1/5,          # 2  P → K   (y measurement leaves P)
            2*np.pi * 2/5,          # 3  K → P   (u control returns)
            2*np.pi * 1/5,          # 4  P → Φ   (q leaves P, same as y)
            2*np.pi * 2/5,          # 5  Φ → P   (v returns, same as u)
        ]

        for (dot, glow), wire, offset in zip(wire_dots, wires, causal_offsets):
            ph  = phi_dot + offset
            t_n = (ph % (2*np.pi)) / (2*np.pi)
            x, y = lerp_path(wire, t_n)
            dot.set_data([x], [y]);  glow.set_data([x], [y])
            a = 0.40 + 0.60*(0.5 + 0.5*np.sin(ph))
            dot.set_alpha(a);        glow.set_alpha(a * 0.25)

    save_gif(fig, update,
             os.path.join(OUT, 'gif-control-diagram.gif'), n=N_FRAMES)


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
    def to_ax(v):  return 0.50 + v * 0.53

    xt_ax = to_ax(xt);  yt_ax = to_ax(yt)

    # sparse noisy observations
    idx = np.arange(0, 200, 7)
    xo  = np.clip(xt_ax[idx] + 0.025 * np.random.randn(len(idx)), 0.05, 0.95)
    yo  = np.clip(yt_ax[idx] + 0.025 * np.random.randn(len(idx)), 0.05, 0.95)

    fig, ax = make_fig()

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
    mag0     = mag(log_f, 0.02)
    max_mag  = mag0.max()
    mag0_n   = np.clip(mag0 / max_mag, 0, 1) * 0.86 + 0.07

    fig, ax = make_fig()

    # faint linear reference
    ax.plot(f_norm, mag0_n, color=DIM_RGB, lw=1.0, alpha=0.35,
            linestyle='--', zorder=2)

    main_line, = ax.plot([], [], color=CU, lw=2.2, zorder=5,
                         solid_capstyle='round')

    add_vignette(ax)

    def update(i):
        A   = amps[i]
        m   = mag(log_f, A)
        m_n = np.clip(m / max_mag, 0, 1) * 0.86 + 0.07
        main_line.set_data(f_norm, m_n)

        for c_ in [a for a in ax.collections if a.zorder < 10]:
            c_.remove()
        ax.fill_between(f_norm, 0.05, m_n,
                        alpha=0.11, color=CU, zorder=3)

    save_gif(fig, update, os.path.join(OUT, 'gif-nonlinear.gif'))


# ─────────────────────────────────────────────────────────────────────────────
# GIF 4 — System-Level Diagnostics  ·  fault propagation & detection
# ─────────────────────────────────────────────────────────────────────────────
def make_diagnostics_gif():
    # node positions in axes coords
    nodes = np.array([
        [0.50, 0.90],   # 0 — hub
        [0.20, 0.55],   # 1
        [0.50, 0.55],   # 2
        [0.80, 0.55],   # 3
        [0.08, 0.12],   # 4  ← fault origin
        [0.35, 0.12],   # 5
        [0.65, 0.12],   # 6
        [0.92, 0.12],   # 7
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


    save_gif(fig, update, os.path.join(OUT, 'gif-diagnostics.gif'))


# ─────────────────────────────────────────────────────────────────────────────
# GIF 4b — Chaotic machine network · real engineering schematic symbols
#           20 nodes (no hub) · 55 edges · 4 corner probes launch simultaneous
#           BFS; false leads fade, winning path to pump fault stays lit
# ─────────────────────────────────────────────────────────────────────────────
def make_diagnostics_complex_gif():
    from collections import deque

    AR = W / H  # 1.5 — aspect-ratio correction for visually symmetric symbols

    # ── Chaotic node layout — 20 nodes, no grid, mixed symbol types ───────────
    nodes = np.array([
        [0.09, 0.85],   #  0  spring      ← probe
        [0.28, 0.90],   #  1  capacitor
        [0.51, 0.86],   #  2  opamp
        [0.73, 0.90],   #  3  resistor    ← probe
        [0.93, 0.81],   #  4  sensor      ← probe
        [0.17, 0.67],   #  5  controller
        [0.40, 0.70],   #  6  inductor
        [0.63, 0.68],   #  7  valve
        [0.88, 0.62],   #  8  spring
        [0.06, 0.49],   #  9  opamp       ← probe
        [0.27, 0.50],   # 10  damper
        [0.50, 0.52],   # 11  capacitor
        [0.72, 0.50],   # 12  resistor
        [0.93, 0.47],   # 13  sensor
        [0.14, 0.31],   # 14  inductor
        [0.37, 0.32],   # 15  spring
        [0.59, 0.29],   # 16  opamp
        [0.82, 0.30],   # 17  valve
        [0.22, 0.11],   # 18  damper
        [0.52, 0.11],   # 19  pump        ← FAULT
    ])

    sym_types = [
        'spring','capacitor','opamp','resistor','sensor',
        'controller','inductor','valve','spring','opamp',
        'sensor','capacitor','resistor','sensor','inductor',
        'spring','opamp','valve','motor','pump',
    ]

    FAULT  = 19
    PROBES = [0, 3, 4, 9]   # four monitoring nodes at corners — no hub

    edges = [
        # local horizontal links
        (0,1),(1,2),(2,3),(3,4),
        (5,6),(6,7),(7,8),
        (9,10),(10,11),(11,12),(12,13),
        (14,15),(15,16),(16,17),
        (18,19),
        # downward struts
        (0,5),(1,6),(2,6),(2,7),(3,7),(4,8),
        (5,9),(5,10),(6,10),(6,11),(7,11),(7,12),(8,12),(8,13),
        (9,14),(10,14),(10,15),(11,15),(11,16),(12,16),(12,17),(13,17),
        (14,18),(15,18),(15,19),(16,19),(17,19),
        # long-range chaos (create crossings)
        (0,9),(1,5),(3,8),(4,13),
        (2,11),(4,12),(0,6),
        (6,15),(7,16),(8,17),
        (10,16),(11,17),
        (13,16),
    ]

    # deduplicate while preserving order
    seen = set(); edges_uniq = []
    for e in edges:
        key = (min(e), max(e))
        if key not in seen:
            seen.add(key); edges_uniq.append(e)
    edges = edges_uniq

    adj = {i: set() for i in range(len(nodes))}
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)

    def bfs_from(src):
        d = {src: 0}; q = deque([src])
        while q:
            v = q.popleft()
            for u in adj[v]:
                if u not in d:
                    d[u] = d[v] + 1; q.append(u)
        return d

    # multi-source: minimum distance from ANY probe to each node
    dist_probes  = {p: bfs_from(p) for p in PROBES}
    dist_any     = {v: min(dist_probes[p].get(v, 99) for p in PROBES)
                    for v in range(len(nodes))}
    max_dist_any = max(dist_any.values())

    # winning probe = closest to fault
    winning = min(PROBES, key=lambda p: dist_probes[p].get(FAULT, 99))

    # shortest path: winning probe → fault
    parent_w = {winning: None}; q = deque([winning])
    while q:
        v = q.popleft()
        if v == FAULT: break
        for u in adj[v]:
            if u not in parent_w:
                parent_w[u] = v; q.append(u)
    true_path = []
    v = FAULT
    while v is not None:
        true_path.append(v); v = parent_w.get(v)
    true_path.reverse()
    true_path_set   = set(true_path)
    true_path_edges = {(a, b) for a, b in zip(true_path[:-1], true_path[1:])}
    true_path_edges |= {(b, a) for (a, b) in true_path_edges}

    fig, ax = make_fig()

    # ── Engineering schematic symbol drawers ──────────────────────────────────
    def draw_controller(cx, cy, s, col, lw):
        sx, sy = s, s * AR
        arts = []
        arts += ax.plot([cx-sx,cx+sx,cx+sx,cx-sx,cx-sx],
                        [cy-sy*.60,cy-sy*.60,cy+sy*.60,cy+sy*.60,cy-sy*.60],
                        color=col, lw=lw, solid_capstyle='round', zorder=5)
        arts += ax.plot([cx-sx*.45,cx+sx*.45,cx+sx*.45,cx-sx*.45,cx-sx*.45],
                        [cy-sy*.28,cy-sy*.28,cy+sy*.28,cy+sy*.28,cy-sy*.28],
                        color=col, lw=lw*.60, zorder=5)
        for dx in [-sx*.42, 0, sx*.42]:
            arts += ax.plot([cx+dx,cx+dx],[cy+sy*.60,cy+sy*.82], color=col,lw=lw*.70,zorder=5)
            arts += ax.plot([cx+dx,cx+dx],[cy-sy*.82,cy-sy*.60], color=col,lw=lw*.70,zorder=5)
        return arts

    def draw_opamp(cx, cy, s, col, lw):
        sx, sy = s, s * AR
        arts = []
        arts += ax.plot([cx-sx,cx-sx,cx+sx,cx-sx],
                        [cy+sy*.76,cy-sy*.76,cy,cy+sy*.76],
                        color=col, lw=lw, solid_capstyle='round', zorder=5)
        arts += ax.plot([cx-sx,cx-sx*.70],[cy+sy*.30,cy+sy*.30], color=col,lw=lw*.70,zorder=5)
        arts += ax.plot([cx-sx*.85,cx-sx*.85],[cy+sy*.15,cy+sy*.45], color=col,lw=lw*.70,zorder=5)
        arts += ax.plot([cx-sx,cx-sx*.70],[cy-sy*.30,cy-sy*.30], color=col,lw=lw*.70,zorder=5)
        return arts

    def draw_resistor(cx, cy, s, col, lw):
        sy = s * AR * 0.36
        arts = []
        arts += ax.plot([cx-s,cx-s*.55],[cy,cy], color=col,lw=lw,zorder=5)
        arts += ax.plot([cx+s*.55,cx+s],[cy,cy], color=col,lw=lw,zorder=5)
        arts += ax.plot([cx-s*.55,cx+s*.55,cx+s*.55,cx-s*.55,cx-s*.55],
                        [cy-sy,cy-sy,cy+sy,cy+sy,cy-sy], color=col,lw=lw,zorder=5)
        return arts

    def draw_capacitor(cx, cy, s, col, lw):
        sy = s * AR
        arts = []
        arts += ax.plot([cx-s,cx-s*.14],[cy,cy], color=col,lw=lw,zorder=5)
        arts += ax.plot([cx+s*.14,cx+s],[cy,cy], color=col,lw=lw,zorder=5)
        arts += ax.plot([cx-s*.14,cx-s*.14],[cy-sy*.52,cy+sy*.52], color=col,lw=lw*2.0,zorder=5)
        arts += ax.plot([cx+s*.14,cx+s*.14],[cy-sy*.52,cy+sy*.52], color=col,lw=lw*2.0,zorder=5)
        return arts

    def draw_inductor(cx, cy, s, col, lw):
        arts = []
        arts += ax.plot([cx-s,cx-s*.72],[cy,cy], color=col,lw=lw,zorder=5)
        arts += ax.plot([cx+s*.72,cx+s],[cy,cy], color=col,lw=lw,zorder=5)
        t = np.linspace(0, np.pi, 14)
        for k in range(3):
            off = -s*.55 + k*s*.37
            arts += ax.plot(cx+off+s*.185*np.cos(t), cy+s*AR*.30*np.sin(t),
                            color=col, lw=lw, zorder=5)
        return arts

    def draw_spring(cx, cy, s, col, lw):
        sy = s * AR * 0.32
        arts = []
        arts += ax.plot([cx-s,cx-s*.85],[cy,cy], color=col,lw=lw,zorder=5)
        arts += ax.plot([cx+s*.85,cx+s],[cy,cy], color=col,lw=lw,zorder=5)
        xs = np.linspace(cx-s*.85, cx+s*.85, 7)
        ys = np.array([cy + sy*((-1)**k) for k in range(7)])
        arts += ax.plot(xs, ys, color=col, lw=lw, zorder=5)
        return arts

    def draw_damper(cx, cy, s, col, lw):
        sy = s * AR * 0.46
        arts = []
        arts += ax.plot([cx-s,cx-s*.10],[cy,cy], color=col,lw=lw,zorder=5)
        arts += ax.plot([cx-s*.10,cx-s*.10],[cy-sy,cy+sy], color=col,lw=lw*2.2,zorder=5)
        arts += ax.plot([cx-s*.10,cx+s*.58],[cy+sy,cy+sy], color=col,lw=lw,zorder=5)
        arts += ax.plot([cx-s*.10,cx+s*.58],[cy-sy,cy-sy], color=col,lw=lw,zorder=5)
        arts += ax.plot([cx+s*.58,cx+s*.58],[cy-sy,cy+sy], color=col,lw=lw,zorder=5)
        arts += ax.plot([cx+s*.58,cx+s],[cy,cy], color=col,lw=lw,zorder=5)
        return arts

    def draw_sensor(cx, cy, s, col, lw):
        sx, sy = s*.56, s*.56*AR
        arts = []
        t = np.linspace(0, 2*np.pi, 36)
        arts += ax.plot(cx+sx*np.cos(t), cy+sy*np.sin(t), color=col,lw=lw,zorder=5)
        arts += ax.plot([cx-sx*.62,cx+sx*.62],[cy-sy*.62,cy+sy*.62], color=col,lw=lw*.70,zorder=5)
        arts += ax.plot([cx+sx*.62,cx-sx*.62],[cy-sy*.62,cy+sy*.62], color=col,lw=lw*.70,zorder=5)
        return arts

    def draw_pump(cx, cy, s, col, lw):
        sx, sy = s*.56, s*.56*AR
        arts = []
        t = np.linspace(0, 2*np.pi, 36)
        arts += ax.plot(cx+sx*np.cos(t), cy+sy*np.sin(t), color=col,lw=lw,zorder=5)
        t2 = np.linspace(.20, 5.50, 22); ri = .52
        arts += ax.plot(cx+sx*ri*np.cos(t2), cy+sy*ri*np.sin(t2), color=col,lw=lw,zorder=5)
        ae, da = t2[-1], 0.50
        arts += ax.plot([cx+sx*ri*np.cos(ae-da), cx+sx*ri*np.cos(ae),
                         cx+sx*ri*np.cos(ae-da)],
                        [cy+sy*ri*np.sin(ae-da), cy+sy*ri*np.sin(ae),
                         cy+sy*ri*np.sin(ae+.28)],
                        color=col, lw=lw, zorder=5)
        return arts

    def draw_motor(cx, cy, s, col, lw):
        sx, sy = s*.56, s*.56*AR
        arts = []
        t = np.linspace(0, 2*np.pi, 36)
        arts += ax.plot(cx+sx*np.cos(t), cy+sy*np.sin(t), color=col,lw=lw,zorder=5)
        arts += ax.plot([cx-s,cx-sx],[cy,cy], color=col,lw=lw,zorder=5)
        arts += ax.plot([cx+sx,cx+s],[cy,cy], color=col,lw=lw,zorder=5)
        tb = np.linspace(0, np.pi, 10); rb = sx*.24
        for k in range(3):
            xo = cx + (k-1)*sx*.46
            arts += ax.plot(xo+rb*np.cos(tb), cy+sy*.16+rb*AR*np.sin(tb),
                            color=col,lw=lw,zorder=5)
        return arts

    def draw_valve(cx, cy, s, col, lw):
        # Globe valve: circle body + horizontal disc plate + pipe stubs + actuator stem
        sx, sy = s*.50, s*.50*AR
        arts = []
        t = np.linspace(0, 2*np.pi, 36)
        arts += ax.plot(cx+sx*np.cos(t), cy+sy*np.sin(t), color=col,lw=lw,zorder=5)
        arts += ax.plot([cx-sx,cx+sx],[cy,cy], color=col,lw=lw*1.6,zorder=5)
        arts += ax.plot([cx-s,cx-sx],[cy,cy], color=col,lw=lw,zorder=5)
        arts += ax.plot([cx+sx,cx+s],[cy,cy], color=col,lw=lw,zorder=5)
        arts += ax.plot([cx,cx],[cy+sy,cy+sy*1.55], color=col,lw=lw,zorder=5)
        arts += ax.plot([cx-sx*.42,cx+sx*.42],[cy+sy*1.55,cy+sy*1.55], color=col,lw=lw,zorder=5)
        return arts

    sym_drawers = {
        'controller':draw_controller,'opamp':draw_opamp,
        'resistor':draw_resistor,'capacitor':draw_capacitor,'inductor':draw_inductor,
        'spring':draw_spring,'damper':draw_damper,
        'sensor':draw_sensor,'pump':draw_pump,'valve':draw_valve,'motor':draw_motor,
    }
    sym_scale = {
        'controller':0.056,'opamp':0.040,
        'resistor':0.038,'capacitor':0.038,'inductor':0.038,
        'spring':0.036,'damper':0.036,
        'sensor':0.038,'pump':0.040,'valve':0.036,'motor':0.040,
    }

    # ── Draw edges ────────────────────────────────────────────────────────────
    # mark long-range chaos edges (last 13) as dashed
    N_LOCAL = len(edges) - 13
    edge_lines = []
    for ei, (a, b) in enumerate(edges):
        ic = ei >= N_LOCAL
        l, = ax.plot([nodes[a,0],nodes[b,0]], [nodes[a,1],nodes[b,1]],
                     color=DIM_RGB, lw=0.55 if ic else 0.85, alpha=0.28,
                     linestyle=(0,(4,3)) if ic else '-', zorder=2)
        edge_lines.append(l)

    # ── Draw node symbols (dim — full schematic visible at all times) ─────────
    DIM_RGBA   = (*DIM_RGB.tolist(), 0.42)
    PROBE_RGBA = (*CU_RGB.tolist(),  0.22)   # probes glow faintly even at idle

    node_artists = []
    for i in range(len(nodes)):
        cx, cy = nodes[i]
        s    = sym_scale[sym_types[i]]
        init = PROBE_RGBA if i in PROBES else DIM_RGBA
        arts = sym_drawers[sym_types[i]](cx, cy, s, init, 1.0)
        node_artists.append(arts)

    add_vignette(ax)

    def col_node(ni, rgba):
        for a in node_artists[ni]:
            a.set_color(rgba)

    # ── Animation ─────────────────────────────────────────────────────────────
    def update(frame):
        # wave front in dist-from-nearest-probe units, travels over frames 21–44
        search_wave = max(0.0, (frame - 21) * max_dist_any / 23.0)
        det = frame >= 45

        for ni in range(len(nodes)):
            d_any   = dist_any[ni]
            on_path = ni in true_path_set
            reached = search_wave >= d_any
            t_local = max(0.0, search_wave - d_any)
            is_probe = ni in PROBES

            if frame < 11:                              # idle
                col_node(ni, PROBE_RGBA if is_probe else DIM_RGBA)
            elif frame < 21:                            # fault blinks
                if ni == FAULT:
                    b_ = 0.55 + 0.45 * np.sin(frame * 1.6)
                    col_node(ni, mcolors.to_rgba(CU, b_))
                else:
                    col_node(ni, PROBE_RGBA if is_probe else DIM_RGBA)
            elif not det:                               # multi-source search
                if ni == FAULT:
                    col_node(ni, mcolors.to_rgba(CU, 1.0))
                elif not reached:
                    col_node(ni, PROBE_RGBA if is_probe else DIM_RGBA)
                elif on_path:                           # true path stays lit
                    col_node(ni, mcolors.to_rgba(CU, 1.0))
                else:                                   # false lead fades
                    fade = max(0.0, 1.0 - t_local / 1.5)
                    col_node(ni, mcolors.to_rgba(CU2, max(0.04, fade * 0.65)))
            else:                                       # fault isolated
                if ni == FAULT:
                    b_ = 0.55 + 0.45 * np.sin(frame * 1.6)
                    col_node(ni, mcolors.to_rgba(CU, b_))
                elif ni == winning:                     # winning probe pulses
                    b_ = 0.55 + 0.45 * np.sin(frame * 2.2)
                    col_node(ni, mcolors.to_rgba("#E8C45A", b_))
                elif on_path:
                    col_node(ni, mcolors.to_rgba(CU, 1.0))
                else:
                    col_node(ni, PROBE_RGBA if is_probe else DIM_RGBA)

        for ei, ((a, b), l) in enumerate(zip(edges, edge_lines)):
            ic        = ei >= N_LOCAL
            on_true   = (a, b) in true_path_edges
            d_edge    = min(dist_any[a], dist_any[b])
            reached_e = search_wave >= d_edge
            t_local_e = max(0.0, search_wave - d_edge)

            if frame < 21:
                l.set_color(DIM_RGB); l.set_alpha(0.25)
                l.set_linewidth(0.55 if ic else 0.85)
            elif not det:
                if not reached_e:
                    l.set_color(DIM_RGB); l.set_alpha(0.25)
                    l.set_linewidth(0.55 if ic else 0.85)
                elif on_true:
                    l.set_color(CU_RGB); l.set_alpha(0.82)
                    l.set_linewidth(0.90 if ic else 1.60)
                else:
                    fade = max(0.0, 1.0 - t_local_e / 1.5)
                    l.set_color(CU_RGB); l.set_alpha(fade * 0.42)
                    l.set_linewidth(0.45 if ic else 0.75)
            else:
                if on_true:
                    l.set_color(CU_RGB); l.set_alpha(0.88)
                    l.set_linewidth(1.00 if ic else 2.00)
                else:
                    l.set_color(DIM_RGB); l.set_alpha(0.15)
                    l.set_linewidth(0.45 if ic else 0.65)

    save_gif(fig, update, os.path.join(OUT, 'gif-diagnostics-complex.gif'))


if __name__ == '__main__':
    print('Generating GIFs …')
    make_pde_gif()
    make_control_diagram_gif()
    make_learning_gif()
    make_nonlinear_gif()
    make_diagnostics_gif()
    make_diagnostics_complex_gif()
    print('Done.')
