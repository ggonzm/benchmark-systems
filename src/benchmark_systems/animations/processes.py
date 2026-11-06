"""Process-system animators."""

import matplotlib.pyplot as plt
import numpy as np

from ._core import _initialized, _matching_trajectories
from ._process_components import BubblePopulation, DialGauge, Mixer, VerticalBar


def JohanssonTanks(axis, *, h_max, heights, pumps, gammas):
    """Draw the crossed-feed Johansson four-tank process.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Two-dimensional axes on which to draw the scene.
    h_max : array_like of shape (4,)
        Positive finite maximum tank heights in cm, ordered tank 1 through tank
        4. These set the fill-height scales.
    heights : sequence of four array_like of shape (N,)
        Liquid-height trajectories in cm, ordered h1, h2, h3, h4.
    pumps : sequence of two array_like of shape (N,)
        Pump-command trajectories on the 0 to 100 percent scale. Pump labels
        show the commands, and branch labels show their shares on that same
        scale after applying the valve fractions.
    gammas : sequence of two array_like of shape (N,)
        Valve-split fractions gamma1 and gamma2, normally from 0 to 1.

    Returns
    -------
    update : callable
        Initialized callback ``update(frame)``. The zero-based integer frame
        selects a sample and the callback returns the modified Matplotlib
        artists. Frame zero is drawn before this factory returns.

    Notes
    -----
    All supplied trajectories must be finite, nonempty and have the same
    sample count ``N``. Repeat constant values over the sample grid.

    Pump 1 feeds tank 1 and, through its gamma1 branch, tank 4. Pump 2 feeds
    tank 2 and, through gamma2, tank 3. Tank 3 drains into tank 1 and tank 4
    into tank 2. Fill fractions are clipped for drawing, not for simulation.
    """
    import matplotlib.patches as _mp
    from matplotlib.patches import Circle, Arc

    h_max = np.asarray(h_max, dtype=float)
    if h_max.shape != (4,) or not np.isfinite(h_max).all() or np.any(h_max <= 0):
        raise ValueError("h_max must contain four positive finite values")
    if len(heights) != 4 or len(pumps) != 2 or len(gammas) != 2:
        raise ValueError("heights, pumps, and gammas must contain 4, 2, and 2 trajectories")
    trajectories = _matching_trajectories(
        **{f"height_{i}": value for i, value in enumerate(heights)},
        **{f"pump_{i}": value for i, value in enumerate(pumps)},
        **{f"gamma_{i}": value for i, value in enumerate(gammas)},
    )
    heights = tuple(trajectories[f"height_{i}"] for i in range(4))
    pumps = tuple(trajectories[f"pump_{i}"] for i in range(2))
    gammas = tuple(trajectories[f"gamma_{i}"] for i in range(2))
    ax = axis
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')

    # ── colours ──────────────────────────────────────────────────────────────
    C_WATER = '#b8d4ee'
    C_TANK  = '#223344'
    C_PIPE  = '#444444'
    C_PUMP  = '#e8e8e8'
    C_VALVE = '#cc3333'
    C_NUM   = '#0044bb'
    LW = 1.8

    # ── layout (all in plot units 0-10 x 0-12) ───────────────────────────────
    # Tank geometry
    TW  = 2.6   # tank width
    BTH = 3.2   # bottom tank plot height
    TTH = 2.4   # top tank plot height
    # Tank left-edge x positions
    T1lx = 1.5; T2lx = 5.9
    T1cx = T1lx + TW/2; T2cx = T2lx + TW/2   # centres
    BTy = 1.8   # bottom tank base y
    TTy = 7.2   # top tank base y

    # Pump centres
    Pu1x = 0.35; Pu1y = BTy + BTH/2 + 0.05
    Pu2x = 9.65; Pu2y = BTy + BTH/2 + 0.05
    R_P = 0.33;  R_V = 0.27

    # Valve centres
    V1x = Pu1x; V1y = Pu1y + BTy + 0.05
    V2x = Pu2x; V2y = Pu2y + BTy + 0.05

    # Top cross-pipe y (above both top tanks)
    TopY = TTy + TTH + 0.6

    # Bottom sump y
    SumpY = BTy - 0.45

    # ── helpers ───────────────────────────────────────────────────────────────
    def _pipe(*pts, lw=LW):
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=C_PIPE, lw=lw,
                solid_capstyle='round', solid_joinstyle='round', zorder=4)

    def _arr(x0, y0, x1, y1, lw=LW):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=C_PIPE,
                                    lw=lw, mutation_scale=12), zorder=5)

    def _pump(cx, cy):
        ax.add_patch(Circle((cx, cy), R_P,
                     facecolor=C_PUMP, edgecolor='#555', lw=1.4, zorder=6))
        ax.add_patch(Arc((cx, cy), R_P*1.05, R_P*1.05,
                         angle=0, theta1=40, theta2=310,
                         color='#555', lw=1.2, zorder=7))
        ang = np.radians(40)
        ax.annotate('', xy=(cx + R_P*0.52*np.cos(ang),
                             cy + R_P*0.52*np.sin(ang)),
                    xytext=(cx + R_P*0.52*np.cos(ang+0.4),
                             cy + R_P*0.52*np.sin(ang+0.4)),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.1), zorder=7)

    def _valve(cx, cy, orientation):
        s = R_V * 0.9
        ax.add_patch(plt.Polygon(
            [[cx,cy],[cx+s*0.85,cy-s],[cx-s*0.85,cy-s]],
            facecolor=C_VALVE, edgecolor='#111', zorder=6))
        if orientation == 'right':
            ax.add_patch(plt.Polygon(
                [[cx,cy], [cx+s,cy+s*0.85], [cx+s,cy-s*0.85]],
                facecolor=C_VALVE, edgecolor='#111', zorder=6))
        elif orientation == 'left':
            ax.add_patch(plt.Polygon(
                [[cx,cy], [cx-s,cy-s*0.85], [cx-s,cy+s*0.85]],
                facecolor=C_VALVE, edgecolor='#111', zorder=6))
        ax.add_patch(plt.Polygon(
            [[cx,cy],[cx-s*0.85,cy+s],[cx+s*0.85,cy+s]],
            facecolor=C_VALVE, edgecolor='#111', zorder=6))

        ax.plot(cx, cy, 'o', color='#111', ms=3.5, zorder=7)

    # ── Common style for animated numeric labels ───────────────────────────────────────────────
    _nkw = dict(ha='center', va='center', fontsize=9,
                fontweight='bold', zorder=8,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.5))

    # ── tanks: static outline + animated water fill ───────────────────────────
    # Store (water_patch, base_y, plot_height, h_max_physical)
    tank_info = []
    meas_lbls = [] # store text labels for measured heights
    for (lx, by, pw, ph, hmax_phys, idx) in [
            (T1lx, BTy, TW, BTH, h_max[0], 1),
            (T2lx, BTy, TW, BTH, h_max[1], 2),
            (T1lx, TTy, TW, TTH, h_max[2], 3),
            (T2lx, TTy, TW, TTH, h_max[3], 4)]:
        h0_phys = float(heights[idx-1][0])
        h0_plot = np.clip(h0_phys / hmax_phys, 0, 1) * ph
        wp = _mp.Rectangle((lx+0.05, by+0.02), pw-0.1, h0_plot,
                            facecolor=C_WATER, edgecolor='none', alpha=0.88, zorder=2)
        ax.add_patch(wp)
        ax.add_patch(_mp.Rectangle((lx, by), pw, ph,
                     facecolor='none', edgecolor=C_TANK, lw=LW, zorder=3))
        # h label on right side
        ax.text(lx + pw + 0.15, by + ph/2, f'h{idx}',
                ha='left', va='center', fontsize=10, color='#333', zorder=5)
        # measured height label (animated)
        meas_lbls.append(ax.text(lx + pw/2, by + ph/2 - 0.3, f'{h0_phys:.1f} cm',
                           color=C_PIPE, **_nkw))
        tank_info.append((wp, by, ph, hmax_phys, idx-1))

    # ── pumps & valves ────────────────────────────────────────────────────────
    _pump(Pu1x, Pu1y); _pump(Pu2x, Pu2y)
    _valve(V1x, V1y, orientation='right');  _valve(V2x, V2y, orientation='left')

    # ── static pipes ─────────────────────────────────────────────────────────
    # Pump input arrows from outside
    _arr(-0.3, Pu1y, Pu1x-R_P-0.02, Pu1y)
    _arr(10.3, Pu2y, Pu2x+R_P+0.02, Pu2y)
    ax.text(-0.35, Pu1y, 'u1', ha='right', va='center',
            fontsize=9, color=C_NUM, fontweight='bold', zorder=8)
    ax.text(10.35, Pu2y, 'u2', ha='left', va='center',
            fontsize=9, color=C_NUM, fontweight='bold', zorder=8)
    # gamma1/gamma2 input arrows from outside
    _arr(-0.3, V1y, V1x-R_V-0.08, V1y)
    _arr(10.3, V2y, V2x+R_V+0.08, V2y)
    ax.text(-0.6, V1y-0.15, 'γ1', ha='center', va='bottom',
            fontsize=9, color=C_NUM, fontweight='bold', zorder=8)
    ax.text(10.6, V2y-0.15, 'γ2', ha='center', va='bottom',
            fontsize=9, color=C_NUM, fontweight='bold', zorder=8)

    # pump1 → valve1
    _pipe((Pu1x, Pu1y), (V1x, V1y))
    # pump2 → valve2
    _pipe((Pu2x, Pu2y), (V2x, V2y))

    # ── LEFT pump piping ──────────────────────────────────────────────────────
    # (1-γ1)·q1: valve1 right → tank1 left wall   [horizontal, with arrow]
    _arr(V1x, V1y, T1lx+TW/4, V1y)

    # γ1·q1 CROSS: valve1 right → up → across top → down into tank4 (upper RIGHT)
    offset = 0.05 # small offset to avoid overlapping with the pipe from valve2 to tank3
    _pipe((V1x, V1y), (V1x, TopY*(1+offset)),   # go up left side
          (T2cx, TopY*(1+offset)))                        # cross to right column
    _arr(T2cx, TopY*(1+offset), T2cx, TTy+TTH+0.05)      # down into tank4

    # ── RIGHT pump piping ─────────────────────────────────────────────────────
    # (1-γ2)·q2: valve2 left → tank2 right wall   [horizontal, with arrow]
    _arr(V2x, V2y, T2cx+TW/4, V2y)

    # γ2·q2 CROSS: valve2 left → up → across top → down into tank3 (upper LEFT)
    _pipe((V2x, V2y), (V2x, TopY),    # go up right side
          (T1cx, TopY))                          # cross to left column
    _arr(T1cx, TopY, T1cx, TTy+TTH+0.05)       # down into tank3

    # ── Tank drain pipes ──────────────────────────────────────────────────────
    # tank3 left wall drains down into tank1 top
    _arr(T1lx+0.5*TW, TTy, T1lx+0.5*TW, BTy+BTH+0.05)
    # tank4 right wall drains down into tank2 top
    _arr(T2lx+0.5*TW, TTy, T2lx+0.5*TW, BTy+BTH+0.05)

    # ── Bottom sump ───────────────────────────────────────────────────────────
    _pipe((Pu1x, Pu1y-R_P), (Pu1x, SumpY),
          (Pu2x, SumpY), (Pu2x, Pu2y-R_P))
    _pipe((T1cx, BTy), (T1cx, SumpY))
    _pipe((T2cx, BTy), (T2cx, SumpY))

    # ── Animated numeric labels ───────────────────────────────────────────────
    _nkw = dict(ha='center', va='center', fontsize=9,
                fontweight='bold', zorder=8,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.5))

    # q1: under u1 label
    lbl_q1    = ax.text(-0.55, Pu1y - 0.4, '', color=C_NUM, **_nkw)
    lbl_gamma1= ax.text(-0.55, V1y - 0.4, '', color=C_NUM, **_nkw)
    # (1-γ1)·q1: midpoint of horizontal pipe to tank1, above pipe
    lbl_1g1q1 = ax.text((V1x+R_V + T1lx)/2, V1y + 0.34, '', color=C_PIPE, **_nkw)
    # γ1·q1: on the vertical segment going up, to the left of the pipe
    lbl_g1q1  = ax.text(V1x-R_V - 0.42, (V1y + TopY*0.6)/2 + V1y*0.2, '', color=C_PIPE, **_nkw)

    # q2: under u2 label
    lbl_q2    = ax.text(10.55, Pu2y - 0.4, '', color=C_NUM, **_nkw)
    lbl_gamma2= ax.text(10.55, V2y - 0.4, '', color=C_NUM, **_nkw)
    # (1-γ2)·q2: midpoint of horizontal pipe to tank2, above pipe
    lbl_1g2q2 = ax.text((V2x-R_V + T2lx+TW)/2, V2y + 0.34, '', color=C_PIPE, **_nkw)
    # γ2·q2: on the vertical segment going up, to the right of the pipe
    lbl_g2q2  = ax.text(V2x+R_V + 0.42, (V2y + TopY*0.6)/2 + V2y*0.2, '', color=C_PIPE, **_nkw)

    all_labels = [lbl_q1, lbl_1g1q1, lbl_g1q1, lbl_q2, lbl_1g2q2, lbl_g2q2, lbl_gamma1, lbl_gamma2] + meas_lbls

    # ── update ────────────────────────────────────────────────────────────────
    def update(frame):
        # water fills — scale physical cm to plot units
        for (wp, base_y, plot_h, hmax_phys, hi) in tank_info:
            h_phys = float(heights[hi][frame])
            h_plot = np.clip(h_phys / hmax_phys, 0.0, 1.0) * plot_h
            wp.set_height(h_plot)

        # numeric labels
        u1v = float(pumps[0][frame])
        u2v = float(pumps[1][frame])
        g1v = float(gammas[0][frame]); g2v = float(gammas[1][frame])
        lbl_q1.set_text(f'{u1v:.1f}%')
        lbl_1g1q1.set_text(f'{(1-g1v)*u1v:.1f}%')
        lbl_g1q1.set_text(f'{g1v*u1v:.1f}%')
        lbl_q2.set_text(f'{u2v:.1f}%')
        lbl_1g2q2.set_text(f'{(1-g2v)*u2v:.1f}%')
        lbl_g2q2.set_text(f'{g2v*u2v:.1f}%')
        lbl_gamma1.set_text(f'{g1v:.2f}')
        lbl_gamma2.set_text(f'{g2v:.2f}')
        for i in range(4):
            meas_lbls[i].set_text(f'{float(heights[i][frame]):.1f} cm')

        return tuple(wp for wp,*_ in tank_info) + tuple(all_labels)

    return _initialized(update)


def BatchBioreactor(axis, *, X, S, P, V, u=None, S_in=None,
                    V_max=None, X_max=None, S_max=None, P_max=None):
    """Draw bioreactor volume, concentrations and optional feed annotations.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Two-dimensional axes on which to draw the scene.
    X : array_like of shape (N,)
        Biomass-concentration trajectory, displayed in mol/L.
    S : array_like of shape (N,)
        Substrate-concentration trajectory, displayed in mol/L.
    P : array_like of shape (N,)
        Product-concentration trajectory, displayed in mol/L.
    V : array_like of shape (N,)
        Liquid-volume trajectory in m^3, used for vessel fill height.
    u : array_like of shape (N,) or None, default=None
        Optional feed-flow annotations in m^3/min. None omits the numeric feed-
        flow text.
    S_in : array_like of shape (N,) or None, default=None
        Optional feed-substrate concentrations in mol/L. None omits the numeric
        concentration text.
    V_max : float or None, default=None
        Positive volume display scale in m^3. None uses the maximum supplied V,
        with a small positive floor.
    X_max : float or None, default=None
        Positive biomass display scale in mol/L. None uses the maximum X with a
        small positive floor.
    S_max : float or None, default=None
        Positive substrate display scale in mol/L. None uses the maximum S with a
        small positive floor.
    P_max : float or None, default=None
        Positive product display scale in mol/L. None uses the maximum P with a
        small positive floor.

    Returns
    -------
    update : callable
        Initialized callback ``update(frame)``. The zero-based integer frame
        selects a sample and the callback returns the modified Matplotlib
        artists. Frame zero is drawn before this factory returns.

    Notes
    -----
    All supplied trajectories must be finite, nonempty and have the same
    sample count ``N``. Repeat constant values over the sample grid.

    Concentration bubbles, substrate sediment and the mixer are visual
    indicators, not additional model states. Display scales affect drawing
    only; the helper does not reconcile inconsistent process-unit scales.
    """
    import matplotlib.patches as _mp
    from matplotlib.patches import FancyBboxPatch, Circle, Arc, Ellipse

    values = {"X": X, "S": S, "P": P, "V": V}
    if u is not None:
        values["u"] = u
    if S_in is not None:
        values["S_in"] = S_in
    trajectories = _matching_trajectories(**values)
    X, S, P, V = (trajectories[name] for name in ("X", "S", "P", "V"))
    u, S_in = trajectories.get("u"), trajectories.get("S_in")
    ax = axis
    ax.set_xlim(0, 7.2); ax.set_ylim(0, 9.0)
    ax.set_aspect('equal'); ax.axis('off')

    _Vmax = float(V_max  if V_max  is not None else max(V.max(),  1e-9))
    _Xmax = float(X_max  if X_max  is not None else max(X.max(),  1e-9))
    _Smax = float(S_max  if S_max  is not None else max(S.max(),  1e-9))
    _Pmax = float(P_max  if P_max  is not None else max(P.max(),  1e-9))

    # ── layout ───────────────────────────────────────────────────────────────
    LX=0.5; BY=0.8; VW=4.4; VH=6.2; RAD=0.32
    CX = LX + VW/2
    MOTOR_TOP = BY+VH+0.05; MOTOR_H=0.36; MOTOR_W=0.38
    BLADE_R=0.85; BLADE_Y_FRAC=0.38
    BAR_X = LX+VW+0.35; BAR_W=0.24; BAR_H=0.70; BAR_GAP=0.90

    # ── static vessel outline ─────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((LX,BY), VW, VH,
        boxstyle=f'round,pad=0,rounding_size={RAD}',
        facecolor='none', edgecolor='#334455', lw=2.0, zorder=3))

    # ── motor housing ─────────────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch(
        (CX-MOTOR_W/2, MOTOR_TOP), MOTOR_W, MOTOR_H,
        boxstyle='round,pad=0.04', facecolor='#dddddd', edgecolor='#555',
        lw=1.0, zorder=6))

    # ── animated water fill ───────────────────────────────────────────────────
    def _fill_h(frame):
        return VH * np.clip(float(V[frame])/_Vmax, 0, 1)

    fh0 = _fill_h(0)
    water = FancyBboxPatch((LX+0.06, BY+0.06), VW-0.12, max(fh0-0.06, 0.02),
        boxstyle=f'round,pad=0,rounding_size={RAD-0.08}',
        facecolor='#c8e4f5', edgecolor='none', alpha=0.80, zorder=2)
    ax.add_patch(water)

    # ── animated substrate sediment ───────────────────────────────────────────
    sed = FancyBboxPatch((LX+0.18, BY+0.06), VW-0.36, 0.30,
        boxstyle='round,pad=0.02',
        facecolor='#c8a84b', edgecolor='none', alpha=0.70, zorder=3)
    ax.add_patch(sed)

    # ── pre-placed bubble positions (fixed random, scaled each frame) ─────────
    N_BUB = 22
    rng1 = np.random.default_rng(42)
    bxX = rng1.uniform(LX+0.25, CX-0.10, N_BUB)
    byX = rng1.uniform(0.0, 1.0, N_BUB)   # relative within fill
    rng2 = np.random.default_rng(7)
    bxP = rng2.uniform(CX+0.05, LX+VW-0.25, N_BUB)
    byP = rng2.uniform(0.0, 1.0, N_BUB)

    bub_X, bub_P = [], []
    for i in range(N_BUB):
        c = Circle((bxX[i], BY+0.5), 0.001,
                   facecolor='#2a8c44', edgecolor='none', alpha=0.58, zorder=4)
        ax.add_patch(c); bub_X.append(c)
        c = Circle((bxP[i], BY+0.5), 0.001,
                   facecolor='#8844cc', edgecolor='none', alpha=0.58, zorder=4)
        ax.add_patch(c); bub_P.append(c)
    biomass_bubbles = BubblePopulation(bub_X, bxX, byX)
    product_bubbles = BubblePopulation(bub_P, bxP, byP)

    # ── static labels ─────────────────────────────────────────────────────────
    ax.text(CX-VW/4, BY+0.52, 'X', fontsize=10, color='#2a8c44', fontweight='bold')
    ax.text(CX+VW/4, BY+0.52, 'P', fontsize=10, color='#8844cc', fontweight='bold')
    ax.text(CX,      BY+0.22, 'S', fontsize=9,  color='#7a5c10', fontweight='bold', ha='center')

    # ── inlet flow ────────────────────────────────────────────────────────────
    ax.annotate('', xy=(CX-0.70, MOTOR_TOP+MOTOR_H*0.5),
                xytext=(CX-2.0, MOTOR_TOP+MOTOR_H*0.5),
                arrowprops=dict(arrowstyle='-', color='#cc2200', lw=1.5), zorder=5)
    ax.annotate('', xy=(CX-0.75, MOTOR_TOP+MOTOR_H*0.5+0.05),
                xytext=(CX-0.75, MOTOR_TOP-MOTOR_H),
                arrowprops=dict(arrowstyle='<-', color='#cc2200', lw=1.5), zorder=5)
    if u is not None or S_in is not None:
        if u is not None:
            u_txt = ax.text(CX-0.70, MOTOR_TOP+MOTOR_H*1.3, '',
                        ha='right', fontsize=8, color='#cc2200')
        if S_in is not None:
            S_in_txt = ax.text(CX-0.70, MOTOR_TOP+MOTOR_H*0.75, '',
                        ha='right', fontsize=8, color='#cc2200')
    else:
        ax.text(CX-1.25, MOTOR_TOP+MOTOR_H*0.8, 'u', ha='right', fontsize=10, color='#cc2200', fontweight='bold')
    # ── side bars ─────────────────────────────────────────────────────────────
    bar_defs = [('X', _Xmax, '#2a8c44', 'mol/L'), ('S', _Smax, '#c8a84b', 'mol/L'),
                ('P', _Pmax, '#8844cc', 'mol/L'), ('V', _Vmax, '#2255aa', 'm³')]
    n_bars = len(bar_defs)
    total_h = n_bars*(BAR_H+BAR_GAP)-BAR_GAP
    y_top = BY + VH*0.975

    bars = []
    for i, (lbl, vmax, col, units) in enumerate(bar_defs):
        yi = y_top - i*(BAR_H+BAR_GAP)
        ax.add_patch(FancyBboxPatch((BAR_X, yi-BAR_H), BAR_W, BAR_H,
            boxstyle='round,pad=0.02',
            facecolor='#eeeeee', edgecolor='#bbbbbb', lw=0.8, zorder=5))
        bar = FancyBboxPatch((BAR_X+0.01, yi-BAR_H+0.01), BAR_W-0.02, 0.02,
            boxstyle='round,pad=0.01',
            facecolor=col, edgecolor='none', alpha=0.88, zorder=6)
        ax.add_patch(bar)
        ax.text(BAR_X+BAR_W/2, yi+0.07, lbl,
                ha='center', va='bottom', fontsize=8, color=col,
                fontweight='bold', zorder=7)
        vtxt = ax.text(BAR_X+BAR_W/2, yi-BAR_H-0.08, '',
                       ha='center', va='top', fontsize=7.5, color='#333', zorder=7)
        ax.text(BAR_X+BAR_W/2, yi-BAR_H-BAR_GAP/3, units,
                ha='center', va='top', fontsize=7, color='#333', zorder=7)
        bars.append(VerticalBar(bar, vtxt, BAR_H, vmax))

    # ── static mixer shaft ────────────────────────────────────────────────────
    shaft, = ax.plot([CX, CX], [MOTOR_TOP, BY+VH*0.25],
                     color='#999', lw=2.0, zorder=6)
    blade_l, = ax.plot([], [], color='#555', lw=2.5,
                       solid_capstyle='round', zorder=7)
    blade_r, = ax.plot([], [], color='#555', lw=2.5,
                       solid_capstyle='round', zorder=7)
    hub = Circle((CX, BY+VH*0.25), 0.07, color='#555', zorder=8)
    ax.add_patch(hub)
    depth_ell = Ellipse((CX, BY+VH*0.25-0.07), BLADE_R*1.7, 0.13,
        facecolor='none', edgecolor='#aaa', lw=0.8, alpha=0.5, zorder=5)
    ax.add_patch(depth_ell)
    mixer = Mixer(shaft, (blade_l, blade_r), hub, depth_ell, CX, MOTOR_TOP, BLADE_R)

    # ── update ────────────────────────────────────────────────────────────────
    state_arrs = [X, S, P, V]

    def update(frame):
        fh = _fill_h(frame)
        water.set_height(max(fh - 0.06, 0.02))

        # inlet flow
        if u is not None:
            u_txt.set_text(f'{u[frame]:.2f} m³/min')
        if S_in is not None:
            S_in_txt.set_text(f'S={S_in[frame]:.0f} mol/L')

        # sediment scales with S
        s_frac = np.clip(float(S[frame])/_Smax, 0.04, 1.0)
        sed.set_height(0.30 * s_frac)

        # bubbles X
        fx = np.clip(float(X[frame])/_Xmax, 0, 1)
        biomass_bubbles.update(fx, y_bottom=BY + 0.50, usable_height=fh - 0.60)

        # bubbles P
        fp = np.clip(float(P[frame])/_Pmax, 0, 1)
        product_bubbles.update(fp, y_bottom=BY + 0.50, usable_height=fh - 0.60)

        # side bars
        for bar, arr in zip(bars, state_arrs):
            bar.update(float(arr[frame]))

        # mixer
        blade_y = BY + fh * BLADE_Y_FRAC
        mixer.update(frame, blade_y)

        inlet_labels = []
        if u is not None:
            inlet_labels.append(u_txt)
        if S_in is not None:
            inlet_labels.append(S_in_txt)
        return ([water, sed, *mixer.artists]
                + list(biomass_bubbles.artists) + list(product_bubbles.artists)
                + [artist for bar in bars for artist in bar.artists] + inlet_labels)

    return _initialized(update)


def CSTR(axis, *, Ca, Cb, Tr, Tk, dQ=None,
         F=None, T_in=None, Ca_in=None,
         Ca_max=None, Cb_max=None,
         Tr_min=100.0, Tr_max=220.0,
         Tk_min=100.0, Tk_max=220.0,
         dQ_max_abs=8500.0):
    """Draw a stirred-tank reactor with concentration and temperature indicators.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Two-dimensional axes on which to draw the scene.
    Ca : array_like of shape (N,)
        Reactant-concentration trajectory, displayed in mol/L.
    Cb : array_like of shape (N,)
        Product-concentration trajectory, displayed in mol/L.
    Tr : array_like of shape (N,)
        Reactor-temperature trajectory in degrees Celsius.
    Tk : array_like of shape (N,)
        Jacket-temperature trajectory in degrees Celsius.
    dQ : array_like of shape (N,) or None, default=None
        Optional signed jacket heat-flow values in kJ/h. Negative values cool
        the jacket. The annotation converts each value to kW. None uses zero
        heat flow for the drawing.
    F : array_like of shape (N,) or None, default=None
        Optional volumetric inlet/outlet flow in L/h. Multiply the model's
        dilution rate by the reactor volume in L to obtain this trajectory.
        None omits numeric flow text.
    T_in : array_like of shape (N,) or None, default=None
        Optional inlet-temperature annotations in degrees Celsius. None omits
        numeric inlet-temperature text.
    Ca_in : array_like of shape (N,) or None, default=None
        Optional inlet-concentration annotations in mol/L. None omits numeric
        inlet-concentration text.
    Ca_max : float or None, default=None
        Positive reactant display scale in mol/L. None uses the maximum Ca with a
        small positive floor.
    Cb_max : float or None, default=None
        Positive product display scale in mol/L. None uses the maximum Cb with a
        small positive floor.
    Tr_min : float, default=100.0
        Lower reactor-temperature gauge limit in degrees Celsius.
    Tr_max : float, default=220.0
        Upper reactor-temperature gauge limit in degrees Celsius; use a value
        greater than Tr_min.
    Tk_min : float, default=100.0
        Lower jacket-temperature gauge limit in degrees Celsius.
    Tk_max : float, default=220.0
        Upper jacket-temperature gauge limit in degrees Celsius; use a value
        greater than Tk_min.
    dQ_max_abs : float, default=8500.0
        Positive cooling-colour scale in kJ/h. Larger
        cooling magnitudes turn the jacket from red toward blue up to this scale.

    Returns
    -------
    update : callable
        Initialized callback ``update(frame)``. The zero-based integer frame
        selects a sample and the callback returns the modified Matplotlib
        artists. Frame zero is drawn before this factory returns.

    Notes
    -----
    All supplied trajectories must be finite, nonempty and have the same
    sample count ``N``. Repeat constant values over the sample grid.

    Bubbles and mixer motion are schematic.
    """
    import matplotlib.patches as _mp
    from matplotlib.patches import FancyBboxPatch, Circle, Arc, Ellipse

    values = {"Ca": Ca, "Cb": Cb, "Tr": Tr, "Tk": Tk}
    for name, value in (("dQ", dQ), ("F", F), ("T_in", T_in), ("Ca_in", Ca_in)):
        if value is not None:
            values[name] = value
    trajectories = _matching_trajectories(**values)
    Ca, Cb, Tr, Tk = (trajectories[name] for name in ("Ca", "Cb", "Tr", "Tk"))
    dQ, F = trajectories.get("dQ"), trajectories.get("F")
    T_in, Ca_in = trajectories.get("T_in"), trajectories.get("Ca_in")
    ax = axis
    ax.set_xlim(0, 7.8); ax.set_ylim(0, 9.0)
    ax.set_aspect('equal'); ax.axis('off')

    _Camax = float(Ca_max if Ca_max is not None else max(Ca.max(), 1e-9))
    _Cbmax = float(Cb_max if Cb_max is not None else max(Cb.max(), 1e-9))

    # ── layout ───────────────────────────────────────────────────────────────
    LX=0.5; BY=0.8; VW=4.8; VH=5.8; RAD=0.32
    CX=LX+VW/2
    MOTOR_TOP=BY+VH+0.05; MOTOR_H=0.36; MOTOR_W=0.38
    BLADE_R=0.85; BLADE_Y_FRAC=0.42
    JAC_INSET=0.28; N_ZZ=14; AMP=0.22
    FILL_FRAC=0.62; fill_h=VH*FILL_FRAC
    GAUGE_R=0.50
    BAR_X=LX+VW+0.30; BAR_W=0.24; BAR_H=0.62; BAR_GAP=0.90

    # ── static vessel ─────────────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((LX,BY), VW, VH,
        boxstyle=f'round,pad=0,rounding_size={RAD}',
        facecolor='none', edgecolor='#334455', lw=2.0, zorder=3))

    # ── static water fill (CSTR always full) ──────────────────────────────────
    ax.add_patch(FancyBboxPatch((LX+0.06, BY+0.06), VW-0.12, fill_h-0.06,
        boxstyle=f'round,pad=0,rounding_size={RAD-0.08}',
        facecolor='#c8e4f5', edgecolor='none', alpha=0.80, zorder=2))

    # ── motor housing ─────────────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch(
        (CX-MOTOR_W/2, MOTOR_TOP), MOTOR_W, MOTOR_H,
        boxstyle='round,pad=0.04', facecolor='#dddddd', edgecolor='#555',
        lw=1.0, zorder=6))

    # ── jacket zigzags (animated colour) ──────────────────────────────────────
    JAC_Y0 = BY + 0.22; JAC_Y1 = BY + fill_h - 0.18
    jac_ys = np.linspace(JAC_Y0, JAC_Y1, N_ZZ+2)
    JAC_XL = LX + JAC_INSET; JAC_XR = LX + VW - JAC_INSET
    jac_l, = ax.plot([], [], lw=2.2,
                     solid_capstyle='round', solid_joinstyle='round', zorder=5)
    jac_r, = ax.plot([], [], lw=2.2,
                     solid_capstyle='round', solid_joinstyle='round', zorder=5)
    dQ_lbl = ax.text(LX+0.05, BY+fill_h*1.05, '',
                     ha='left', va='center', fontsize=9,
                     fontweight='bold', color='#cc0000', zorder=8)

    # ── bubbles Ca and Cb ─────────────────────────────────────────────────────
    N_BUB = 20
    rng1 = np.random.default_rng(3)
    bxA = rng1.uniform(LX+0.40, CX-0.15, N_BUB)
    byA = rng1.uniform(0.0, 1.0, N_BUB)
    rng2 = np.random.default_rng(11)
    bxB = rng2.uniform(CX+0.10, LX+VW-0.40, N_BUB)
    byB = rng2.uniform(0.0, 1.0, N_BUB)
    bub_a, bub_b = [], []
    for i in range(N_BUB):
        c = Circle((bxA[i], BY+0.4), 0.001,
                   facecolor='#cc4444', edgecolor='none', alpha=0.58, zorder=4)
        ax.add_patch(c); bub_a.append(c)
        c = Circle((bxB[i], BY+0.4), 0.001,
                   facecolor='#2a8c44', edgecolor='none', alpha=0.58, zorder=4)
        ax.add_patch(c); bub_b.append(c)
    reactant_bubbles = BubblePopulation(bub_a, bxA, byA)
    product_bubbles = BubblePopulation(bub_b, bxB, byB)
    ax.text(CX-VW/4, BY+0.25, 'Ca', fontsize=9, color='#cc4444', fontweight='bold')
    ax.text(CX+VW/4, BY+0.25, 'Cb', fontsize=9, color='#2a8c44', fontweight='bold')

    # ── gauges Tr and Tk (inside vessel, top corners) ─────────────────────────
    GCxL=LX+0.72; GCxR=LX+VW-0.72; GCy=BY+VH-0.62

    def _static_gauge(cx, cy, r, label):
        ax.add_patch(Circle((cx,cy), r, facecolor='#f8f8f8',
                     edgecolor='#444', lw=1.5, zorder=9))
        ax.add_patch(Arc((cx,cy), r*1.7, r*1.7,
                     angle=0, theta1=-30, theta2=210,
                     color='#ddd', lw=2.5, zorder=10))
        ax.text(cx, cy-r*0.45, label, ha='center', va='center',
                fontsize=7.5, color='#555', zorder=12)

    _static_gauge(GCxL, GCy, GAUGE_R, 'Tr (°C)')
    _static_gauge(GCxR, GCy, GAUGE_R, 'Tk (°C)')

    gauge_data = {}
    for key, cx, vmin, vmax in [('Tr',GCxL,Tr_min,Tr_max),
                                  ('Tk',GCxR,Tk_min,Tk_max)]:
        arc = Arc((cx,GCy), GAUGE_R*1.7, GAUGE_R*1.7,
                  angle=0, theta1=210, theta2=210, color='gray', lw=2.5, zorder=11)
        ax.add_patch(arc)
        needle, = ax.plot([], [], color='#333', lw=1.2, zorder=12)
        ax.add_patch(Circle((cx,GCy), GAUGE_R*0.10, color='#333', zorder=13))
        lbl = ax.text(cx, GCy+GAUGE_R*0.45, '',
                      ha='center', va='center', fontsize=7.5,
                      fontweight='bold', color='#222', zorder=13)
        gauge_data[key] = DialGauge(arc, needle, lbl, (cx, GCy), GAUGE_R, (vmin, vmax))

    # ── static mixer shaft ────────────────────────────────────────────────────
    shaft, = ax.plot([CX,CX], [MOTOR_TOP, BY+fill_h*BLADE_Y_FRAC],
                     color='#999', lw=2.0, zorder=6)
    blade_l, = ax.plot([], [], color='#555', lw=2.5,
                       solid_capstyle='round', zorder=7)
    blade_r_ln, = ax.plot([], [], color='#555', lw=2.5,
                          solid_capstyle='round', zorder=7)
    hub = Circle((CX, BY+fill_h*BLADE_Y_FRAC), 0.07, color='#555', zorder=8)
    ax.add_patch(hub)
    depth_ell = Ellipse((CX, BY+fill_h*BLADE_Y_FRAC-0.07), BLADE_R*1.7, 0.13,
        facecolor='none', edgecolor='#aaa', lw=0.8, alpha=0.5, zorder=5)
    ax.add_patch(depth_ell)
    mixer = Mixer(shaft, (blade_l, blade_r_ln), hub, depth_ell, CX, MOTOR_TOP, BLADE_R)

    # ── static arrows ─────────────────────────────────────────────────────────
    #ax.annotate('', xy=(CX-0.65, MOTOR_TOP+MOTOR_H*0.5),
    #            xytext=(CX-2.10, MOTOR_TOP+MOTOR_H*0.5),
    #            arrowprops=dict(arrowstyle='->', color='#cc2200', lw=1.5))
    #ax.text(CX-2.15, MOTOR_TOP+MOTOR_H*0.62, 'F, Ca_in', ha='right',
    #        fontsize=8, color='#cc2200')
    #ax.text(CX-2.15, MOTOR_TOP+MOTOR_H*0.15, 'T_in', ha='right',
    #        fontsize=8, color='#cc2200')

    # ── inlet flow ────────────────────────────────────────────────────────────
    ax.annotate('', xy=(CX-0.7, MOTOR_TOP+MOTOR_H*0.5),
                xytext=(CX-1.5 if all([F is None, T_in is None, Ca_in is None]) else CX-2.5,
                        MOTOR_TOP+MOTOR_H*0.5),
                arrowprops=dict(arrowstyle='-', color='#cc2200', lw=1.5), zorder=5)
    ax.annotate('', xy=(CX-0.7, MOTOR_TOP+MOTOR_H*0.5),
                xytext=(CX-0.7, MOTOR_TOP-MOTOR_H),
                arrowprops=dict(arrowstyle='<-', color='#cc2200', lw=1.5), zorder=5)
    inlet_artists = []
    if F is not None or T_in is not None or Ca_in is not None:
        if F is not None:
            F_x = CX-1.35 if T_in is None else CX-1.85
            F_y = MOTOR_TOP+MOTOR_H*0.75 if Ca_in is None else MOTOR_TOP+MOTOR_H*1.4
            F_txt = ax.text(F_x, F_y, '', ha='right', fontsize=8, color='#cc2200')
            inlet_artists.append(F_txt)
        if T_in is not None:
            T_in_x = CX-1.35 if F is None else CX-0.75
            T_in_y = MOTOR_TOP+MOTOR_H*0.75 if Ca_in is None else MOTOR_TOP+MOTOR_H*1.4
            T_in_txt = ax.text(T_in_x, T_in_y, '', ha='right', fontsize=8, color='#cc2200')
            inlet_artists.append(T_in_txt)
        if Ca_in is not None:
            Ca_in_txt = ax.text(CX-0.90, MOTOR_TOP+MOTOR_H*0.75, '',
                        ha='right', fontsize=8, color='#cc2200')
            inlet_artists.append(Ca_in_txt)
    else:
        ax.text(CX-1.25, MOTOR_TOP+MOTOR_H*0.8, 'u', ha='right', fontsize=10, color='#cc2200', fontweight='bold')

    # ── outlet flow ────────────────────────────────────────────────────────────
    ax.annotate('', xy=(LX+VW+0.65, BY), xytext=(LX+VW-0.30, BY),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    out_txt = ax.text(LX+VW+0.70, BY, 'F', color='#333',
                      fontsize=8 if F is not None else 10,
                      fontweight='bold' if F is None else 'normal')
    if F is not None:
        inlet_artists.append(out_txt)

    # ── side bars ─────────────────────────────────────────────────────────────
    bar_defs = [
        ('Ca', Ca,  _Camax,      '#cc4444', 'mol/L'),
        ('Cb', Cb,  _Cbmax,      '#2a8c44', 'mol/L'),
        #('Tr', Tr,  Tr_max,      '#dd7700', '°C'),
        #('Tk', Tk,  Tk_max,      '#2255cc', '°C'),
        #('dQ', None, dQ_max_abs, '#884488', 'kW'),
    ]
    n_bars = len(bar_defs)
    y_top = BY + VH*0.50
    bars, bar_arrs = [], []
    for i, (lbl, arr, vmax, col, units) in enumerate(bar_defs):
        yi = y_top - i*(BAR_H+BAR_GAP) #if lbl != 'Tk' else y_top - i*(BAR_H+BAR_GAP*0.9) # If the bar is for Tk, we reduce the gap with the previous bar to make it more compact.
        ax.add_patch(FancyBboxPatch((BAR_X, yi-BAR_H), BAR_W, BAR_H,
            boxstyle='round,pad=0.02',
            facecolor='#eeeeee', edgecolor='#bbb', lw=0.8, zorder=5))
        bar = FancyBboxPatch((BAR_X+0.01, yi-BAR_H+0.01), BAR_W-0.02, 0.02,
            boxstyle='round,pad=0.01',
            facecolor=col, edgecolor='none', alpha=0.88, zorder=6)
        ax.add_patch(bar)
        ax.text(BAR_X+BAR_W/2, yi+0.07, lbl,
                ha='center', va='bottom', fontsize=7.5, color=col,
                fontweight='bold', zorder=7)
        vtxt = ax.text(BAR_X+BAR_W/2, yi-BAR_H-0.08, '',
                       ha='center', va='top', fontsize=7, color='#333', zorder=7)
        #if units != '°C': # if units is not °C, show units below the value
        ax.text(BAR_X+BAR_W/2, yi-BAR_H-BAR_GAP/3, units,
                ha='center', va='top', fontsize=7, color='#333', zorder=7)
        #else: # if units is °C, show units next to the value
        #    ax.text(BAR_X+BAR_W/2+0.30, yi-BAR_H-0.08, units,
        #            ha='left', va='top', fontsize=7, color='#333', zorder=7)
        bars.append(VerticalBar(bar, vtxt, BAR_H, vmax))
        bar_arrs.append(arr)

    # ── update ────────────────────────────────────────────────────────────────
    def update(frame):
        # inlet flow
        if F is not None:
            F_txt.set_text(f'{F[frame]:.2f} L/h')
            out_txt.set_text(f'{F[frame]:.2f} L/h')
        if T_in is not None:
            T_in_txt.set_text(f'{T_in[frame]:.1f} °C')
        if Ca_in is not None:
            Ca_in_txt.set_text(f'Ca={Ca_in[frame]:.1f} mol/L')

        # jacket colour
        dq = float(dQ[frame]) if dQ is not None else 0.0
        frac_dq = np.clip(-dq / dQ_max_abs, 0, 1)
        jcol = (1-frac_dq, 0.15*(1-frac_dq), frac_dq) # RGB
        xs_l = np.array([JAC_XL+(AMP if i%2==1 else -AMP)
                         for i in range(N_ZZ+2)], dtype=float)
        xs_l[[0,-1]] = JAC_XL
        jac_l.set_data(xs_l, jac_ys); jac_l.set_color(jcol)
        xs_r = np.array([JAC_XR-(AMP if i%2==1 else -AMP)
                         for i in range(N_ZZ+2)], dtype=float)
        xs_r[[0,-1]] = JAC_XR
        jac_r.set_data(xs_r, jac_ys); jac_r.set_color(jcol)
        dQ_lbl.set_color(jcol)
        dQ_lbl.set_text(f'dQ={dq/3600:.1f} kW')

        # bubbles Ca
        fca = np.clip(float(Ca[frame])/_Camax, 0, 1)
        reactant_bubbles.update(fca, y_bottom=BY + 0.35, usable_height=fill_h - 0.50)
        # bubbles Cb
        fcb = np.clip(float(Cb[frame])/_Cbmax, 0, 1)
        product_bubbles.update(fcb, y_bottom=BY + 0.35, usable_height=fill_h - 0.50)

        # gauges
        for key, val_arr in [('Tr',Tr),('Tk',Tk)]:
            val = float(val_arr[frame])
            gauge = gauge_data[key]
            low, high = gauge.limits
            frac = np.clip((val-low)/max(high-low, 1e-9), 0, 1)
            col = plt.cm.RdYlBu_r(frac)
            gauge.update(val, color=col)

        # side bars
        for bar, arr in zip(bars, bar_arrs):
            val = abs(float(arr[frame])) if arr is not None else abs(dq)
            text_value = float(arr[frame]) if arr is not None else dq
            bar.update(val, text=f'{text_value:.2f}')

        # mixer
        blade_y = BY + fill_h*BLADE_Y_FRAC
        mixer.update(frame, blade_y)

        return ([jac_l,jac_r,dQ_lbl,*mixer.artists]
                + [artist for gauge in gauge_data.values() for artist in gauge.artists]
                + list(reactant_bubbles.artists) + list(product_bubbles.artists)
                + [artist for bar in bars for artist in bar.artists]+inlet_artists)

    return _initialized(update)


def OilWell(axis, *, x0, x1, x2, P_tt, w_out, u1, u2,
            w_G_in=None, P_at=None, P_ab=None, P_bh=None,
            rho_m=None, alpha_L=None,
            w_G_in_max=None, w_out_max=None):
    """Draw a gas-lift well with flow, pressure and mass indicators.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Two-dimensional axes on which to draw the scene.
    x0 : array_like of shape (N,)
        Gas mass in the annulus in kg, corresponding to model state row 0.
    x1 : array_like of shape (N,)
        Gas mass in the tubing in kg, corresponding to state row 1.
    x2 : array_like of shape (N,)
        Liquid mass in the tubing in kg, corresponding to state row 2.
    P_tt : array_like of shape (N,)
        Tubing-top pressure in bar, corresponding to state row 7.
    w_out : array_like of shape (N,)
        Produced mixture mass flow in kg/s, corresponding to state row 8.
    u1 : array_like of shape (N,)
        Production-choke opening fraction, normally from 0 to 1; this is the
        first control input.
    u2 : array_like of shape (N,)
        Gas-lift-choke opening fraction, normally from 0 to 1; this is the second
        control input.
    w_G_in : array_like of shape (N,) or None, default=None
        Optional annulus gas-inflow trajectory in kg/s, state row 3. None omits
        its numeric text and uses a fixed bubble-intensity fraction of 0.5.
    P_at : array_like of shape (N,) or None, default=None
        Optional annulus-top pressure in bar, state row 4. None omits this gauge.
    P_ab : array_like of shape (N,) or None, default=None
        Optional annulus-bottom pressure in bar, state row 5. None omits this
        gauge.
    P_bh : array_like of shape (N,) or None, default=None
        Optional bottom-hole pressure in bar, state row 6. None omits this gauge.
    rho_m : array_like of shape (N,) or None, default=None
        Optional tubing-top mixture density in kg/m^3, model state rho_m_tt at
        row 9. None omits the density bar.
    alpha_L : array_like of shape (N,) or None, default=None
        Optional dimensionless tubing-top liquid volume fraction, model state
        alpha_L_tt at row 10. None omits its bar.
    w_G_in_max : float or None, default=None
        Positive gas-inflow scale in kg/s for annulus bubble intensity. None uses
        the maximum w_G_in when provided, otherwise 1.0. Supply a positive scale
        for an all-zero inflow trajectory.
    w_out_max : float or None, default=None
        Optional production-flow scale in kg/s. Accepted by the interface, but
        the current drawing does not use this value to change any artist.

    Returns
    -------
    update : callable
        Initialized callback ``update(frame)``. The zero-based integer frame
        selects a sample and the callback returns the modified Matplotlib
        artists. Frame zero is drawn before this factory returns.

    Notes
    -----
    All supplied trajectories must be finite, nonempty and have the same
    sample count ``N``. Repeat constant values over the sample grid.

    Pressure and flow data are displayed in the stated units without model
    recomputation. Annulus bubbles move downward, tubing bubbles upward;
    neither represents tracked particles. Tubing bubble intensity follows
    ``x1``. The drawing does not solve algebraic pressure constraints.
    """
    import matplotlib.patches as _mp
    from matplotlib.patches import FancyBboxPatch, Circle

    values = {"x0": x0, "x1": x1, "x2": x2, "P_tt": P_tt,
              "w_out": w_out, "u1": u1, "u2": u2}
    optional = {"w_G_in": w_G_in, "P_at": P_at, "P_ab": P_ab,
                "P_bh": P_bh, "rho_m": rho_m, "alpha_L": alpha_L}
    values.update({name: value for name, value in optional.items() if value is not None})
    trajectories = _matching_trajectories(**values)
    x0, x1, x2, P_tt, w_out, u1, u2 = (
        trajectories[name] for name in ("x0", "x1", "x2", "P_tt", "w_out", "u1", "u2")
    )
    w_G_in, P_at, P_ab, P_bh, rho_m, alpha_L = (
        trajectories.get(name)
        for name in ("w_G_in", "P_at", "P_ab", "P_bh", "rho_m", "alpha_L")
    )
    ax = axis
    ax.set_xlim(0, 10); ax.set_ylim(0, 16)
    ax.set_aspect('equal'); ax.axis('off')

    # ── Geometry ──────────────────────────────────────────────────────────────
    ANN_LX = 2.5; ANN_RX = 5.5
    TUB_LX = 3.2; TUB_RX = 4.8
    Y_TOP  = 13.2   # ground surface
    Y_INJ  = 4.2    # injection valve depth
    Y_BOT  = 1.4    # reservoir
    ANN_CX = (ANN_LX + ANN_RX) / 2
    TUB_W  = TUB_RX - TUB_LX
    ANN_W  = ANN_RX - ANN_LX

    _wGmax  = float(w_G_in_max  if w_G_in_max  is not None
                    else (w_G_in.max()  if w_G_in  is not None else 1.0))
    _wOmax  = float(w_out_max   if w_out_max   is not None
                    else max(w_out.max(), 1e-9))

    # ── Static background fills ───────────────────────────────────────────────
    # Annulus: gas (pale yellow) — two strips flanking the tubing
    for lx, w in [(ANN_LX+0.06, TUB_LX-ANN_LX-0.12),
                   (TUB_RX+0.06, ANN_RX-TUB_RX-0.12)]:
        ax.add_patch(_mp.Rectangle(
            (lx, Y_INJ), w, Y_TOP-Y_INJ-0.08,
            facecolor='#f0e8b8', edgecolor='none', alpha=0.90, zorder=1))

    # Tubing: crude oil (very dark reddish-brown), full height
    ax.add_patch(_mp.Rectangle(
        (TUB_LX+0.06, Y_BOT+0.05), TUB_W-0.12, Y_TOP-Y_BOT-0.15,
        facecolor='#1a0d05', edgecolor='none', alpha=0.92, zorder=1))

    # ── Static walls ──────────────────────────────────────────────────────────
    WKW = dict(color='#3a3a3a', lw=2.5, solid_capstyle='round', zorder=3)
    # Annulus outer walls
    ax.plot([ANN_LX,ANN_LX],[Y_INJ,Y_TOP], **WKW)
    ax.plot([ANN_RX,ANN_RX],[Y_INJ,Y_TOP], **WKW)
    # Annulus bottom caps (sides that connect to tubing)
    ax.plot([ANN_LX,TUB_LX],[Y_INJ,Y_INJ], **WKW)
    ax.plot([TUB_RX,ANN_RX],[Y_INJ,Y_INJ], **WKW)
    # Tubing walls (full depth)
    ax.plot([TUB_LX,TUB_LX],[Y_BOT,Y_TOP], **WKW)
    ax.plot([TUB_RX,TUB_RX],[Y_BOT,Y_TOP], **WKW)
    # Tubing bottom (open to reservoir — dashed)
    ax.plot([TUB_LX,TUB_RX],[Y_BOT,Y_BOT],
            color='#3a3a3a', lw=1.8, ls='--', zorder=3)

    # ── Wellhead cap ──────────────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch(
        (ANN_LX, Y_TOP), ANN_W, 0.45,
        boxstyle='round,pad=0.04', facecolor='#666', edgecolor='#333',
        lw=2, zorder=5))
    ax.add_patch(FancyBboxPatch(
        (TUB_LX-0.14, Y_TOP+0.45), TUB_W+0.28, 0.30,
        boxstyle='round,pad=0.03', facecolor='#777', edgecolor='#333',
        lw=1.5, zorder=6))

    # ── Injection valve ───────────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch(
        (TUB_LX-0.35, Y_INJ-0.20), 0.32, 0.40,
        boxstyle='round,pad=0.03', facecolor='#888', edgecolor='#333',
        lw=1.5, zorder=5))
    ax.annotate('',xy=(TUB_LX+0.02,Y_INJ),xytext=(TUB_LX-0.04,Y_INJ),
                arrowprops=dict(arrowstyle='->',color='#555',lw=1.2,
                                mutation_scale=10), zorder=6)
    ax.text(ANN_LX-0.12,Y_INJ-0.1,'Inj.\nvalve',
            ha='right',va='center',fontsize=7,color='#444')

    # ── Production choke (top, horizontal pipe) ───────────────────────────────
    PROD_Y = Y_TOP + 1.55
    ax.plot([ANN_CX,ANN_CX],[Y_TOP+0.75,PROD_Y], color='#555',lw=3,zorder=4)
    ax.plot([ANN_CX,ANN_CX+1.1],[PROD_Y,PROD_Y], color='#555',lw=3,zorder=4)
    s=0.24; cx=ANN_CX+1.38; cy=PROD_Y
    ax.add_patch(plt.Polygon([[cx,cy],[cx-s,cy+s*.85],[cx-s,cy-s*.85]],
                 facecolor='#666',edgecolor='#333',lw=1.2,zorder=6))
    ax.add_patch(plt.Polygon([[cx,cy],[cx+s,cy+s*.85],[cx+s,cy-s*.85]],
                 facecolor='#666',edgecolor='#333',lw=1.2,zorder=6))
    ax.plot([cx+s+0.05,cx+s+1.1],[PROD_Y,PROD_Y], color='#555',lw=3,zorder=4)
    ax.annotate('',xy=(cx+s+1.45,PROD_Y),xytext=(cx+s+1.0,PROD_Y),
                arrowprops=dict(arrowstyle='->',color='#333',lw=2,mutation_scale=13),
                zorder=5)
    ax.text(cx+s+1.55,PROD_Y+0.32,'w_out',fontsize=8.5,va='center',color='#333')
    # u2 label above choke
    ax.text(cx,PROD_Y+0.40,'u1',ha='center',fontsize=8,color='#0044cc',
            fontweight='bold')
    u1_txt = ax.text(cx, PROD_Y-0.55, '', ha='center', fontsize=8, color='#0044cc', fontweight='bold')

    # ── Gas lift choke (left side) ────────────────────────────────────────────
    GAS_Y = Y_TOP - 1.4; GX = ANN_LX
    ax.plot([GX,GX],[GAS_Y,Y_TOP+0.05], color='#555',lw=3,zorder=4)
    ax.plot([GX-1.7,GX],[GAS_Y,GAS_Y],  color='#555',lw=3,zorder=4)
    cx2=GX-0.62; cy2=GAS_Y
    ax.add_patch(plt.Polygon([[cx2,cy2],[cx2-s,cy2+s*.85],[cx2-s,cy2-s*.85]],
                 facecolor='#666',edgecolor='#333',lw=1.2,zorder=6))
    ax.add_patch(plt.Polygon([[cx2,cy2],[cx2+s,cy2+s*.85],[cx2+s,cy2-s*.85]],
                 facecolor='#666',edgecolor='#333',lw=1.2,zorder=6))
    ax.plot([cx2-s-0.05,GX-1.45],[GAS_Y,GAS_Y], color='#555',lw=3,zorder=4)
    ax.annotate('',xy=(GX-1.98,GAS_Y),xytext=(GX-1.5,GAS_Y),
                arrowprops=dict(arrowstyle='->',color='#335',lw=2,mutation_scale=13),
                zorder=5)
    ax.text(GX-2.08,GAS_Y+0.16,'Gas in',fontsize=8,ha='right',va='bottom',color='#335')
    gas_in_txt = ax.text(GX-2.08, GAS_Y-0.32, '', fontsize=8, ha='right', va='bottom', color='#335')
    # u2 label above choke
    ax.text(cx2,GAS_Y+0.40,'u2',ha='center',fontsize=8,color='#0044cc',
            fontweight='bold')
    u2_txt = ax.text(cx2, GAS_Y-0.55, '', ha='center', fontsize=8, color='#0044cc', fontweight='bold')

    # ── Reservoir wavelines ───────────────────────────────────────────────────
    for k in range(3):
        y = Y_BOT - 0.38 - k*0.32
        xs = np.linspace(ANN_LX-0.3, ANN_RX+0.3, 24)
        ax.plot(xs, y+0.07*np.sin(np.linspace(0,4*np.pi,24)),
                color='#6b3a0d', lw=1.5, alpha=0.55, zorder=2)

    # ── Pressure gauges (needle-style, same as bioreactors) ──────────────────
    # Each gauge is positioned beside the well, connected by a thin tap line.
    # Layout:
    #   P_at  – left of annulus top,   connected horizontally to left wall
    #   P_ab  – left of annulus bottom, connected horizontally to left wall
    #   P_bh  – right of reservoir,        connected horizontally to right wall
    #   P_tt  – right of tubing top,    connected horizontally to right wall
    from matplotlib.patches import Arc as _Arc, Circle as _Circ

    GR = 0.55   # gauge radius (plot units)

    # (key, tap_x, tap_y, gauge_cx, gauge_cy, label_colour, vmin, vmax)
    GAUGE_GAP = 1  # gap between measurement point and gauge body
    _gauge_specs = [
        ('P_at', ANN_LX,       Y_TOP - 0.35,
                 ANN_LX - GR - GAUGE_GAP, Y_TOP - 0.35, '#cc5500'),
        ('P_ab', ANN_LX,       Y_INJ + 0.35,
                 ANN_LX - GR - GAUGE_GAP, Y_INJ + 0.35, '#cc5500'),
        ('P_bh', TUB_RX,       Y_BOT,
                 TUB_RX + GR + GAUGE_GAP, Y_BOT, '#cc5500'),
        ('P_tt', TUB_RX,       Y_TOP - 0.35,
                 TUB_RX + GR + GAUGE_GAP, Y_TOP - 0.35, '#0044cc'),
    ]
    pdata = {'P_at': P_at, 'P_ab': P_ab, 'P_bh': P_bh, 'P_tt': P_tt}

    # Check if data is available for each gauge and remove the gauge if not.
    _gauge_specs = [spec for spec in _gauge_specs if pdata.get(spec[0]) is not None]

    # Compute per-gauge vmin/vmax from data (or fall back to fixed ranges)
    _p_ranges = {}
    for key, *_ in _gauge_specs:
        arr = pdata.get(key)
        if arr is not None:
            _p_ranges[key] = (max(arr.min() * 0.95, 0), arr.max() * 1.05)
        else:
            _p_ranges[key] = (0.0, 200.0)   # sensible default (bar)

    # Draw static gauge bodies and tap lines
    gauges = {}

    for key, tx, ty, gcx, gcy, col in _gauge_specs:
        arr = pdata.get(key)

        # Tap line from well wall to gauge body
        ax.plot([tx, gcx + (0 if abs(ty - gcy) > 0.1 else
                             np.sign(tx - gcx) * GR)],
                [ty, gcy + (np.sign(ty - gcy) * GR
                             if abs(ty - gcy) > 0.1 else 0)],
                color='#777', lw=1.2, zorder=6)

        # Gauge body (circle)
        ax.add_patch(_Circ((gcx, gcy), GR,
                     facecolor='#f8f8f8', edgecolor='#444',
                     lw=1.5, zorder=9))
        # Grey track arc (−30° → 210°, CCW)
        ax.add_patch(_Arc((gcx, gcy), GR*1.7, GR*1.7,
                     angle=0, theta1=-30, theta2=210,
                     color='#ddd', lw=2.5, zorder=10))
        # Static label (key name) inside at bottom
        ax.text(gcx, gcy - GR*0.44, key,
                ha='center', va='center', fontsize=6.5,
                color='#555', style='italic', zorder=12)

        # Animated coloured arc (value track)
        val_arc = _Arc((gcx, gcy), GR*1.7, GR*1.7,
                       angle=0, theta1=210, theta2=210,
                       color=col, lw=2.5, zorder=11)
        ax.add_patch(val_arc)

        # Animated needle
        needle, = ax.plot([], [], color='#333', lw=1.2, zorder=12)

        # Centre hub
        hub = _Circ((gcx, gcy), GR*0.10, color='#333', zorder=13)
        ax.add_patch(hub)

        # Animated numeric text (above centre)
        lbl = ax.text(gcx, gcy + GR*0.22, '',
                      ha='center', va='center',
                      fontsize=6.5, fontweight='bold',
                      color='#222', zorder=13)
        gauges[key] = (DialGauge(
            val_arc, needle, lbl, (gcx, gcy), GR, _p_ranges[key]
        ), col)

    # ── Side labels ───────────────────────────────────────────────────────────
    ax.text(ANN_LX-0.22,(Y_INJ+Y_TOP)/2,'Annulus',
            ha='right',va='center',fontsize=8.5,color='#554400',
            rotation=90,fontweight='bold')
    ax.text(TUB_LX-0.12,(Y_BOT+Y_TOP)/2,'Tubing',
            ha='right',va='center',fontsize=8,color="#666666",
            rotation=90,fontweight='bold')

    # ── Side bar panel ────────────────────────────────────────────────────────
    BAR_X=6.55; BAR_W=0.24; BAR_H=0.70; BAR_GAP=0.85
    bar_defs = [
        ('x0',   x0,    max(x0.max(),1e-9),  '#cc9900', 'kg'),
        ('x1',   x1,    max(x1.max(),1e-9),  '#4488cc', 'kg'),
        ('x2',   x2,    max(x2.max(),1e-9),  '#6b3a0d', 'kg'),
    ]
    if rho_m is not None:
        bar_defs.append(('rho_m', rho_m, max(rho_m.max(),1e-9), '#884488', 'kg/m³'))
    if alpha_L is not None:
        bar_defs.append(('alpha_L', alpha_L, 1.0,               '#2255cc', ''))

    y_top_bar = len(bar_defs)*(BAR_H+BAR_GAP) + Y_INJ-0.40
    bars, bar_arrs, bar_units = [], [], []
    for i, (lbl, arr, vmax, col, unit) in enumerate(bar_defs):
        yi = y_top_bar - i*(BAR_H+BAR_GAP)
        if yi - BAR_H < Y_BOT: break   # don't go below reservoir
        ax.add_patch(FancyBboxPatch((BAR_X, yi-BAR_H), BAR_W, BAR_H,
            boxstyle='round,pad=0.02',
            facecolor='#eeeeee', edgecolor='#aaa', lw=0.8, zorder=7))
        bar = FancyBboxPatch((BAR_X+0.01, yi-BAR_H+0.01), BAR_W-0.02, 0.02,
            boxstyle='round,pad=0.01',
            facecolor=col, edgecolor='none', alpha=0.88, zorder=8)
        ax.add_patch(bar)
        ax.text(BAR_X+BAR_W/2, yi+0.07, lbl,
                ha='center', va='bottom', fontsize=7, color=col,
                fontweight='bold', zorder=9)
        vtxt = ax.text(BAR_X+BAR_W/2, yi-BAR_H-0.07, '',
                       ha='center', va='top', fontsize=6.5, color='#333', zorder=9)
        bars.append(VerticalBar(bar, vtxt, BAR_H, vmax))
        bar_arrs.append(arr); bar_units.append(unit)

    # ── Pre-compute bubble positions (fixed layout, phase-shifted per frame) ──
    # Annulus bubbles: descend from top to injection valve
    N_ANN = 20
    rng_a = np.random.default_rng(99)
    # x positions: alternate left and right annulus gap
    ann_bub_x = []
    for i in range(N_ANN):
        if i % 2 == 0:
            ann_bub_x.append(rng_a.uniform(ANN_LX+0.10, TUB_LX-0.14))
        else:
            ann_bub_x.append(rng_a.uniform(TUB_RX+0.10, ANN_RX-0.14))
    ann_bub_x = np.array(ann_bub_x)
    ann_bub_phase = np.linspace(0, 1, N_ANN, endpoint=False)   # stagger phases

    # Tubing bubbles: ascend from injection point to top
    N_TUB = 32
    rng_t = np.random.default_rng(55)
    tub_bub_x = rng_t.uniform(TUB_LX+0.12, TUB_RX-0.12, N_TUB)
    tub_bub_phase = np.linspace(0, 1, N_TUB, endpoint=False)

    ANN_HEIGHT = Y_TOP - Y_INJ
    TUB_HEIGHT = Y_TOP - Y_BOT

    # Create Circle patches (start invisible)
    ann_circles = [Circle((ann_bub_x[i], Y_TOP-0.3), 0.001,
                          facecolor='#ffffaa', edgecolor='#bbaa44',
                          lw=0.6, alpha=0.0, zorder=2)
                   for i in range(N_ANN)]
    tub_circles = [Circle((tub_bub_x[i], Y_BOT+0.2), 0.001,
                          facecolor='#e0e0e0', edgecolor='#999999',
                          lw=0.5, alpha=0.0, zorder=2)
                   for i in range(N_TUB)]
    for c in ann_circles + tub_circles:
        ax.add_patch(c)

    # ── w_out animated label ──────────────────────────────────────────────────
    wout_lbl = ax.text(cx+s+1.60, PROD_Y-0.32, '',
                       ha='left', va='center', fontsize=8,
                       color='#228844', fontweight='bold', zorder=9)

    # ── Update function ───────────────────────────────────────────────────────
    SPEED_ANN = 0.018   # fraction of annulus height per frame (downward)
    SPEED_TUB = 0.020   # fraction of tubing height per frame (upward)

    def update(frame):
        # ── Annulus bubbles (gas going DOWN) ──────────────────────────────────
        frac_a = np.clip(float(w_G_in[frame])/_wGmax, 0, 1) if w_G_in is not None else 0.5
        n_a = max(1, int(round(frac_a * N_ANN)))
        r_a = 0.055 + 0.060 * frac_a
        for i, c in enumerate(ann_circles):
            if i < n_a:
                phase = (ann_bub_phase[i] + frame * SPEED_ANN) % 1.0
                y = Y_TOP - 0.15 - phase * (ANN_HEIGHT - 0.30)
                c.center = (ann_bub_x[i], y)
                c.radius = r_a * (0.7 + 0.3*np.sin(i*1.3))   # slight variety
                # fade in/out near top and bottom
                fade = np.clip(min(phase, 1-phase) * 12, 0, 1)
                c.set_alpha(0.75 * fade)
            else:
                c.radius = 0.0; c.set_alpha(0.0)

        # ── Tubing bubbles (gas+oil going UP) ─────────────────────────────────
        # Use x1 (gas mass in tubing) to set bubble intensity
        frac_t = np.clip(float(x1[frame]) / max(x1.max(), 1e-9), 0, 1)
        n_t = max(1, int(round(frac_t * N_TUB)))
        r_t = 0.042 + 0.050 * frac_t
        for i, c in enumerate(tub_circles):
            if i < n_t:
                phase = (tub_bub_phase[i] + frame * SPEED_TUB) % 1.0
                y = Y_BOT + 0.15 + phase * (TUB_HEIGHT - 0.35)
                c.center = (tub_bub_x[i], y)
                c.radius = r_t * (0.65 + 0.35*np.sin(i*2.1))
                fade = np.clip(min(phase, 1-phase) * 10, 0, 1)
                c.set_alpha(0.55 * fade)
            else:
                c.radius = 0.0; c.set_alpha(0.0)

        # ── Pressure gauges ───────────────────────────────────────────────────
        _pdata_frame = {
            'P_at': P_at,  'P_ab': P_ab,
            'P_bh': P_bh,  'P_tt': P_tt,
        }
        for key, (gauge, base_color) in gauges.items():
            arr = _pdata_frame[key]
            low, high = gauge.limits
            val = float(arr[frame]) if arr is not None else (low + high) / 2
            fraction = (val - low) / max(high - low, 1e-9)
            color = plt.cm.RdYlGn_r(fraction) if key == 'P_bh' else base_color
            gauge.update(val, color=color)

        # ── Side bars ─────────────────────────────────────────────────────────
        for bar, arr, unit in zip(bars, bar_arrs, bar_units):
            val = float(arr[frame])
            bar.update(val, text=f'{val:.2f} {unit}')

        # ── w_out label ───────────────────────────────────────────────────────
        wout_lbl.set_text(f'{float(w_out[frame]):.3f} kg/s')
        # ── gas in label ──────────────────────────────────────────────────────
        if w_G_in is not None:
            gas_in_txt.set_text(f'{float(w_G_in[frame]):.3f} kg/s')
        # ── u1/u2 labels ──────────────────────────────────────────────────────
        u1_txt.set_text(f'{float(u1[frame]):.2f}')
        u2_txt.set_text(f'{float(u2[frame]):.2f}')

        return (ann_circles + tub_circles
                + [artist for gauge, _ in gauges.values() for artist in gauge.artists]
                + [artist for bar in bars for artist in bar.artists]
                + [wout_lbl, gas_in_txt, u1_txt, u2_txt])

    return _initialized(update)


def pHNeutralization(axis, *, h, pH, h_max=25.0,
                     q1=None, q2=None, q3=None):
    """Draw neutralization-reactor height, pH and optional feed labels.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Two-dimensional axes on which to draw the scene.
    h : array_like of shape (N,)
        Liquid-height trajectory in cm, normally model state row 2.
    pH : array_like of shape (N,)
        Dimensionless pH trajectory, normally algebraic state row 3. The helper
        displays it without solving the chemical constraint.
    h_max : float, default=25.0
        Positive finite height display scale in cm for vessel fill and the height
        gauge.
    q1 : array_like of shape (N,) or None, default=None
        Optional acid-feed trajectory in mL/s, displayed as supplied.
        None omits the numeric label.
    q2 : array_like of shape (N,) or None, default=None
        Optional buffer-feed trajectory in mL/s, displayed as supplied.
        None omits the numeric label.
    q3 : array_like of shape (N,) or None, default=None
        Optional alkaline-feed trajectory in mL/s, displayed as supplied.
        None omits the numeric label.

    Returns
    -------
    update : callable
        Initialized callback ``update(frame)``. The zero-based integer frame
        selects a sample and the callback returns the modified Matplotlib
        artists. Frame zero is drawn before this factory returns.

    Notes
    -----
    All supplied trajectories must be finite, nonempty and have the same
    sample count ``N``. Repeat constant values over the sample grid.

    Height and pH affect the vessel fill and colour; clipping is visual only.
    Flow values in mL/s are numerically identical to values in cm^3/s.
    Mixer motion is schematic and does not change the simulated solution.
    """
    import matplotlib.patches as _mp
    from matplotlib.patches import FancyBboxPatch, Circle, Arc, Ellipse

    values = {"h": h, "pH": pH}
    for name, value in (("q1", q1), ("q2", q2), ("q3", q3)):
        if value is not None:
            values[name] = value
    trajectories = _matching_trajectories(**values)
    h, pH = trajectories["h"], trajectories["pH"]
    q1, q2, q3 = (trajectories.get(name) for name in ("q1", "q2", "q3"))
    if not np.isfinite(h_max) or h_max <= 0:
        raise ValueError("h_max must be a positive finite value")
    ax = axis
    ax.set_xlim(0, 8.2); ax.set_ylim(0, 9.5)
    ax.set_aspect('equal'); ax.axis('off')

    def _ph_col(ph_val):
        t = np.clip(ph_val/14, 0, 1)
        return (max(0.0,1-2*t), float(1-2*abs(t-0.5)), max(0.0,2*t-1))

    # ── layout ───────────────────────────────────────────────────────────────
    LX=0.5; BY=0.8; VW=5.0; VH=6.2; RAD=0.32
    CX=LX+VW/2
    MOTOR_TOP=BY+VH+0.05; MOTOR_H=0.36; MOTOR_W=0.38
    BLADE_R=0.65; BLADE_Y_FRAC=0.38
    GAUGE_R=0.65

    # ── static vessel ─────────────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((LX,BY), VW, VH,
        boxstyle=f'round,pad=0,rounding_size={RAD}',
        facecolor='none', edgecolor='#334455', lw=2.0, zorder=3))

    # ── motor housing ─────────────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch(
        (CX-MOTOR_W/2, MOTOR_TOP), MOTOR_W, MOTOR_H,
        boxstyle='round,pad=0.04', facecolor='#dddddd', edgecolor='#555',
        lw=1.0, zorder=6))

    # ── animated water fill (colour + height) ─────────────────────────────────
    h0_frac = np.clip(float(h[0])/h_max, 0, 1)
    fh0 = VH*h0_frac
    water_ph = FancyBboxPatch((LX+0.06,BY+0.06), VW-0.12, max(fh0-0.06,0.02),
        boxstyle=f'round,pad=0,rounding_size={RAD-0.08}',
        facecolor=_ph_col(float(pH[0])), edgecolor='none', alpha=0.38, zorder=2)
    water_blue = FancyBboxPatch((LX+0.06,BY+0.06), VW-0.12, max(fh0-0.06,0.02),
        boxstyle=f'round,pad=0,rounding_size={RAD-0.08}',
        facecolor='#c8e4f5', edgecolor='none', alpha=0.48, zorder=2)
    ax.add_patch(water_ph); ax.add_patch(water_blue)

    # ── outlet pipe ────────────────────────────────────────────────────────
    PIPE_Y = BY + 0.50
    ax.plot([LX+VW+0.05, LX+VW+1.6], [PIPE_Y,PIPE_Y],
            color='#444', lw=2, zorder=4)
    ax.annotate('', xy=(LX+VW+2.3,PIPE_Y), xytext=(LX+VW+1.9,PIPE_Y),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    ax.text(CX+VW/2+2.3, PIPE_Y, 'q4', fontsize=9, color='#333', va='center', fontweight='bold')
    GCx=LX+VW+0.95; GCy=PIPE_Y+GAUGE_R+0.18

    # ── pH and height gauges ────────────────────────────────────────────────

    def _static_gauge(cx, cy, r, label):
        ax.add_patch(Circle((cx,cy), r, facecolor='#f8f8f8',
                     edgecolor='#444', lw=1.5, zorder=9))
        ax.add_patch(Arc((cx,cy), r*1.7, r*1.7,
                     angle=0, theta1=-30, theta2=210,
                     color='#ddd', lw=2.5, zorder=10))
        ax.text(cx, cy-r*0.45, label, ha='center', va='center',
                fontsize=7.5, color='#555', zorder=12)

    _static_gauge(GCx, GCy, GAUGE_R, 'pH')
    _static_gauge(GCx, 1.75*GCy, GAUGE_R, 'h')

    gauge_data = {}
    for key, cx, cy, vmin, vmax in [('pH',GCx,GCy,0,14),
                                  ('h (cm)',GCx,1.75*GCy,0,h_max)]:
        arc = Arc((cx,cy), GAUGE_R*1.7, GAUGE_R*1.7,
                  angle=0, theta1=210, theta2=210, color='gray', lw=2.5, zorder=11)
        ax.add_patch(arc)
        needle, = ax.plot([], [], color='#333', lw=1.2, zorder=12)
        ax.add_patch(Circle((cx,cy), GAUGE_R*0.10, color='#333', zorder=13))
        lbl = ax.text(cx, cy+GAUGE_R*0.45, '',
                      ha='center', va='center', fontsize=7.5,
                      fontweight='bold', color='#222', zorder=13)
        gauge_data[key] = DialGauge(
            arc, needle, lbl, (cx, cy), GAUGE_R, (vmin, vmax)
        )

    '''ax.add_patch(Circle((GCx,GCy), GAUGE_R,
                 facecolor='#f8f8f8', edgecolor='#444', lw=1.5, zorder=9))
    ax.add_patch(Arc((GCx,GCy), GAUGE_R*1.7, GAUGE_R*1.7,
                 angle=0, theta1=-30, theta2=210, color='#ddd', lw=2.5, zorder=10))
    ax.text(GCx, GCy-GAUGE_R*0.50, 'pH', ha='center', va='center',
            fontsize=8, color='#555', fontweight='bold', zorder=12)
    ax.plot([GCx,GCx],[PIPE_Y,GCy-GAUGE_R], color='#444', lw=1.5, zorder=4)
    ph_arc = Arc((GCx,GCy), GAUGE_R*1.7, GAUGE_R*1.7,
                 angle=0, theta1=210, theta2=210, color='green', lw=2.5, zorder=11)
    ax.add_patch(ph_arc)
    ph_needle, = ax.plot([],[],color='#333',lw=1.2,zorder=12)
    ax.add_patch(Circle((GCx,GCy),GAUGE_R*0.10,color='#333',zorder=13))
    ph_lbl = ax.text(GCx, GCy+GAUGE_R*0.45, '',
                     ha='center', va='center', fontsize=8,
                     fontweight='bold', color='#222', zorder=13)'''

    # ── inlet arrows ──────────────────────────────────────────────────────────
    qi_params = [(CX+0.75,'q1',"#5faa22"),
                (CX+1.95,    'q2',"#666666"),
                (CX-1.45,'q3',"#cc2200")]
    for xi, qi_lbl, col in qi_params:
        ax.annotate('', xy=(xi,MOTOR_TOP+0.05), xytext=(xi,MOTOR_TOP+0.65),
                    arrowprops=dict(arrowstyle='->', color=col, lw=1.4))
        ax.text(xi+0.12, MOTOR_TOP+0.45, qi_lbl,
                fontsize=8.5, color=col, fontweight='bold')

    q_lbls = {}
    for qi_arr, xi, key, col in zip([q1, q2, q3], [qi[0] for qi in qi_params], [qi[1] for qi in qi_params], [qi[2] for qi in qi_params]):
        if qi_arr is not None:
            lbl = ax.text(xi, MOTOR_TOP+0.72, '',
                          ha='center', fontsize=7.5, color=col, zorder=8,
                          bbox=dict(facecolor='white',edgecolor='none',alpha=0.75,pad=1))
            q_lbls[key] = (lbl, qi_arr)

    # ── static mixer shaft ────────────────────────────────────────────────────
    shaft, = ax.plot([CX,CX],[MOTOR_TOP, BY+VH*0.22],
                     color='#999', lw=2.0, zorder=6)
    blade_l, = ax.plot([],[],color='#555',lw=2.5,solid_capstyle='round',zorder=7)
    blade_r, = ax.plot([],[],color='#555',lw=2.5,solid_capstyle='round',zorder=7)
    hub = Circle((CX,BY+VH*0.22),0.07,color='#555',zorder=8)
    ax.add_patch(hub)
    depth_ell = Ellipse((CX,BY+VH*0.22-0.07),BLADE_R*1.7,0.13,
        facecolor='none',edgecolor='#aaa',lw=0.8,alpha=0.5,zorder=5)
    ax.add_patch(depth_ell)
    mixer = Mixer(shaft, (blade_l, blade_r), hub, depth_ell, CX, MOTOR_TOP, BLADE_R)

    # ── update ────────────────────────────────────────────────────────────────
    def update(frame):
        h_frac = np.clip(float(h[frame])/h_max, 0, 1)
        fh = VH*h_frac
        fh_plot = max(fh-0.06, 0.02)
        water_ph.set_height(fh_plot)
        water_ph.set_facecolor(_ph_col(float(pH[frame])))
        water_blue.set_height(fh_plot)

        # h level gauge (right side)
        '''H_X = LX - 0.55
        ax.annotate('', xy=(H_X, BY+0.10), xytext=(H_X, BY+VH*0.50),
                    arrowprops=dict(arrowstyle='<->', color='#555', lw=1.1))
        h_top_mk, = ax.plot([H_X-0.12, H_X+0.12], [BY+0.10, BY+0.10],
                            color='#555', lw=1.0)
        h_lbl_txt = ax.text(H_X-0.22, BY+VH*0.25, '',
                            ha='center', fontsize=10, color='#555')

        h_top_mk.set_ydata([BY+fh_plot+0.06, BY+fh_plot+0.06])
        h_lbl_txt.set_text(f'{float(h[frame]):.2f} m')'''

        # gauges
        for key, val_arr in [('pH',pH),('h (cm)',h)]:
            val = float(val_arr[frame])
            gauge = gauge_data[key]
            low, high = gauge.limits
            frac = np.clip((val-low)/max(high-low, 1e-9), 0, 1)
            col = plt.cm.RdYlBu_r(frac)
            gauge.update(val, color=col, fmt='.2f')

        # flow labels
        for key,(lbl,arr) in q_lbls.items():
            lbl.set_text(f'{float(arr[frame]):.1f} mL/s')

        # mixer
        blade_y = BY + fh*BLADE_Y_FRAC
        mixer.update(frame, blade_y)

        return ([water_ph, water_blue, *mixer.artists]
                + [artist for gauge in gauge_data.values() for artist in gauge.artists]
                + [v[0] for v in q_lbls.values()])

    return _initialized(update)

__all__ = ['JohanssonTanks', 'BatchBioreactor', 'CSTR', 'OilWell', 'pHNeutralization']
