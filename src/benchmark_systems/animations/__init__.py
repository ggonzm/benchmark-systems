"""
benchmark_systems.animations
============================

Matplotlib-based animation helpers for all benchmark systems.

Each public function follows the same two-step pattern:

1. Constructor call  – pass the axes and the pre-computed simulation
   trajectory.  The function draws the initial frame and returns an
   update callable.
2. AutoAnimation call – pass the figure, the update callable, and
   timing parameters to get a FuncAnimation object.

Example
-------
>>> import matplotlib.pyplot as plt
>>> from benchmark_systems.animations import Pendulum, AutoAnimation
>>> fig, ax = plt.subplots()
>>> update = Pendulum(ax, L=2.0, theta=theta_traj)
>>> anim, _ = AutoAnimation(fig, update, duration=20.0, dt=0.02)
>>> plt.show()
"""

from __future__ import annotations
from typing import Callable, Sequence, Literal, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation


# ---------------------------------------------------------------------------
# AutoAnimation
# ---------------------------------------------------------------------------

def AutoAnimation(fig, update_fn: Callable, duration: float, dt: float, speed: int=1, fps: Optional[int]=None, time_label=True, time_unit: Literal['s', 'min', 'hr']='s', time_fmt='.1f', **kwargs):
    '''
    Creates a FuncAnimation object with the given update function and duration, adjusting automatically the number of
    frames for the given interval between them.

    See answer from https://stackoverflow.com/questions/22010586/how-to-adjust-animation-duration

    Parameters
    ----------
    fig : plt.Figure
        Figure object for the animation.
    update_fn : Callable
        Update function for the animation.
    duration : float
        Duration of the animation.
    dt : float
        Time step between simulation points.
    speed : int, optional
        Speed factor for the animation. Default is 1.
        Youtube style: 1, 2, 3, etc.
    fps: int, optional
        Frames per second for the animation. Default is 1/dt * TIME_2_SEC (e.g., 1/dt for seconds, 60/dt for minutes, etc.).
    time_label : bool or plt.Text
        If True  (default) a text object is created at the bottom centre of
        the figure and updated automatically with the simulation time.
        If a ``plt.Text`` instance is passed, that object is used instead
        (useful when the caller already placed a text in a specific position).
        If False, no time label is shown.
    time_unit : Literal['s', 'min', 'hr']
        Unit string appended to the time value (default ``'s'``).
    time_fmt : str
        Python format spec for the time value (default ``'.1f'``).
    **kwargs
        Additional arguments for the FuncAnimation object.
        Different from the original function, the repeat argument is False by default.

    Returns
    -------
    FuncAnimation
        Animation object.
    range
        Range object representing the frames selected between the animation points.
    '''
    assert speed >= 1, "Speed factor must be an integer greater than or equal to 1."
    repeat          = kwargs.pop('repeat', False)
    TIME_2_SEC      = 1 if time_unit == 's' else 60 if time_unit == 'min' else 3600
    _fps            = fps if fps is not None else 1 / dt * TIME_2_SEC
    MSEC_PER_FRAME  = 1000 / _fps
    points          = int(duration / dt)
    frames          = range(0, points, speed)

    # ── time label ────────────────────────────────────────────────────────────
    if time_label is True:
        _time_txt = fig.text(
            0.5, 0.01, f't = 0{time_unit}',
            ha='center', va='bottom', fontsize=10, color='#333333',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2))
    elif time_label is False:
        _time_txt = None
    else:
        # caller supplied a plt.Text object
        _time_txt = time_label
 
    def _wrapped_update(frame):
        artists = update_fn(frame)
        if _time_txt is not None:
            t_sim = frame * dt
            _time_txt.set_text(f't = {t_sim:{time_fmt}} {time_unit}')
            # Return the text artist so blitting works correctly
            if isinstance(artists, (list, tuple)):
                return list(artists) + [_time_txt]
            return [_time_txt]
        return artists

    anim = FuncAnimation(fig, _wrapped_update, frames=frames, repeat=repeat, interval=MSEC_PER_FRAME, **kwargs)
    return anim, frames


# ---------------------------------------------------------------------------
# Pendulum
# ---------------------------------------------------------------------------

def Pendulum(axis, *, L, theta, pivot=(0.0, 0.0)):
    px, py = pivot
    axis.set_xlim(px - L * 1.5, px + L * 1.5)
    axis.set_ylim(py - L * 1.5, py + L * 1.5)
    axis.set_aspect('equal')
    axis.axhline(py, color='gray', lw=1, alpha=0.4)

    bob_x0 = px + L * np.sin(theta[0])
    bob_y0 = py - L * np.cos(theta[0])

    rod, = axis.plot([px, bob_x0], [py, bob_y0], color='dimgray', lw=2, zorder=2)
    bob  = plt.Circle((bob_x0, bob_y0), radius=L*0.08, color='steelblue', zorder=3)
    piv  = plt.Circle((px, py), radius=L*0.04, color='black', zorder=4)
    axis.add_patch(bob)
    axis.add_patch(piv)

    def update(frame):
        bx = px + L * np.sin(theta[frame])
        by = py - L * np.cos(theta[frame])
        rod.set_data([px, bx], [py, by])
        bob.set_center((bx, by))
        return rod, bob

    return update


# ---------------------------------------------------------------------------
# DC Motor
# ---------------------------------------------------------------------------

def DCMotor(axis, *, theta, omega, current):
    axis.set_xlim(-2.5, 2.5)
    axis.set_ylim(-1.5, 1.5)
    axis.set_aspect('equal')
    axis.axis('off')

    axis.add_patch(plt.Circle((0, 0), radius=0.9, color='royalblue', alpha=0.3, zorder=1))
    axis.add_patch(plt.Circle((0, 0), radius=0.9, fill=False, color='royalblue', lw=2, zorder=2))

    r = 0.85
    shaft, = axis.plot([0, r*np.sin(theta[0])], [0, r*np.cos(theta[0])],
                       color='navy', lw=3, zorder=3)

    omega_max = max(np.abs(omega).max(), 1e-3)
    curr_max  = max(np.abs(current).max(), 1e-3)

    axis.add_patch(mpatches.FancyBboxPatch((1.2, -0.8), 0.3, 1.6,
        boxstyle='round,pad=0.02', facecolor='lightgray', edgecolor='gray'))
    h_o0 = 0.8 * np.clip(omega[0]/omega_max, -1, 1)
    bar_omega = mpatches.FancyBboxPatch((1.22, min(0.0, h_o0)), 0.26, abs(h_o0),
        boxstyle='round,pad=0.01', facecolor='steelblue')
    axis.add_patch(bar_omega)
    axis.text(1.35, -1.05, 'omega', ha='center', fontsize=8, color='navy')

    axis.add_patch(mpatches.FancyBboxPatch((1.7, -0.8), 0.3, 1.6,
        boxstyle='round,pad=0.02', facecolor='lightgray', edgecolor='gray'))
    h_c0 = 0.8 * np.clip(current[0]/curr_max, -1, 1)
    bar_curr = mpatches.FancyBboxPatch((1.72, min(0.0, h_c0)), 0.26, abs(h_c0),
        boxstyle='round,pad=0.01', facecolor='tomato')
    axis.add_patch(bar_curr)
    axis.text(1.85, -1.05, 'I', ha='center', fontsize=9, color='firebrick')

    info = axis.text(0, -1.25, '', ha='center', fontsize=8, color='dimgray')

    def update(frame):
        shaft.set_data([0, r*np.sin(theta[frame])], [0, r*np.cos(theta[frame])])
        h_o = 0.8 * np.clip(omega[frame]/omega_max, -1, 1)
        bar_omega.set_bounds(1.22, min(0.0, h_o), 0.26, abs(h_o))
        h_c = 0.8 * np.clip(current[frame]/curr_max, -1, 1)
        bar_curr.set_bounds(1.72, min(0.0, h_c), 0.26, abs(h_c))
        info.set_text(f'theta={np.degrees(theta[frame]):.1f}º  '
                      f'omega={omega[frame]:.2f} rad/s  '
                      f'I={current[frame]:.3f} A')
        return shaft, bar_omega, bar_curr, info

    return update


# ---------------------------------------------------------------------------
# Cart-Pendulum
# ---------------------------------------------------------------------------

def CartPendulum(axis, *, L, pos, theta, obstacles=[]):
    axis.set_xlim(np.min(pos) - L, np.max(pos) + L)
    axis.set_ylim(-L*1.5, L*1.5)
    axis.set_aspect('equal')

    bob  = plt.Circle((L*np.sin(theta[0]) + pos[0], -L*np.cos(theta[0])),
                       radius=0.1, color='black', zorder=3)
    line, = axis.plot([], [], color='gray', zorder=2)
    cart  = plt.Rectangle((pos[0]-0.5, -0.25), 1, 0.5, color='royalblue', zorder=1)
    rail  = plt.Rectangle((axis.get_xlim()[0], -0.1),
                           axis.get_xlim()[1]-axis.get_xlim()[0], 0.2,
                           color='gray', zorder=0, alpha=0.5)
    axis.add_patch(bob); axis.add_patch(cart); axis.add_patch(rail)

    for (ox, oy), r in obstacles:
        axis.add_patch(plt.Circle((ox, oy), r, color='red', zorder=4))

    from functools import partial

    def _update(frame, _bob, _cart, _line, _pos, _theta):
        bx = L*np.sin(_theta[frame]) + _pos[frame]
        by = -L*np.cos(_theta[frame])
        _bob.set_center((bx, by))
        _cart.set_xy([_pos[frame]-0.5, -0.25])
        _line.set_data([bx, _pos[frame]], [by, 0.0])
        return _bob, _cart, _line

    return partial(_update, _bob=bob, _cart=cart, _line=line, _pos=pos, _theta=theta)


# ---------------------------------------------------------------------------
# Multi-level Cart-Pendulum
# ---------------------------------------------------------------------------

def MultiCartPendulum(axis, *, L, pos, thetas):
    npend  = len(L)
    L_tot  = sum(L)
    margin = L_tot + 0.5

    axis.set_xlim(np.min(pos) - margin, np.max(pos) + margin)
    axis.set_ylim(-L_tot*1.2, L_tot*1.2)
    axis.set_aspect('equal')

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, npend))

    def _joints(frame):
        pts = [(pos[frame], 0.0)]
        for i in range(npend):
            x0, y0 = pts[-1]
            pts.append((x0 + L[i]*np.sin(thetas[i][frame]),
                         y0 - L[i]*np.cos(thetas[i][frame])))
        return pts

    axis.add_patch(plt.Rectangle((axis.get_xlim()[0], -0.12),
                                  axis.get_xlim()[1]-axis.get_xlim()[0], 0.12,
                                  color='gray', alpha=0.5, zorder=0))
    cart_patch = plt.Rectangle((pos[0]-0.4, -0.25), 0.8, 0.25,
                                 color='royalblue', zorder=1)
    axis.add_patch(cart_patch)

    pts0 = _joints(0)
    rods, joints = [], []
    for i in range(npend):
        rod, = axis.plot([pts0[i][0], pts0[i+1][0]],
                         [pts0[i][1], pts0[i+1][1]],
                         color=colors[i], lw=3, zorder=2)
        jdot = plt.Circle(pts0[i+1], radius=0.05, color=colors[i], zorder=3)
        axis.add_patch(jdot)
        rods.append(rod); joints.append(jdot)

    def update(frame):
        pts = _joints(frame)
        cart_patch.set_xy([pts[0][0]-0.4, -0.25])
        for i in range(npend):
            rods[i].set_data([pts[i][0], pts[i+1][0]],
                              [pts[i][1], pts[i+1][1]])
            joints[i].set_center(pts[i+1])
        return tuple(rods) + tuple(joints) + (cart_patch,)

    return update


# ---------------------------------------------------------------------------
# Multimass Spring
# ---------------------------------------------------------------------------

def MultimassSpring(axis, *, K, thetas, u_true):
    """
    Multimass spring-disc system animation.
 
    Draws the system as seen in the classic schematic: two stepper-motor
    blocks on the left and right, N discs in perspective (ellipses viewed
    slightly from the side) connected by zigzag springs along a common
    horizontal shaft.  A dot on each disc rim shows the current angle.
 
    Parameters
    ----------
    axis : plt.Axes
    K : Sequence[float]
        Spring constants (n+1 values; used only for layout, not drawn).
    thetas : list of np.ndarray (N_frames,)
        Angular position of each disc (rad).  theta=0 → dot at top.
    u_true : list of np.ndarray (N_frames,) of length 2
        True angular positions of the two stepper motors.
    """
    import matplotlib.patches as mp
    from matplotlib.patches import Ellipse, FancyArrowPatch
 
    n = len(thetas)
 
    # --- Layout constants ---
    gap      = 2.2          # horizontal spacing between disc centres
    rx       = 0.18         # ellipse x half-axis (perspective depth)
    ry       = 0.75         # ellipse y half-axis (disc radius in plot)
    shaft_y  = 0.0          # y of the common shaft
    motor_w  = 0.55         # motor block half-width (x)
    motor_h  = 1.1          # motor block full height
    spring_h = 0.28         # spring zigzag amplitude
    n_zz     = 7            # number of zigzag segments per spring
 
    disc_cx = np.arange(n) * gap          # disc centre x positions
    left_motor_x  = disc_cx[0]  - gap     # centre of left motor block
    right_motor_x = disc_cx[-1] + gap     # centre of right motor block
 
    x_min = left_motor_x  - motor_w - 0.2
    x_max = right_motor_x + motor_w + 0.2
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(-ry * 1.6, ry * 1.6)
    axis.set_aspect('equal')
    axis.axis('off')
 
    # ---- Helper: zigzag spring between x0 and x1 at height y ----
    def _spring_points(x0, x1, y=shaft_y, amp=spring_h, nzz=n_zz):
        xs = np.linspace(x0, x1, nzz + 2)
        ys = np.zeros_like(xs)
        ys[1:-1] = [amp if i % 2 == 0 else -amp for i in range(nzz)]
        return xs, ys + y
 
    # ---- Static: shaft line ----
    axis.plot([left_motor_x + motor_w, right_motor_x - motor_w],
              [shaft_y, shaft_y], color='#555', lw=2.5, zorder=1)
 
    # ---- Static: motor blocks ----
    motor_color = '#444'
    for mx in [left_motor_x, right_motor_x]:
        axis.add_patch(mp.FancyBboxPatch(
            (mx - motor_w, shaft_y - motor_h/2), 2*motor_w, motor_h,
            boxstyle='round,pad=0.05', facecolor=motor_color,
            edgecolor='#222', lw=1.5, zorder=6))
 
    # ---- Layout: shaft stubs and motor ellipse positions ----
    # Structure (left side):  block | stub_a | motor_ellipse | stub_b | spring
    # Structure (right side): spring | stub_b | motor_ellipse | stub_a | block
    stub_a = rx * 1.2     # short gap between block face and motor ellipse
    stub_b = rx * 1.2     # short gap between motor ellipse and spring start
 
    # x-centres of the motor ellipses
    left_face_x  = left_motor_x  + motor_w + stub_a + rx
    right_face_x = right_motor_x - motor_w - stub_a - rx
 
    # spring endpoints
    left_spring_start  = left_face_x  + rx + stub_b
    right_spring_start = right_face_x - rx - stub_b
 
    # ---- Perspective convention ----
    # The system is viewed slightly from the left, so:
    #   - Shaft/spring arrives at the FRONT face (cx) of each disc/motor-ellipse
    #     coming from the left  → endpoint = cx  (centre, visible face)
    #   - Shaft/spring leaves from the BACK face (cx + rx) going to the right
    #     → startpoint = cx + rx  (hidden behind the disc)
    # This makes left-going connections clearly visible and right-going ones
    # appear to vanish behind each element.
 
    motor_ry     = motor_h / 2   # motor ellipses same height as the block
 
    # ---- Static: shaft — left block to left motor-ellipse (front face) ----
    axis.plot([left_motor_x + motor_w, left_face_x],
              [shaft_y, shaft_y], color='#888', lw=4, zorder=5)
    # left motor-ellipse back face → spring start
    axis.plot([left_face_x + rx, left_spring_start],
              [shaft_y, shaft_y], color='#888', lw=4, zorder=5)
    # right spring end → right motor-ellipse front face (arrives at centre)
    axis.plot([right_spring_start, right_face_x],
              [shaft_y, shaft_y], color='#888', lw=4, zorder=5)
    # right motor-ellipse back face → right block
    axis.plot([right_face_x + rx, right_motor_x - motor_w],
              [shaft_y, shaft_y], color='#888', lw=4, zorder=5)
 
    # ---- Static: springs ----
    # Each spring arrives at the front face (cx) of the right disc
    # and departs from the back face (cx + rx) of the left disc.
    spring_lines = []
    # left motor-ellipse back → disc 0 front
    xs, ys = _spring_points(left_face_x + rx, disc_cx[0])
    sl, = axis.plot(xs, ys, color='#888', lw=1.2, zorder=4)
    spring_lines.append(sl)
    # disc i back → disc i+1 front
    for i in range(n - 1):
        xs, ys = _spring_points(disc_cx[i] + rx, disc_cx[i + 1])
        sl, = axis.plot(xs, ys, color='#888', lw=1.2, zorder=4)
        # shaft from the left of the ellipse to the center of it
        axis.plot([disc_cx[i] - rx, disc_cx[i]], [shaft_y, shaft_y], color='#555', lw=2.5, zorder=5)
        spring_lines.append(sl)
    # disc n-1 back → right motor-ellipse front
    xs, ys = _spring_points(disc_cx[-1] + rx, right_face_x)
    sl, = axis.plot(xs, ys, color='#888', lw=1.2, zorder=4)
    axis.plot([disc_cx[-1] - rx, disc_cx[-1]], [shaft_y, shaft_y], color='#555', lw=2.5, zorder=5)
    spring_lines.append(sl)
 
    # ---- Static: disc ellipses — drawn BEFORE springs so they overlap them ----
    disc_color = '#cce0ff'
    disc_edge  = '#2255aa'
    for i in range(n):
        axis.add_patch(Ellipse((disc_cx[i], shaft_y),
                               width=2*rx, height=2*ry,
                               facecolor=disc_color, edgecolor=disc_edge,
                               lw=2, zorder=3))
        axis.add_patch(Ellipse((disc_cx[i], shaft_y),
                               width=2*rx*0.55, height=2*ry*0.55,
                               facecolor='none', edgecolor=disc_edge,
                               lw=0.8, alpha=0.4, zorder=3))
    
    # ---- Text at the bottom in order to see angular position of each disc ----
    ang_pos_txt = []
    for i in range(n):
        txt = axis.text(disc_cx[i], shaft_y - ry*1.2, f'θ{i+1}={np.degrees(thetas[i][0]):.1f} rad',
                        ha='center', va='top', fontsize=9, color='#333333', zorder=8)
        ang_pos_txt.append(txt)
 
    # ---- Static: motor ellipses — same height as motor block ----
    motor_face_x = [left_face_x, right_face_x]
    for fx in motor_face_x:
        axis.add_patch(Ellipse((fx, shaft_y),
                               width=2*rx, height=2*motor_ry,
                               facecolor='#666', edgecolor='#aaa',
                               lw=1.5, zorder=3))
        axis.add_patch(Ellipse((fx, shaft_y),
                               width=2*rx*0.55, height=2*motor_ry*0.55,
                               facecolor='none', edgecolor='#aaa',
                               lw=0.8, alpha=0.4, zorder=3))
        
    # ---- Text at the bottom in order to see angular position of each motor ----
    for i, fx in enumerate(motor_face_x):
        txt = axis.text(fx, shaft_y - motor_ry*1.2, f'θM{i+1}={np.degrees(u_true[i][0]):.1f} rad',
                        ha='center', va='top', fontsize=9, color='#cc2200', zorder=8)
        ang_pos_txt.append(txt)
 
    # ---- Animated: angle dots ----
    dot_r_x  = rx       * 0.92
    dot_r_y  = ry       * 0.92
    mdot_r_x = rx       * 0.92
    mdot_r_y = motor_ry * 0.92
 
    marks = []
    for i in range(n):
        th0 = float(thetas[i][0])
        dot, = axis.plot(
            [disc_cx[i] + dot_r_x * np.sin(th0)],
            [shaft_y    + dot_r_y * np.cos(th0)],
            'o', color='#cc2200', ms=6, zorder=7)
        marks.append(dot)
 
    motor_dots = []
    for fx, ut in zip(motor_face_x, u_true):
        th0 = float(ut[0])
        md, = axis.plot(
            [fx + mdot_r_x * np.sin(th0)],
            [shaft_y + mdot_r_y * np.cos(th0)],
            'o', color='#ffffff', ms=5, zorder=7,
            markeredgecolor='#cc2200', markeredgewidth=1.2)
        motor_dots.append(md)
 
    def update(frame):
        for i in range(n):
            th = float(thetas[i][frame])
            marks[i].set_data(
                [disc_cx[i] + dot_r_x * np.sin(th)],
                [shaft_y    + dot_r_y * np.cos(th)])
        for (fx, ut, md) in zip(motor_face_x, u_true, motor_dots):
            th = float(ut[frame])
            md.set_data(
                [fx + mdot_r_x * np.sin(th)],
                [shaft_y + mdot_r_y * np.cos(th)])
        # Update the text labels for angles, avoiding sign jumping when the angle is very close to zero
        for i, txt in enumerate(ang_pos_txt[:-2]):
            theta_sanitized = thetas[i][frame] if np.abs(thetas[i][frame]) > 1e-4 else 0.0
            txt.set_text(f'θ{i+1}={np.degrees(theta_sanitized):.1f}º')
        for i, txt in enumerate(ang_pos_txt[-2:]):
            u_true_sanitized = u_true[i][frame] if np.abs(u_true[i][frame]) > 1e-4 else 0.0
            txt.set_text(f'θM{i+1}={np.degrees(u_true_sanitized):.1f}º')
        return tuple(marks) + tuple(motor_dots)
 
    return update


# ---------------------------------------------------------------------------
# Johansson Four-Tank
# ---------------------------------------------------------------------------

def JohanssonTanks(axis, *, h_max, heights, pumps, gammas):
    """
    Four-tank system (Johansson) process diagram animation.
 
    Layout matches the reference diagram:
      - Tank 1 (lower-left),  Tank 2 (lower-right)
      - Tank 3 (upper-left),  Tank 4 (upper-right)
      - Pump 1 (left):  (1-γ1)·q1 → tank1,  γ1·q1 → tank4 (CROSS: upper-right)
      - Pump 2 (right): (1-γ2)·q2 → tank2,  γ2·q2 → tank3 (CROSS: upper-left)
      - Tank 3 drains into tank 1;  Tank 4 drains into tank 2.
 
    Water fills animate as true fractions of h_max.
    Numeric labels on pump outputs and valve branches update every frame.
 
    Parameters
    ----------
    axis : plt.Axes
    h_max : sequence of 4 floats   [h1_max, h2_max, h3_max, h4_max]  (cm)
    heights : list of 4 np.ndarray  [h1, h2, h3, h4]  (cm)
    pumps : list of 2 np.ndarray        [u1, u2]  pump setpoints (0-100%)
    gammas : list of 2 np.ndarray    [γ1, γ2]  valve fractions (0–1)
    """
    import matplotlib.patches as _mp
    from matplotlib.patches import Circle, Arc
 
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
    # u1/u2 input arrows from outside
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
        u1v = float(pumps[0][frame]); u2v = float(pumps[1][frame])
        g1v = float(gammas[0][frame]); g2v = float(gammas[1][frame])
        lbl_q1.set_text(f'{u1v:.1f}')
        lbl_1g1q1.set_text(f'{(1-g1v)*u1v:.1f}')
        lbl_g1q1.set_text(f'{g1v*u1v:.1f}')
        lbl_q2.set_text(f'{u2v:.1f}')
        lbl_1g2q2.set_text(f'{(1-g2v)*u2v:.1f}')
        lbl_g2q2.set_text(f'{g2v*u2v:.1f}')
        lbl_gamma1.set_text(f'{g1v:.2f}')
        lbl_gamma2.set_text(f'{g2v:.2f}')
        for i in range(4):
            meas_lbls[i].set_text(f'{float(heights[i][frame]):.1f} cm')
 
        return tuple(wp for wp,*_ in tank_info) + tuple(all_labels)
 
    return update


# ---------------------------------------------------------------------------
# Batch Bioreactor
# ---------------------------------------------------------------------------

def BatchBioreactor(axis, *, X, S, P, V, u=None, S_in=None,
                    V_max=None, X_max=None, S_max=None, P_max=None):
    """
    Batch bioreactor animation.

    Rectangular vessel with rounded corners (single FancyBboxPatch, no clipping).
    - Water level rises/falls with V
    - Green bubbles (X), purple bubbles (P), sand-like sediment (S)
    - Sliding bars on the right for X, S, P, V
    - Rotating mixer, feed inlet arrow (u), and feed concentration label (S_in)
    """
    import matplotlib.patches as _mp
    from matplotlib.patches import FancyBboxPatch, Circle, Arc, Ellipse

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

    bar_patches, bar_txts = [], []
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
        bar_patches.append((bar, BAR_H, vmax))
        bar_txts.append(vtxt)

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
        n_x = max(0, int(round(fx * N_BUB)))
        r_x = 0.04 + 0.08*fx
        for i, c in enumerate(bub_X):
            if i < n_x:
                y = BY + 0.50 + byX[i] * (fh - 0.60)
                c.set_center((bxX[i], y)); c.set_radius(r_x); c.set_alpha(0.58)
            else:
                c.set_radius(0.0); c.set_alpha(0.0)

        # bubbles P
        fp = np.clip(float(P[frame])/_Pmax, 0, 1)
        n_p = max(0, int(round(fp * N_BUB)))
        r_p = 0.04 + 0.08*fp
        for i, c in enumerate(bub_P):
            if i < n_p:
                y = BY + 0.50 + byP[i] * (fh - 0.60)
                c.set_center((bxP[i], y)); c.set_radius(r_p); c.set_alpha(0.58)
            else:
                c.set_radius(0.0); c.set_alpha(0.0)

        # side bars
        for (bar, bh, vmax), vtxt, arr in zip(bar_patches, bar_txts, state_arrs):
            val = float(arr[frame])
            frac = np.clip(val/vmax, 0, 1)
            bar.set_height(max((bh-0.02)*frac, 0.01))
            vtxt.set_text(f'{val:.2f}')

        # mixer
        ang = np.radians(frame * 6)
        blade_y = BY + fh * BLADE_Y_FRAC
        hub.set_center((CX, blade_y))
        depth_ell.set_center((CX, blade_y - 0.07))
        shaft.set_data([CX, CX], [MOTOR_TOP, blade_y])
        for sign, bl in [(1, blade_l), (-1, blade_r)]:
            a = ang + (0 if sign == 1 else np.pi)
            bx = CX + BLADE_R*np.cos(a)
            by_ = blade_y + BLADE_R*0.28*np.sin(a) - 0.06
            bl.set_data([CX, bx], [blade_y, by_])

        return ([water, sed, shaft, blade_l, blade_r, hub]
                + bub_X + bub_P
                + [b for b,*_ in bar_patches] + bar_txts)

    return update
 
 
def CSTR(axis, *, Ca, Cb, Tr, Tk, dQ=None,
         F=None, T_in=None, Ca_in=None,
         Ca_max=None, Cb_max=None,
         Tr_min=100.0, Tr_max=220.0,
         Tk_min=100.0, Tk_max=220.0,
         dQ_max_abs=8500.0):
    """
    CSTR animation.

    Rectangular rounded vessel. Ca/Cb as bubbles. Cooling jacket as
    coloured zigzags (red→blue with |dQ|). Two circular gauges Tr/Tk.
    Side bars for Ca and Cb.
    """
    import matplotlib.patches as _mp
    from matplotlib.patches import FancyBboxPatch, Circle, Arc, Ellipse

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
        gauge_data[key] = (arc, needle, lbl, cx, vmin, vmax)

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
    if F is not None or T_in is not None or Ca_in is not None:
        if F is not None:
            F_x = CX-1.35 if T_in is None else CX-1.85
            F_y = MOTOR_TOP+MOTOR_H*0.75 if Ca_in is None else MOTOR_TOP+MOTOR_H*1.4
            F_txt = ax.text(F_x, F_y, '', ha='right', fontsize=8, color='#cc2200')
        if T_in is not None:
            T_in_x = CX-1.35 if F is None else CX-0.75
            T_in_y = MOTOR_TOP+MOTOR_H*0.75 if Ca_in is None else MOTOR_TOP+MOTOR_H*1.4
            T_in_txt = ax.text(T_in_x, T_in_y, '', ha='right', fontsize=8, color='#cc2200')
        if Ca_in is not None:
            Ca_in_txt = ax.text(CX-0.90, MOTOR_TOP+MOTOR_H*0.75, '',
                        ha='right', fontsize=8, color='#cc2200')
    else:
        ax.text(CX-1.25, MOTOR_TOP+MOTOR_H*0.8, 'u', ha='right', fontsize=10, color='#cc2200', fontweight='bold')
    
    # ── outlet flow ────────────────────────────────────────────────────────────
    ax.annotate('', xy=(LX+VW+0.65, BY), xytext=(LX+VW-0.30, BY),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    out_txt = ax.text(LX+VW+0.70, BY, 'F', color='#333',
                      fontsize=8 if F is not None else 10,
                      fontweight='bold' if F is None else 'normal')

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
    bar_patches, bar_txts, bar_arrs = [], [], []
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
        bar_patches.append((bar, BAR_H, vmax))
        bar_txts.append(vtxt)
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
        dQ_lbl.set_text(f'dQ={dq/1000:.1f} kW')

        # bubbles Ca
        fca = np.clip(float(Ca[frame])/_Camax, 0, 1)
        n_a = max(0, int(round(fca*N_BUB))); r_a = 0.04+0.08*fca
        for i, c in enumerate(bub_a):
            if i < n_a:
                y = BY+0.35+byA[i]*(fill_h-0.50)
                c.set_center((bxA[i],y)); c.set_radius(r_a); c.set_alpha(0.58)
            else:
                c.set_radius(0.0); c.set_alpha(0.0)
        # bubbles Cb
        fcb = np.clip(float(Cb[frame])/_Cbmax, 0, 1)
        n_b = max(0, int(round(fcb*N_BUB))); r_b = 0.04+0.08*fcb
        for i, c in enumerate(bub_b):
            if i < n_b:
                y = BY+0.35+byB[i]*(fill_h-0.50)
                c.set_center((bxB[i],y)); c.set_radius(r_b); c.set_alpha(0.58)
            else:
                c.set_radius(0.0); c.set_alpha(0.0)

        # gauges
        for key, val_arr in [('Tr',Tr),('Tk',Tk)]:
            arc, needle, lbl, cx, vmin, vmax = gauge_data[key]
            val = float(val_arr[frame])
            frac = np.clip((val-vmin)/(vmax-vmin), 0, 1)
            col = plt.cm.RdYlBu_r(frac)
            sweep = 240*frac
            arc.theta1 = 210-sweep; arc.theta2 = 210
            arc.set_edgecolor(col)
            a_rad = np.radians(210-frac*240)
            nx = cx + GAUGE_R*0.65*np.cos(a_rad)
            ny = GCy + GAUGE_R*0.65*np.sin(a_rad)
            needle.set_data([cx,nx],[GCy,ny])
            lbl.set_text(f'{val:.1f}')

        # side bars
        for (bar, bh, vmax), vtxt, arr in zip(bar_patches, bar_txts, bar_arrs):
            val = abs(float(arr[frame])) if arr is not None else abs(dq)
            frac = np.clip(val/vmax, 0, 1)
            bar.set_height(max((bh-0.02)*frac, 0.01))
            vtxt.set_text(f'{float(arr[frame]) if arr is not None else dq:.2f}')

        # mixer
        ang = np.radians(frame*6)
        blade_y = BY + fill_h*BLADE_Y_FRAC
        hub.set_center((CX,blade_y))
        depth_ell.set_center((CX,blade_y-0.07))
        shaft.set_data([CX,CX],[MOTOR_TOP,blade_y])
        for sign, bl in [(1,blade_l),(-1,blade_r_ln)]:
            a = ang+(0 if sign==1 else np.pi)
            bx = CX+BLADE_R*np.cos(a)
            by_ = blade_y+BLADE_R*0.28*np.sin(a)-0.06
            bl.set_data([CX,bx],[blade_y,by_])

        return ([jac_l,jac_r,dQ_lbl,shaft,blade_l,blade_r_ln,hub]
                + list(gauge_data.keys())
                + bub_a+bub_b
                + [b for b,*_ in bar_patches]+bar_txts)

    return update


# ---------------------------------------------------------------------------
# Oil Well
# ---------------------------------------------------------------------------

def OilWell(axis, *, x0, x1, x2, P_tt, w_out, u1, u2,
            w_G_in=None, P_at=None, P_ab=None, P_bh=None,
            rho_m=None, alpha_L=None,
            w_G_in_max=None, w_out_max=None):
    """
    Oil well process animation.

    Draws a cross-section of the well matching the Jahanshahi schematic:
    - Annulus (outer tube, gas-filled, pale yellow)
    - Tubing (inner tube, crude oil + gas mixture, dark)
    - Gas lift choke (left side) injects gas into the annulus
    - Injection valve at the bottom of the annulus feeds into the tubing
    - Production choke at the top outputs the oil-gas mixture

    Animated elements
    -----------------
    - Annulus: small bright bubbles drift **downward** (gas sinking to injection
      valve). Count and size scale with w_G_in.
    - Tubing: small light bubbles drift **upward** through the dark crude oil.
      Count and size scale with the injected gas flow. Crude level is always full.
    - Sliding bar panel on the right shows all state/output variables.
    - Pressure gauges (P_at, P_ab, P_tt, P_bh) update numerically each frame.

    Parameters
    ----------
    axis : plt.Axes
    x0   : np.ndarray (N,)  gas mass in annulus (kg)
    x1   : np.ndarray (N,)  gas mass in tubing  (kg)
    x2   : np.ndarray (N,)  liquid mass in tubing (kg)
    P_tt : np.ndarray (N,)  tubing top pressure (bar)
    w_out: np.ndarray (N,)  total mass flow out (kg/s)
    u1   : np.ndarray (N,)  opening of injection valve (0-1)
    u2   : np.ndarray (N,)  opening of production choke (0-1)
    w_G_in   : np.ndarray (N,) optional  inlet gas mass flow (kg/s)
    P_at     : np.ndarray (N,) optional  annulus top pressure (bar)
    P_ab     : np.ndarray (N,) optional  annulus bottom pressure (bar)
    P_bh     : np.ndarray (N,) optional  bottom-hole pressure (bar)
    rho_m    : np.ndarray (N,) optional  mixture density at tubing top
    alpha_L  : np.ndarray (N,) optional  liquid volume fraction at tubing top
    w_G_in_max  : float   scale for bubble count in annulus
    w_out_max   : float   scale for bar display
    """
    import matplotlib.patches as _mp
    from matplotlib.patches import FancyBboxPatch, Circle

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
    gauge_arcs    = {}   # key → Arc patch
    gauge_needles = {}   # key → Line2D
    gauge_hubs    = {}   # key → Circle (for centre dot)
    gauge_lbls    = {}   # key → Text (numeric value)

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
        gauge_arcs[key] = (val_arc, gcx, gcy,
                           _p_ranges[key][0], _p_ranges[key][1], col)

        # Animated needle
        needle, = ax.plot([], [], color='#333', lw=1.2, zorder=12)
        gauge_needles[key] = needle

        # Centre hub
        hub = _Circ((gcx, gcy), GR*0.10, color='#333', zorder=13)
        ax.add_patch(hub)

        # Animated numeric text (above centre)
        lbl = ax.text(gcx, gcy + GR*0.22, '',
                      ha='center', va='center',
                      fontsize=6.5, fontweight='bold',
                      color='#222', zorder=13)
        gauge_lbls[key] = lbl

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
    bar_patches, bar_txts, bar_arrs, bar_units = [], [], [], []
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
        bar_patches.append((bar, BAR_H, vmax))
        bar_txts.append(vtxt); bar_arrs.append(arr); bar_units.append(unit)

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
        for key, (arc, gcx, gcy, vmin, vmax, col) in gauge_arcs.items():
            arr = _pdata_frame[key]
            val = float(arr[frame]) if arr is not None else (vmin+vmax)/2
            frac = np.clip((val - vmin) / max(vmax - vmin, 1e-9), 0, 1)
            sweep = 240 * frac
            arc.theta1 = 210 - sweep
            arc.theta2 = 210
            arc.set_edgecolor(plt.cm.RdYlGn_r(frac) if 'P_bh' in key
                              else col)
            a_rad = np.radians(210 - frac * 240)
            nx = gcx + GR * 0.65 * np.cos(a_rad)
            ny = gcy + GR * 0.65 * np.sin(a_rad)
            gauge_needles[key].set_data([gcx, nx], [gcy, ny])
            gauge_lbls[key].set_text(f'{val:.1f}')

        # ── Side bars ─────────────────────────────────────────────────────────
        for (bar, bh, vmax), vtxt, arr, unit in zip(bar_patches, bar_txts, bar_arrs, bar_units):
            val = float(arr[frame])
            frac = np.clip(val/vmax, 0, 1)
            bar.set_height(max((bh-0.02)*frac, 0.01))
            vtxt.set_text(f'{val:.2f} {unit}')

        # ── w_out label ───────────────────────────────────────────────────────
        wout_lbl.set_text(f'{float(w_out[frame]):.3f} kg/s')
        # ── gas in label ──────────────────────────────────────────────────────
        if w_G_in is not None:
            gas_in_txt.set_text(f'{float(w_G_in[frame]):.3f} kg/s')
        # ── u1/u2 labels ──────────────────────────────────────────────────────
        u1_txt.set_text(f'{float(u1[frame]):.2f}')
        u2_txt.set_text(f'{float(u2[frame]):.2f}')

        return (ann_circles + tub_circles
                + list(gauge_needles.values())
                + list(gauge_lbls.values())
                + [b for b,*_ in bar_patches]
                + bar_txts + [wout_lbl])

    return update


# ---------------------------------------------------------------------------
# pH Neutralization
# ---------------------------------------------------------------------------

def pHNeutralization(axis, *, h, pH, h_max=25.0,
                     q1=None, q2=None, q3=None):
    """
    pH neutralization reactor animation.

    Rectangular rounded vessel. Fill colour changes with pH (red/green/blue).
    pH circular gauge on the outlet pipe.
    Inlet arrows q1/q2/q3 with optional numeric labels. Rotating mixer.
    """
    import matplotlib.patches as _mp
    from matplotlib.patches import FancyBboxPatch, Circle, Arc, Ellipse

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
                                  ('h (m)',GCx,1.75*GCy,0,h_max)]:
        arc = Arc((cx,cy), GAUGE_R*1.7, GAUGE_R*1.7,
                  angle=0, theta1=210, theta2=210, color='gray', lw=2.5, zorder=11)
        ax.add_patch(arc)
        needle, = ax.plot([], [], color='#333', lw=1.2, zorder=12)
        ax.add_patch(Circle((cx,cy), GAUGE_R*0.10, color='#333', zorder=13))
        lbl = ax.text(cx, cy+GAUGE_R*0.45, '',
                      ha='center', va='center', fontsize=7.5,
                      fontweight='bold', color='#222', zorder=13)
        gauge_data[key] = (arc, needle, lbl, cx, cy, vmin, vmax)
    
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
        for key, val_arr in [('pH',pH),('h (m)',h)]:
            arc, needle, lbl, cx, cy, vmin, vmax = gauge_data[key]
            val = float(val_arr[frame])
            frac = np.clip((val-vmin)/(vmax-vmin), 0, 1)
            col = plt.cm.RdYlBu_r(frac)
            sweep = 240*frac
            arc.theta1 = 210-sweep; arc.theta2 = 210
            arc.set_edgecolor(col)
            a_rad = np.radians(210-frac*240)
            nx = cx + GAUGE_R*0.65*np.cos(a_rad)
            ny = cy + GAUGE_R*0.65*np.sin(a_rad)
            needle.set_data([cx,nx],[cy,ny])
            lbl.set_text(f'{val:.2f}')

        # flow labels
        for key,(lbl,arr) in q_lbls.items():
            lbl.set_text(f'{float(arr[frame]):.1f} cm³/s')

        # mixer
        ang = np.radians(frame*6)
        blade_y = BY + fh*BLADE_Y_FRAC
        hub.set_center((CX,blade_y))
        depth_ell.set_center((CX,blade_y-0.07))
        shaft.set_data([CX,CX],[MOTOR_TOP,blade_y])
        for sign,bl in [(1,blade_l),(-1,blade_r)]:
            a = ang+(0 if sign==1 else np.pi)
            bx = CX+BLADE_R*np.cos(a)
            by_ = blade_y+BLADE_R*0.28*np.sin(a)-0.06
            bl.set_data([CX,bx],[blade_y,by_])

        return (list(gauge_data.keys())
                + [v[0] for v in q_lbls.values()])

    return update


# ---------------------------------------------------------------------------
# Quadrotor
# ---------------------------------------------------------------------------

def Quadrotor(fig: plt.Figure, axis_3d: Axes3D, axis_xy: plt.Axes, axis_xz: plt.Axes, axis_yz: plt.Axes,
        *, L: float, xyz: np.ndarray, angles: np.ndarray, xyz_ref: np.ndarray | None = None) -> Callable:
    '''
    Creates a Quadrotor update function for the animation.

    Parameters
    ----------
    fig : plt.Figure
        Figure object for the animation.
    axis_3d : Axes3D
        3D axis object for the animation.
    axis_xy : plt.Axes
        XY projection axis object.
    axis_xz : plt.Axes
        XZ projection axis object.
    axis_yz : plt.Axes
        YZ projection axis object.
    text : plt.Text
        Text object for displaying information.
    L : float
        Length of the arms of the quadrotor for visualization purposes, not necessarily related to the physical parameters of the system.
    xyz : np.ndarray
        Array with the quadrotor positions.
    angles : np.ndarray
        Array with the quadrotor angles (roll, pitch, yaw).
    xyz_ref : np.ndarray, optional
        Array with the reference position for the quadrotor (used for plotting a target point).
    '''
    xyz = xyz.T
    angles = angles.T
    if xyz_ref is not None:
        xyz_ref = xyz_ref.T
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    roll, pitch, yaw = angles[:, 0], angles[:, 1], angles[:, 2]
    colors = {
        'body': 'royalblue',
        'motors': 'dimgray',
        'arms': 'black',
        'trail': 'gray',
        'target': 'green',
    }

    # Plot limits
    margin = 2 * L
    xlim = (x.min() - margin, x.max() + margin)
    ylim = (y.min() - margin, y.max() + margin)
    zlim = (z.min() - margin, z.max() + margin)

    for ax in [axis_xy, axis_xz, axis_yz]:
        ax.set_aspect('equal')

    axis_3d.set_xlim(*xlim); axis_3d.set_ylim(*ylim); axis_3d.set_zlim(*zlim)
    axis_xy.set_xlim(*xlim); axis_xy.set_ylim(*ylim)
    axis_xz.set_xlim(*xlim); axis_xz.set_ylim(*zlim)
    axis_yz.set_xlim(*ylim); axis_yz.set_ylim(*zlim)

    # Local geometry: 4 motors in a cross (+X, -Y, -X, +Y)
    motors_body = np.array([[L, 0, 0], [0, -L, 0], [-L, 0, 0], [0, L, 0]])

    def get_world_motors(k):
        cr, sr = np.cos(roll[k]), np.sin(roll[k])
        cp, sp = np.cos(pitch[k]), np.sin(pitch[k])
        cy, sy = np.cos(yaw[k]), np.sin(yaw[k])
        R = np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                      [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                      [-sp, cp*sr, cp*cr]])
        return xyz[k] + motors_body @ R.T
    
    # System components - 3D
    mw = get_world_motors(0)
    arm_x, = axis_3d.plot(*zip(mw[0], mw[2]), color=colors['arms'], linewidth=3, zorder=1)
    arm_y, = axis_3d.plot(*zip(mw[1], mw[3]), color=colors['arms'], linewidth=3, zorder=1)
    center_dot, = axis_3d.plot([x[0]], [y[0]], [z[0]], 'o', color=colors['body'], markersize=3, zorder=1)
    motors_dot, = axis_3d.plot(mw[:,0], mw[:,1], mw[:,2], 'o', color=colors['motors'], markersize=7, zorder=1)
    trail_3d, = axis_3d.plot([], [], [], color=colors['trail'], linewidth=1, alpha=0.5, linestyle='--', zorder=1)
    if xyz_ref is not None:
        xyz_target, = axis_3d.plot([xyz_ref[0,0]], [xyz_ref[0,1]], [xyz_ref[0,2]], 'x', color=colors['target'], markersize=8, zorder=0)

    # System components - 2D projections
    # X-Y
    arm_x_xy,   = axis_xy.plot(*zip(mw[0,:2], mw[2,:2]), color=colors['arms'], linewidth=3, zorder=1)
    arm_y_xy,   = axis_xy.plot(*zip(mw[1,:2], mw[3,:2]), color=colors['arms'], linewidth=3, zorder=1)
    center_xy,  = axis_xy.plot([x[0]], [y[0]], 'o', color=colors['body'], markersize=3, zorder=1)
    motors_xy,  = axis_xy.plot(mw[:,0], mw[:,1], 'o', color=colors['motors'], markersize=7, zorder=1)
    trail_xy,   = axis_xy.plot([], [], color=colors['trail'], linewidth=1, alpha=0.5, linestyle='--', zorder=1)
    if xyz_ref is not None:
        xy_target,  = axis_xy.plot([xyz_ref[0,0]], [xyz_ref[0,1]], 'x', color=colors['target'], markersize=8, zorder=0)

    # X-Z
    arm_x_xz,  = axis_xz.plot(*zip(mw[0,[0,2]], mw[2,[0,2]]), color=colors['arms'], linewidth=3, zorder=1)
    arm_y_xz,  = axis_xz.plot(*zip(mw[1,[0,2]], mw[3,[0,2]]), color=colors['arms'], linewidth=3, zorder=1)
    center_xz, = axis_xz.plot([x[0]], [z[0]], 'o', color=colors['body'], markersize=3, zorder=1)
    motors_xz, = axis_xz.plot(mw[:,0], mw[:,2], 'o', color=colors['motors'], markersize=7, zorder=1)
    trail_xz,  = axis_xz.plot([], [], color=colors['trail'], linewidth=1, alpha=0.5, linestyle='--', zorder=1)
    if xyz_ref is not None:
        xz_target, = axis_xz.plot([xyz_ref[0,0]], [xyz_ref[0,2]], 'x', color=colors['target'], markersize=8, zorder=0)

    # Y-Z
    arm_x_yz,  = axis_yz.plot(*zip(mw[0,[1,2]], mw[2,[1,2]]), color=colors['arms'], linewidth=3, zorder=1)
    arm_y_yz,  = axis_yz.plot(*zip(mw[1,[1,2]], mw[3,[1,2]]), color=colors['arms'], linewidth=3, zorder=1)
    center_yz, = axis_yz.plot([y[0]], [z[0]], 'o', color=colors['body'], markersize=3, zorder=1)
    motors_yz, = axis_yz.plot(mw[:,1], mw[:,2], 'o', color=colors['motors'], markersize=7, zorder=1)
    trail_yz,  = axis_yz.plot([], [], color=colors['trail'], linewidth=1, alpha=0.5, linestyle='--', zorder=1)
    if xyz_ref is not None:
        yz_target, = axis_yz.plot([xyz_ref[0,1]], [xyz_ref[0,2]], 'x', color=colors['target'], markersize=8, zorder=0)

    # Text line at the bottom
    text = fig.text(0.5, 0.075, '', ha='center', fontsize=10)

    # Update function
    def update(frame):
        mw = get_world_motors(frame)
        k  = frame

        # 3D
        arm_x.set_data_3d(*zip(mw[0], mw[2]))
        arm_y.set_data_3d(*zip(mw[1], mw[3]))
        center_dot.set_data_3d([x[k]], [y[k]], [z[k]])
        motors_dot.set_data_3d(mw[:,0], mw[:,1], mw[:,2])
        trail_3d.set_data_3d(x[:k+1], y[:k+1], z[:k+1])
        if xyz_ref is not None:
            xyz_target.set_data_3d([xyz_ref[k,0]], [xyz_ref[k,1]], [xyz_ref[k,2]])

        # X-Y
        arm_x_xy.set_data(*zip(mw[0,:2], mw[2,:2]))
        arm_y_xy.set_data(*zip(mw[1,:2], mw[3,:2]))
        center_xy.set_data([x[k]], [y[k]])
        motors_xy.set_data(mw[:,0], mw[:,1])
        trail_xy.set_data(x[:k+1], y[:k+1])
        if xyz_ref is not None:
            xy_target.set_data([xyz_ref[k,0]], [xyz_ref[k,1]])

        # X-Z
        arm_x_xz.set_data(*zip(mw[0,[0,2]], mw[2,[0,2]]))
        arm_y_xz.set_data(*zip(mw[1,[0,2]], mw[3,[0,2]]))
        center_xz.set_data([x[k]], [z[k]])
        motors_xz.set_data(mw[:,0], mw[:,2])
        trail_xz.set_data(x[:k+1], z[:k+1])
        if xyz_ref is not None:
            xz_target.set_data([xyz_ref[k,0]], [xyz_ref[k,2]])

        # Y-Z
        arm_x_yz.set_data(*zip(mw[0,[1,2]], mw[2,[1,2]]))
        arm_y_yz.set_data(*zip(mw[1,[1,2]], mw[3,[1,2]]))
        center_yz.set_data([y[k]], [z[k]])
        motors_yz.set_data(mw[:,1], mw[:,2])
        trail_yz.set_data(y[:k+1], z[:k+1])
        if xyz_ref is not None:
            yz_target.set_data([xyz_ref[k,1]], [xyz_ref[k,2]])
        # Text
        text.set_text(
            f"x={x[k]:.2f}  y={y[k]:.2f}  z={z[k]:.2f}  |  "
            f"roll={np.degrees(roll[k]):.1f}°  pitch={np.degrees(pitch[k]):.1f}°  yaw={np.degrees(yaw[k]):.1f}°"
        )

        return (arm_x, arm_y, center_dot, motors_dot, trail_3d,
                arm_x_xy, arm_y_xy, center_xy, motors_xy, trail_xy,
                arm_x_xz, arm_y_xz, center_xz, motors_xz, trail_xz,
                arm_x_yz, arm_y_yz, center_yz, motors_yz, trail_yz, text)

    return update


__all__ = [
    'AutoAnimation',
    'Pendulum', 'DCMotor', 'CartPendulum', 'MultiCartPendulum',
    'MultimassSpring', 'JohanssonTanks', 'BatchBioreactor',
    'CSTR', 'OilWell', 'pHNeutralization', 'Quadrotor',
]