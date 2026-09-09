"""Mechanical-system animators."""

from functools import partial

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ._core import _initialized, _matching_trajectories, _trajectory


def Pendulum(axis, *, L, theta, pivot=(0.0, 0.0)):
    theta = _trajectory("theta", theta)
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

    return _initialized(update)


def DCMotor(axis, *, theta, omega, current):
    trajectories = _matching_trajectories(theta=theta, omega=omega, current=current)
    theta, omega, current = trajectories.values()
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

    return _initialized(update)


def CartPendulum(axis, *, L, pos, theta, obstacles=()):
    trajectories = _matching_trajectories(pos=pos, theta=theta)
    pos, theta = trajectories.values()
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

    return _initialized(partial(
        _update, _bob=bob, _cart=cart, _line=line, _pos=pos, _theta=theta
    ))


def MultiCartPendulum(axis, *, L, pos, thetas):
    L = np.asarray(L, dtype=float)
    if L.ndim != 1 or L.size == 0 or not np.isfinite(L).all() or np.any(L <= 0):
        raise ValueError("L must contain positive finite pendulum lengths")
    if len(thetas) != L.size:
        raise ValueError("L and thetas must describe the same number of pendulums")
    trajectories = _matching_trajectories(
        pos=pos, **{f"theta_{i}": theta for i, theta in enumerate(thetas)}
    )
    pos = trajectories.pop("pos")
    thetas = tuple(trajectories.values())
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

    return _initialized(update)


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
    if n == 0:
        raise ValueError("thetas must contain at least one disc trajectory")
    if len(K) != n + 1:
        raise ValueError("K must contain one more spring constant than thetas")
    if len(u_true) != 2:
        raise ValueError("u_true must contain the two motor trajectories")
    trajectories = _matching_trajectories(
        **{f"theta_{i}": theta for i, theta in enumerate(thetas)},
        motor_0=u_true[0], motor_1=u_true[1],
    )
    thetas = tuple(trajectories[f"theta_{i}"] for i in range(n))
    u_true = (trajectories["motor_0"], trajectories["motor_1"])

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
        txt = axis.text(disc_cx[i], shaft_y - ry*1.2, f'θ{i+1}={np.degrees(thetas[i][0]):.1f}°',
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
        txt = axis.text(fx, shaft_y - motor_ry*1.2, f'θM{i+1}={np.degrees(u_true[i][0]):.1f}°',
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
        return tuple(marks) + tuple(motor_dots) + tuple(ang_pos_txt)

    return _initialized(update)

__all__ = ['Pendulum', 'DCMotor', 'CartPendulum', 'MultiCartPendulum', 'MultimassSpring']
