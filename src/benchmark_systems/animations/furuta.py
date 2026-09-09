"""Furuta-pendulum animator."""

import matplotlib.pyplot as plt
import numpy as np

from ._core import _initialized, _matching_trajectories


def FurutaPendulum(axis_3d, axis_th=None, axis_om=None, axis_ph=None,
                   *, L1, L2,
                   theta1, theta2, omega1=None, omega2=None,
                   times=None, trail_len=60, fixed_view: tuple|None =None):
    """
    Furuta (rotary inverted pendulum) animation.

    Main 3D view shows:
    - Motor base (cylinder)
    - Arm 1 (horizontal, rotates around Z axis with theta1)
    - Arm 2 / pendulum (hangs from end of arm1, angle theta2 from downward
      vertical; theta2=pi means pointing upright)
    - Bob (mass at tip of arm2)
    - Fading trail of the bob's recent path

    Optional side panels (pass plt.Axes objects):
    - axis_th : theta1 and theta2 vs time
    - axis_om : omega1 and omega2 vs time
    - axis_ph : phase portrait theta2 vs omega2

    Parameters
    ----------
    axis_3d  : Axes3D  main 3-D axis
    axis_th  : plt.Axes or None
    axis_om  : plt.Axes or None
    axis_ph  : plt.Axes or None
    L1       : float   arm1 length (m)
    L2       : float   arm2/pendulum length (m)
    theta1   : np.ndarray (N,)  arm rotation around Z (rad)
    theta2   : np.ndarray (N,)  pendulum angle from downward vertical (rad)
    omega1   : np.ndarray (N,) optional
    omega2   : np.ndarray (N,) optional
    times    : np.ndarray (N,) optional simulation times for the side panels
    trail_len: int  number of past frames to show as trail (default 60)
    fixed_view: tuple or None  (elev, azim) for fixed camera view (default None)
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.pyplot as _plt

    if axis_om is not None and (omega1 is None or omega2 is None):
        raise ValueError("omega1 and omega2 are required when axis_om is provided")
    if axis_ph is not None and omega2 is None:
        raise ValueError("omega2 is required when axis_ph is provided")
    values = {"theta1": theta1, "theta2": theta2}
    if omega1 is not None:
        values["omega1"] = omega1
    if omega2 is not None:
        values["omega2"] = omega2
    if times is not None:
        values["times"] = times
    trajectories = _matching_trajectories(**values)
    theta1, theta2 = trajectories["theta1"], trajectories["theta2"]
    omega1 = trajectories.get("omega1")
    omega2 = trajectories.get("omega2")
    times = trajectories.get("times")

    ax = axis_3d
    N  = len(theta1)

    # ── zorders for every geometry ────────────────────────────────────────────
    ZORDER_BASE = 0
    ZORDER_SHAFT = 1
    ZORDER_ARM1 = 3
    ZORDER_ARM2 = 3
    ZORDER_ELBOW = 3
    ZORDER_BOB = 4
    ZORDER_TRAIL = 3

    # ── Static base elements ──────────────────────────────────────────────────
    R_BASE = L1*0.8; H_BASE = L2*1.2
    R_SHAFT = L1*0.2; H_SHAFT = L2*0.2

    def _cylinder(cx, cy, z0, z1, r, color='#999', alpha=0.45, n=32, zorder=0):
        th = np.linspace(0, 2*np.pi, n)
        xs = cx + r*np.cos(th); ys = cy + r*np.sin(th)
        for i in range(n-1):
            verts = [[(xs[i],ys[i],z0),(xs[i+1],ys[i+1],z0),
                      (xs[i+1],ys[i+1],z1),(xs[i],ys[i],z1)]]
            ax.add_collection3d(Poly3DCollection(
                verts, color=color, alpha=alpha, linewidth=0, zorder=zorder))
        for z in [z0, z1]:
            verts = [list(zip(xs, ys, np.full(n, z)))]
            ax.add_collection3d(Poly3DCollection(
                verts, color=color, alpha=alpha+0.10, linewidth=0, zorder=zorder))

    _cylinder(0, 0, -H_BASE, 0,       R_BASE,  color='#aaaaaa', alpha=0.45, zorder=ZORDER_BASE)
    _cylinder(0, 0,  0,      H_SHAFT, R_SHAFT, color='#666666', alpha=0.70, zorder=ZORDER_SHAFT)

    # ── Geometry helper ───────────────────────────────────────────────────────
    def _geom(th1, th2):
        """Return (pivot, elbow, bob) as (3,) arrays."""
        # Unit vector along arm1 direction (horizontal, rotates around Z)
        er    = np.array([np.cos(th1),  np.sin(th1), 0.0])
        # Unit vector of pendulum rotation (vertical, rotates around arm1 axis)
        et    = np.array([-np.sin(th1), np.cos(th1), 0.0])
        # Unit vector along pendulum direction (vertical, rotates around arm1 axis)
        ez    = np.array([0.0, 0.0, 1.0])

        pivot = np.array([0.0, 0.0, H_SHAFT])
        elbow = L1 * er + pivot
        pend  = np.sin(th2) * et - np.cos(th2) * ez
        bob   = elbow + L2 * pend
        return pivot, elbow, bob

    # Pre-compute full bob trajectory for the trail
    traj = np.array([_geom(theta1[k], theta2[k])[2] for k in range(N)])

    # Faint full trajectory trace
    ax.plot(traj[:,0], traj[:,1], traj[:,2],
            color='gray', lw=0.7, alpha=0.20, ls='--', zorder=ZORDER_TRAIL)

    # ── 3D animated artists ───────────────────────────────────────────────────
    piv0, elbow0, bob0 = _geom(theta1[0], theta2[0])

    arm1_ln, = ax.plot([piv0[0],elbow0[0]],[piv0[1],elbow0[1]],[piv0[2],elbow0[2]],
                        color='#2255aa', lw=5,
                        solid_capstyle='round', zorder=ZORDER_ARM1)
    arm2_ln, = ax.plot([elbow0[0],bob0[0]],[elbow0[1],bob0[1]],[elbow0[2],bob0[2]],
                        color='#333333', lw=3,
                        solid_capstyle='round', zorder=ZORDER_ARM2)
    elbow_dot = ax.scatter(*elbow0, color='#2255aa', s=70,  zorder=ZORDER_ELBOW, depthshade=False)
    bob_dot   = ax.scatter(*bob0,   color='#cc3333', s=180, zorder=ZORDER_BOB, depthshade=False)
    trail_ln, = ax.plot([], [], [], color='#cc3333', lw=1.4, alpha=0.65, zorder=ZORDER_TRAIL)

    # 3D axis limits and style
    margin = L1 + L2 + 0.12
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_zlim(-H_BASE - 0.05, L2 + 0.20)
    ax.set_xlabel('X (m)', fontsize=8)
    ax.set_ylabel('Y (m)', fontsize=8)
    ax.set_zlabel('Z (m)', fontsize=8)
    ax.view_init(elev=22, azim=38)
    try:
        ax.set_box_aspect([1, 1, 0.70])
    except AttributeError:
        pass   # older matplotlib

    # ── Optional side panels ──────────────────────────────────────────────────
    panel_times = np.arange(N) if times is None else times

    panel_artists = {}

    if axis_th is not None:
        axis_th.tick_params(labelsize=7)
        axis_th.grid(True, alpha=0.3)
        axis_th.set_xlabel('time' if times is not None else 'frame', fontsize=8)
        axis_th.set_ylabel('angle (rad)', fontsize=8)
        axis_th.set_title('θ₁ (blue)  θ₂ (red)', fontsize=8)
        axis_th.plot(panel_times, theta1, color='#2255aa', lw=0.8, alpha=0.22)
        axis_th.plot(panel_times, theta2, color='#cc3333', lw=0.8, alpha=0.22)
        th1_ln, = axis_th.plot([], [], color='#2255aa', lw=1.5)
        th2_ln, = axis_th.plot([], [], color='#cc3333', lw=1.5)
        axis_th.set_xlim(panel_times[0], panel_times[-1]); axis_th.set_ylim(
            min(theta1.min(), theta2.min())-0.2,
            max(theta1.max(), theta2.max())+0.2)
        panel_artists['th'] = (th1_ln, th2_ln)

    if axis_om is not None:
        axis_om.tick_params(labelsize=7)
        axis_om.grid(True, alpha=0.3)
        axis_om.set_xlabel('time' if times is not None else 'frame', fontsize=8)
        axis_om.set_ylabel('ω (rad/s)', fontsize=8)
        axis_om.set_title('ω₁ (blue)  ω₂ (red)', fontsize=8)
        axis_om.plot(panel_times, omega1, color='#2255aa', lw=0.8, alpha=0.22)
        axis_om.plot(panel_times, omega2, color='#cc3333', lw=0.8, alpha=0.22)
        om1_ln, = axis_om.plot([], [], color='#2255aa', lw=1.5)
        om2_ln, = axis_om.plot([], [], color='#cc3333', lw=1.5)
        axis_om.set_xlim(panel_times[0], panel_times[-1]); axis_om.set_ylim(
            min(omega1.min(), omega2.min())-0.3,
            max(omega1.max(), omega2.max())+0.3)
        panel_artists['om'] = (om1_ln, om2_ln)

    if axis_ph is not None:
        axis_ph.tick_params(labelsize=7)
        axis_ph.grid(True, alpha=0.3)
        axis_ph.set_xlabel('θ₂ (rad)', fontsize=8)
        axis_ph.set_ylabel('ω₂ (rad/s)', fontsize=8)
        axis_ph.set_title('Phase portrait (pendulum)', fontsize=8)
        axis_ph.plot(theta2, omega2, color='#554488', lw=0.8, alpha=0.22)
        ph_ln,  = axis_ph.plot([], [], color='#554488', lw=1.2)
        ph_dot, = axis_ph.plot([], [], marker='o', color='#cc3333',
                                ms=7, zorder=5)
        axis_ph.set_xlim(theta2.min()-0.1, theta2.max()+0.1)
        axis_ph.set_ylim(omega2.min()-0.3, omega2.max()+0.3)
        panel_artists['ph'] = (ph_ln, ph_dot)

    # ── Update function ───────────────────────────────────────────────────────
    def update(frame):
        piv, elbow, bob = _geom(theta1[frame], theta2[frame])
        if fixed_view is not None:
            ax.view_init(elev=fixed_view[0], azim=fixed_view[1])

        # 3D arm1
        arm1_ln.set_data_3d([piv[0], elbow[0]],
                             [piv[1], elbow[1]],
                             [piv[2], elbow[2]])
        # 3D arm2
        arm2_ln.set_data_3d([elbow[0], bob[0]],
                             [elbow[1], bob[1]],
                             [elbow[2], bob[2]])
        # dots
        elbow_dot._offsets3d = ([elbow[0]], [elbow[1]], [elbow[2]])
        bob_dot._offsets3d   = ([bob[0]],   [bob[1]],   [bob[2]])

        # trail
        ts = max(0, frame - trail_len)
        trail_ln.set_data_3d(traj[ts:frame+1, 0],
                              traj[ts:frame+1, 1],
                              traj[ts:frame+1, 2])

        # 3D title
        title = ax.set_title(
            f'θ₁={np.degrees(theta1[frame]):.1f}°  '
            f'θ₂={np.degrees(theta2[frame]):.1f}°',
            fontsize=9)

        # Side panels
        if 'th' in panel_artists:
            th1_ln, th2_ln = panel_artists['th']
            th1_ln.set_data(panel_times[:frame+1], theta1[:frame+1])
            th2_ln.set_data(panel_times[:frame+1], theta2[:frame+1])

        if 'om' in panel_artists:
            om1_ln, om2_ln = panel_artists['om']
            om1_ln.set_data(panel_times[:frame+1], omega1[:frame+1])
            om2_ln.set_data(panel_times[:frame+1], omega2[:frame+1])

        if 'ph' in panel_artists:
            ph_ln, ph_dot = panel_artists['ph']
            ph_ln.set_data(theta2[:frame+1], omega2[:frame+1])
            ph_dot.set_data([theta2[frame]], [omega2[frame]])

        all_artists = [arm1_ln, arm2_ln, elbow_dot, bob_dot, trail_ln, title]
        for v in panel_artists.values():
            all_artists.extend(v)
        return all_artists

    return _initialized(update)

__all__ = ['FurutaPendulum']
