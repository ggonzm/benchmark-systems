"""Quadrotor animator."""

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D

from ._core import _initialized, _vector_trajectory


def Quadrotor(fig: plt.Figure, axis_3d: Axes3D, axis_xy: plt.Axes, axis_xz: plt.Axes, axis_yz: plt.Axes,
        *, L: float, xyz: np.ndarray, angles: np.ndarray, xyz_ref: np.ndarray | None = None, fixed_view: tuple | None = None) -> Callable:
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
    L : float
        Length of the arms of the quadrotor for visualization purposes, not necessarily related to the physical parameters of the system.
    xyz : np.ndarray, shape (3, n_frames)
        Component-major quadrotor positions.
    angles : np.ndarray, shape (3, n_frames)
        Component-major quadrotor angles (roll, pitch, yaw).
    xyz_ref : np.ndarray, optional
        Array with the reference position for the quadrotor (used for plotting a target point).
    fixed_view : tuple, optional
        If provided, sets a fixed view for the 3D plot (elevation, azimuth).
    '''
    xyz = _vector_trajectory("xyz", xyz)
    angles = _vector_trajectory("angles", angles)
    if xyz.shape[1] != angles.shape[1]:
        raise ValueError("xyz and angles must have the same number of frames")
    if xyz_ref is not None:
        xyz_ref = _vector_trajectory("xyz_ref", xyz_ref)
        if xyz_ref.shape[1] != xyz.shape[1]:
            raise ValueError("xyz_ref must have the same number of frames as xyz")
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
        if fixed_view is not None:
            axis_3d.view_init(elev=fixed_view[0], azim=fixed_view[1])

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

        artists = [arm_x, arm_y, center_dot, motors_dot, trail_3d,
                arm_x_xy, arm_y_xy, center_xy, motors_xy, trail_xy,
                arm_x_xz, arm_y_xz, center_xz, motors_xz, trail_xz,
                arm_x_yz, arm_y_yz, center_yz, motors_yz, trail_yz, text]
        if xyz_ref is not None:
            artists.extend([xyz_target, xy_target, xz_target, yz_target])
        return tuple(artists)

    return _initialized(update)

__all__ = ['Quadrotor']
