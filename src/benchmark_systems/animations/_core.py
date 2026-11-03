from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, Literal, Optional

import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist


def _trajectory(name: str, values) -> np.ndarray:
    """Validate and convert a scalar trajectory to a NumPy array.

    Parameters
    ----------
    name : str
        Name used to identify the trajectory in validation errors.
    values : array-like
        Values expected to form a finite, non-empty one-dimensional array.

    Returns
    -------
    np.ndarray
        Validated floating-point trajectory.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional trajectory")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} values must be finite")
    return array


def _matching_trajectories(**values) -> dict[str, np.ndarray]:
    """Validate scalar trajectories and require equal lengths.

    Parameters
    ----------
    **values : array-like
        Named scalar trajectories to validate.

    Returns
    -------
    dict[str, np.ndarray]
        Validated trajectories indexed by their input names.
    """
    arrays = {name: _trajectory(name, value) for name, value in values.items()}
    if len({array.size for array in arrays.values()}) > 1:
        raise ValueError("all trajectories must have the same length")
    return arrays


def _vector_trajectory(name: str, values) -> np.ndarray:
    """Validate and convert a three-dimensional vector trajectory.

    Parameters
    ----------
    name : str
        Name used to identify the trajectory in validation errors.
    values : array-like
        Values with shape ``(3, n_frames)``.

    Returns
    -------
    np.ndarray
        Validated floating-point trajectory.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] != 3 or array.shape[1] == 0:
        raise ValueError(f"{name} must have shape (3, n_frames)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} values must be finite")
    return array


def _initialized(update: Callable[[int], Sequence[Artist]]) -> Callable:
    """Initialize an animation update callback at its first frame.

    Parameters
    ----------
    update : callable
        Frame callback accepting an integer index and returning artists.

    Returns
    -------
    callable
        The same callback after evaluating frame zero.
    """
    update(0)
    return update


def _artists_tuple(artists) -> tuple[Artist, ...]:
    """Normalize an animation callback result to a tuple of artists.

    Parameters
    ----------
    artists : Artist, iterable of Artist, or None
        Result returned by an animation update callback.

    Returns
    -------
    tuple[Artist, ...]
        Normalized artist collection.

    Raises
    ------
    TypeError
        If the result is not an artist or an iterable containing only artists.
    """
    if artists is None:
        return ()
    if isinstance(artists, Artist):
        return (artists,)
    if not isinstance(artists, Iterable):
        raise TypeError("animation update functions must return Matplotlib artists")
    result = tuple(artists)
    if not all(isinstance(artist, Artist) for artist in result):
        raise TypeError("animation update functions must return only Matplotlib artists")
    return result


def AutoAnimation(
    fig, update_fn: Callable, duration: float | None = None,
    dt: float | None = None, speed: int = 1, fps: Optional[float] = None,
    time_label=True, time_unit: Literal['s', 'min', 'hr'] = 's',
    time_fmt='.1f', *, times: Sequence[float] | None = None, **kwargs,
):
    """Create a Matplotlib animation from a sampled-scene callback.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure containing the artists to animate.
    update_fn : callable
        Callback ``update_fn(frame)`` accepting a zero-based sample index. Return
        a Matplotlib artist, an iterable of artists, or None.
    duration : float or None, default=None
        Positive experiment duration for the legacy grid, used only when times
        is None. Supply it in the same units as dt.
    dt : float or None, default=None
        Positive sample spacing for the legacy grid. Together with duration it
        generates ``round(duration / dt)`` samples starting at zero; the duration
        endpoint is not included. Ignored when times is supplied.
    speed : int, default=1
        Positive sample stride per animation frame. The last available sample
        is included even when the stride would skip it.
    fps : float or None, default=None
        Positive playback frame rate in frames/s. None uses the inverse sample
        spacing, or inverse median spacing for times, capped at 60 frames/s.
        A single explicitly timed sample uses 1 frame/s by default.
    time_label : bool or matplotlib.text.Text, default=True
        True creates a label at the bottom of the figure, False disables it,
        and a Text instance reuses that text artist.
    time_unit : {'s', 'min', 'hr'}, default='s'
        Time-label suffix only. Supply already-scaled time values; no conversion
        is performed.
    time_fmt : str, default='.1f'
        Python format specification for each displayed time value.
    times : array_like of shape (N,) or None, default=None
        Explicit finite, nonempty, strictly increasing sample times. The callback
        must support those N sample indices. None requires duration and dt.
    **kwargs : dict
        Options forwarded to matplotlib.animation.FuncAnimation, such as blit
        and repeat. Repeat defaults to False unless explicitly supplied.

    Returns
    -------
    animation : matplotlib.animation.FuncAnimation
        Configured animation. Keep a reference while displaying or saving it.
    frames : tuple of int
        Selected zero-based sample indices, including the last available sample.

    Raises
    ------
    ValueError
        If time values, timing arguments, speed, fps or time_unit are invalid.
    TypeError
        If a callback result contains objects other than Matplotlib artists.

    Notes
    -----
    Playback uses a constant wall-clock frame interval. Nonuniform sample times
    change the labels but do not create nonuniform playback delays. Automatic
    fps uses the numeric time spacing as seconds; set fps explicitly when using
    another time scale. Neither speed nor fps changes the simulated trajectory.
    Reserve a bottom figure margin when using the default time label.
    """
    if isinstance(speed, bool) or not isinstance(speed, (int, np.integer)) or speed < 1:
        raise ValueError("speed must be an integer greater than or equal to 1")
    if time_unit not in ('s', 'min', 'hr'):
        raise ValueError("time_unit must be 's', 'min', or 'hr'")
    if times is None:
        if duration is None or dt is None:
            raise ValueError("provide times or both duration and dt")
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be a positive finite value")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a positive finite value")
        points = int(round(duration / dt))
        if points < 1:
            raise ValueError("duration and dt must define at least one frame")
        simulation_times = np.arange(points, dtype=float) * dt
        sample_dt = float(dt)
    else:
        simulation_times = np.asarray(times, dtype=float)
        if simulation_times.ndim != 1 or simulation_times.size == 0:
            raise ValueError("times must be a non-empty one-dimensional sequence")
        if not np.isfinite(simulation_times).all():
            raise ValueError("times values must be finite")
        if simulation_times.size > 1 and np.any(np.diff(simulation_times) <= 0):
            raise ValueError("times values must be strictly increasing")
        points = simulation_times.size
        sample_dt = float(np.median(np.diff(simulation_times))) if points > 1 else 1.0
    if fps is not None and (not np.isfinite(fps) or fps <= 0):
        raise ValueError("fps must be a positive finite value")
    playback_fps = float(fps) if fps is not None else min(60.0, 1.0 / sample_dt)
    frames = list(range(0, points, int(speed)))
    if frames[-1] != points - 1:
        frames.append(points - 1)
    frames = tuple(frames)
    if time_label is True:
        time_text = fig.text(
            0.5, 0.01, f't = 0{time_unit}', ha='center', va='bottom',
            fontsize=10, color='#333333',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2),
        )
    elif time_label is False:
        time_text = None
    else:
        time_text = time_label

    def wrapped_update(frame):
        artists = _artists_tuple(update_fn(frame))
        if time_text is not None:
            time_text.set_text(f't = {simulation_times[frame]:{time_fmt}} {time_unit}')
            return artists + (time_text,)
        return artists

    animation = FuncAnimation(
        fig, wrapped_update, frames=frames, repeat=kwargs.pop('repeat', False),
        interval=1000.0 / playback_fps, **kwargs,
    )
    return animation, frames
