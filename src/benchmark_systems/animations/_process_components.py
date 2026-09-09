"""Stateful drawing components shared by process animations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Ellipse, Patch
from matplotlib.text import Text


@dataclass
class VerticalBar:
    """Animated vertical value bar and its numeric label.

    Parameters
    ----------
    fill : matplotlib.patches.Patch
        Patch whose height represents the current value.
    value_text : matplotlib.text.Text
        Text artist displaying the current value.
    height : float
        Available display height for the fill patch.
    maximum : float
        Value represented by a completely filled bar.
    minimum_height : float, default=0.01
        Smallest visible fill height.
    """

    fill: Patch
    value_text: Text
    height: float
    maximum: float
    minimum_height: float = 0.01

    @property
    def artists(self) -> tuple[Artist, ...]:
        """Return the artists modified by this component."""
        return self.fill, self.value_text

    def update(self, value: float, *, text: str | None = None) -> tuple[Artist, ...]:
        """Update the fill height and displayed value.

        Parameters
        ----------
        value : float
            Current bar value, clipped to the display range.
        text : str, optional
            Explicit label text. By default, ``value`` is shown with two
            decimal places.

        Returns
        -------
        tuple[Artist, ...]
            Artists modified by the update.
        """
        fraction = np.clip(value / self.maximum, 0.0, 1.0)
        self.fill.set_height(max((self.height - 0.02) * fraction, self.minimum_height))
        self.value_text.set_text(f"{value:.2f}" if text is None else text)
        return self.artists


@dataclass
class Mixer:
    """Animated mixer composed of a shaft, blades, hub, and depth ellipse.

    Parameters
    ----------
    shaft : matplotlib.lines.Line2D
        Vertical mixer shaft.
    blades : sequence of matplotlib.lines.Line2D
        Opposing rotating blade artists.
    hub : matplotlib.patches.Circle
        Blade hub artist.
    depth : matplotlib.patches.Ellipse
        Ellipse providing a visual depth cue below the blades.
    center_x : float
        Horizontal mixer position.
    motor_top : float
        Vertical position where the shaft meets the motor.
    blade_radius : float
        Display radius of each blade.
    """

    shaft: Line2D
    blades: Sequence[Line2D]
    hub: Circle
    depth: Ellipse
    center_x: float
    motor_top: float
    blade_radius: float

    @property
    def artists(self) -> tuple[Artist, ...]:
        """Return the artists modified by this component."""
        return self.shaft, *self.blades, self.hub, self.depth

    def update(self, frame: int, blade_y: float) -> tuple[Artist, ...]:
        """Move and rotate the mixer for one animation frame.

        Parameters
        ----------
        frame : int
            Frame index used to determine blade rotation.
        blade_y : float
            Current vertical position of the blade hub.

        Returns
        -------
        tuple[Artist, ...]
            Artists modified by the update.
        """
        angle = np.radians(frame * 6)
        self.hub.set_center((self.center_x, blade_y))
        self.depth.set_center((self.center_x, blade_y - 0.07))
        self.shaft.set_data([self.center_x, self.center_x], [self.motor_top, blade_y])
        for offset, blade in zip((0.0, np.pi), self.blades):
            blade_angle = angle + offset
            x = self.center_x + self.blade_radius * np.cos(blade_angle)
            y = blade_y + self.blade_radius * 0.28 * np.sin(blade_angle) - 0.06
            blade.set_data([self.center_x, x], [blade_y, y])
        return self.artists


@dataclass
class DialGauge:
    """Animated dial gauge with a value arc, needle, and numeric label.

    Parameters
    ----------
    arc : matplotlib.patches.Arc
        Colored arc representing the normalized value.
    needle : matplotlib.lines.Line2D
        Gauge needle artist.
    value_text : matplotlib.text.Text
        Numeric value label.
    center : tuple[float, float]
        Gauge center in data coordinates.
    radius : float
        Gauge display radius.
    limits : tuple[float, float]
        Lower and upper values represented by the gauge.
    """

    arc: Arc
    needle: Line2D
    value_text: Text
    center: tuple[float, float]
    radius: float
    limits: tuple[float, float]

    @property
    def artists(self) -> tuple[Artist, ...]:
        """Return the artists modified by this component."""
        return self.arc, self.needle, self.value_text

    def update(self, value: float, *, color=None, fmt: str = ".1f") -> float:
        """Update the gauge and return the normalized displayed value.

        Parameters
        ----------
        value : float
            Current gauge value.
        color : color-like, optional
            New edge color for the value arc.
        fmt : str, default='.1f'
            Format specification used by the numeric label.

        Returns
        -------
        float
            Value normalized to the closed interval ``[0, 1]``.
        """
        low, high = self.limits
        fraction = float(np.clip((value - low) / max(high - low, 1e-9), 0.0, 1.0))
        self.arc.theta1 = 210 - 240 * fraction
        self.arc.theta2 = 210
        if color is not None:
            self.arc.set_edgecolor(color)
        angle = np.radians(210 - 240 * fraction)
        x, y = self.center
        self.needle.set_data(
            [x, x + self.radius * 0.65 * np.cos(angle)],
            [y, y + self.radius * 0.65 * np.sin(angle)],
        )
        self.value_text.set_text(format(value, fmt))
        return fraction


@dataclass
class BubblePopulation:
    """Animated population of bubbles with fixed horizontal positions.

    Parameters
    ----------
    circles : sequence of matplotlib.patches.Circle
        Circle artists used to draw the bubbles.
    x : np.ndarray
        Horizontal position of each bubble.
    y_fraction : np.ndarray
        Relative vertical position of each bubble in the usable region.
    """

    circles: Sequence[Circle]
    x: np.ndarray
    y_fraction: np.ndarray

    @property
    def artists(self) -> tuple[Artist, ...]:
        """Return the bubble artists modified by this component."""
        return tuple(self.circles)

    def update(
        self,
        fraction: float,
        *,
        y_bottom: float,
        usable_height: float,
        radius_min: float = 0.04,
        radius_span: float = 0.08,
        alpha: float = 0.58,
    ) -> tuple[Artist, ...]:
        """Update bubble visibility, position, radius, and opacity.

        Parameters
        ----------
        fraction : float
            Population intensity, clipped to ``[0, 1]``.
        y_bottom : float
            Bottom of the usable vertical region.
        usable_height : float
            Height across which relative bubble positions are mapped.
        radius_min : float, default=0.04
            Bubble radius at zero intensity.
        radius_span : float, default=0.08
            Radius added at maximum intensity.
        alpha : float, default=0.58
            Opacity assigned to visible bubbles.

        Returns
        -------
        tuple[Artist, ...]
            Bubble artists modified by the update.
        """
        fraction = float(np.clip(fraction, 0.0, 1.0))
        visible = max(0, int(round(fraction * len(self.circles))))
        radius = radius_min + radius_span * fraction
        for index, circle in enumerate(self.circles):
            if index < visible:
                circle.set_center((self.x[index], y_bottom + self.y_fraction[index] * usable_height))
                circle.set_radius(radius)
                circle.set_alpha(alpha)
            else:
                circle.set_radius(0.0)
                circle.set_alpha(0.0)
        return self.artists
