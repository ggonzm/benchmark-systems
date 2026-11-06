import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.artist import Artist
import numpy as np
import pytest

import benchmark_systems.animations as animations
from benchmark_systems.animations._process_components import (
    BubblePopulation,
    DialGauge,
    Mixer,
    VerticalBar,
)
from benchmark_systems.animations import furuta, mechanical, processes, quadrotor


def _assert_artists(update, frame=2):
    artists = tuple(update(frame))
    assert artists
    assert all(isinstance(artist, Artist) for artist in artists)


def test_public_animation_exports():
    assert "FurutaPendulum" in animations.__all__
    assert set(animations.__all__) == {
        "AutoAnimation", "Pendulum", "DCMotor", "CartPendulum",
        "FurutaPendulum", "MultiCartPendulum", "MultimassSpring",
        "JohanssonTanks", "BatchBioreactor", "CSTR", "OilWell",
        "pHNeutralization", "Quadrotor",
    }


def test_animators_are_importable_from_split_modules():
    assert mechanical.Pendulum is animations.Pendulum
    assert furuta.FurutaPendulum is animations.FurutaPendulum
    assert processes.CSTR is animations.CSTR
    assert quadrotor.Quadrotor is animations.Quadrotor


def test_process_components_own_their_updates():
    fig, ax = plt.subplots()

    fill = ax.bar([0], [0])[0]
    value_text = ax.text(0, 0, "")
    bar = VerticalBar(fill, value_text, height=2.0, maximum=4.0)
    assert bar.update(2.0, text="half") == (fill, value_text)
    assert fill.get_height() == pytest.approx(0.99)
    assert value_text.get_text() == "half"

    arc = matplotlib.patches.Arc((0, 0), 1, 1)
    ax.add_patch(arc)
    needle, = ax.plot([], [])
    gauge_text = ax.text(0, 0, "")
    gauge = DialGauge(arc, needle, gauge_text, (0, 0), 1.0, (0.0, 10.0))
    assert gauge.update(5.0) == pytest.approx(0.5)
    assert gauge_text.get_text() == "5.0"

    shaft, = ax.plot([], [])
    blades = ax.plot([], [], [], [])
    hub = matplotlib.patches.Circle((0, 0), 0.1)
    depth = matplotlib.patches.Ellipse((0, 0), 1, 0.1)
    mixer = Mixer(shaft, blades, hub, depth, 1.0, 3.0, 0.5)
    assert mixer.update(0, 1.5) == mixer.artists
    assert hub.center == (1.0, 1.5)

    circles = [matplotlib.patches.Circle((0, 0), 0.1) for _ in range(2)]
    bubbles = BubblePopulation(circles, np.array([1.0, 2.0]), np.array([0.25, 0.75]))
    assert bubbles.update(0.5, y_bottom=1.0, usable_height=4.0) == tuple(circles)
    assert circles[0].center == (1.0, 2.0)
    assert circles[1].get_radius() == 0.0

    plt.close(fig)


def test_mechanical_animators_return_artists_at_last_frame():
    values = np.array([0.0, 0.1, 0.2])

    fig, ax = plt.subplots()
    _assert_artists(animations.Pendulum(ax, L=1.0, theta=values))
    plt.close(fig)

    fig, ax = plt.subplots()
    _assert_artists(animations.DCMotor(
        ax, theta=values, omega=values, current=values
    ))
    plt.close(fig)

    fig, ax = plt.subplots()
    _assert_artists(animations.CartPendulum(
        ax, L=1.0, pos=values, theta=values
    ))
    plt.close(fig)

    fig = plt.figure()
    ax3d = fig.add_subplot(221, projection="3d")
    ax_th, ax_om, ax_ph = (fig.add_subplot(code) for code in (222, 223, 224))
    _assert_artists(animations.FurutaPendulum(
        ax3d, ax_th, ax_om, ax_ph, L1=1.0, L2=1.0,
        theta1=values, theta2=values, omega1=values, omega2=values,
        times=np.array([0.0, 0.5, 1.0]),
    ))
    plt.close(fig)

    fig, ax = plt.subplots()
    _assert_artists(animations.MultiCartPendulum(
        ax, L=[1.0, 0.5], pos=values, thetas=[values, values]
    ))
    plt.close(fig)

    fig, ax = plt.subplots()
    _assert_artists(animations.MultimassSpring(
        ax, K=[1.0, 1.0, 1.0], thetas=[values, values],
        u_true=[values, values],
    ))
    plt.close(fig)


def test_johansson_displays_pump_commands_and_branch_shares_as_percentages():
    fig, ax = plt.subplots()
    update = animations.JohanssonTanks(
        ax, h_max=[10.0] * 4, heights=[[1.0, 1.0]] * 4,
        pumps=[[10.0, 20.0], [20.0, 40.0]],
        gammas=[[0.25, 0.5], [0.25, 0.75]],
    )
    for frame, expected in enumerate(([10, 7.5, 2.5, 20, 15, 5], [20, 10, 10, 40, 10, 30])):
        update(frame)
        labels = [text.get_text() for text in ax.texts if text.get_text().endswith("%")]
        assert labels == [f"{command:.1f}%" for command in expected]
    plt.close(fig)


def test_cstr_displays_volumetric_flow_and_jacket_power():
    fig, ax = plt.subplots()
    update = animations.CSTR(
        ax, Ca=[1.0, 1.0], Cb=[0.5, 0.5], Tr=[130.0, 130.0], Tk=[125.0, 125.0],
        F=[100.0, 120.0], dQ=[-3600.0, -7200.0],
    )
    assert [text.get_text() for text in ax.texts].count("100.00 L/h") == 2
    assert any(text.get_text() == "dQ=-1.0 kW" for text in ax.texts)
    update(1)
    assert [text.get_text() for text in ax.texts].count("120.00 L/h") == 2
    assert any(text.get_text() == "dQ=-2.0 kW" for text in ax.texts)
    plt.close(fig)


def test_process_animators_return_artists_at_last_frame():
    values = np.array([1.0, 1.5, 2.0])

    fig, ax = plt.subplots()
    _assert_artists(animations.JohanssonTanks(
        ax, h_max=[10.0] * 4, heights=[values] * 4,
        pumps=[values] * 2, gammas=[values / 4] * 2,
    ))
    plt.close(fig)

    fig, ax = plt.subplots()
    _assert_artists(animations.BatchBioreactor(
        ax, X=values, S=values, P=values, V=values,
        u=values, S_in=values,
    ))
    plt.close(fig)

    fig, ax = plt.subplots()
    _assert_artists(animations.CSTR(
        ax, Ca=values, Cb=values, Tr=values + 130, Tk=values + 125,
        dQ=-values, F=values, T_in=values + 120, Ca_in=values,
    ))
    plt.close(fig)

    fig, ax = plt.subplots()
    _assert_artists(animations.OilWell(
        ax, x0=values, x1=values, x2=values, P_tt=values + 100,
        w_out=values, u1=values / 2, u2=values / 2,
        w_G_in=values, P_at=values + 100, P_ab=values + 100,
        P_bh=values + 100, rho_m=values, alpha_L=values / 4,
    ))
    plt.close(fig)

    fig, ax = plt.subplots()
    _assert_artists(animations.pHNeutralization(
        ax, h=values, pH=values + 6, q1=values, q2=values, q3=values,
    ))
    plt.close(fig)


def test_quadrotor_returns_reference_artists():
    values = np.array([
        [0.0, 0.5, 1.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ])
    fig = plt.figure()
    ax3d = fig.add_subplot(221, projection="3d")
    ax_xy, ax_xz, ax_yz = (fig.add_subplot(code) for code in (222, 223, 224))
    update = animations.Quadrotor(
        fig, ax3d, ax_xy, ax_xz, ax_yz, L=0.2, xyz=values,
        angles=np.zeros_like(values), xyz_ref=values,
    )

    artists = tuple(update(2))

    assert all(isinstance(artist, Artist) for artist in artists)
    assert len(artists) == 25
    plt.close(fig)


def test_auto_animation_uses_explicit_times_and_supports_blitting():
    fig, ax = plt.subplots()
    line, = ax.plot([], [])

    def update(frame):
        line.set_data([0, frame], [0, frame])
        return line

    animation, frames = animations.AutoAnimation(
        fig, update, times=[0.0, 0.25, 1.0], speed=2, blit=True
    )

    assert frames == (0, 2)
    assert animation._interval == pytest.approx(500.0)
    assert all(isinstance(artist, Artist) for artist in animation._func(2))
    plt.close(fig)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"times": []}, "non-empty"),
        ({"times": [0.0, 0.0]}, "strictly increasing"),
        ({"duration": 1.0, "dt": 0.0}, "positive finite"),
        ({"times": [0.0], "speed": 1.5}, "speed"),
        ({"times": [0.0], "time_unit": "days"}, "time_unit"),
    ],
)
def test_auto_animation_rejects_invalid_configuration(kwargs, message):
    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    with pytest.raises(ValueError, match=message):
        animations.AutoAnimation(fig, lambda _: (line,), **kwargs)
    plt.close(fig)


def test_animators_reject_mismatched_trajectories():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="same length"):
        animations.DCMotor(
            ax, theta=[0.0, 1.0], omega=[0.0], current=[0.0, 1.0]
        )
    with pytest.raises(ValueError, match="shape"):
        animations.Quadrotor(
            fig, ax, ax, ax, ax, L=1.0,
            xyz=np.zeros((2, 3)), angles=np.zeros((3, 3)),
        )
    plt.close(fig)
