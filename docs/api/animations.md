# Animation helpers

Import from `benchmark_systems.animations`; this module requires the `anim`
extra. The helpers visualize complete trajectories produced by a simulation.
They do not advance a model or compute control inputs. See the
[animation guide](../animations.md) for display and export examples.

## Shared data conventions

Each scene factory draws its initial state and returns a callable
`update(frame)`. `frame` is a zero-based integer sample index. Calling it
updates the existing Matplotlib artists and returns the modified artists,
which allows the callback to be passed to `AutoAnimation`.

Scalar trajectories must be finite, nonempty one-dimensional arrays with
the same sample count. Optional time-varying inputs and annotations follow
the same rule: to display a constant, pass a repeated array, not a scalar.
The quadrotor uses component-major arrays with shape `(3, n_samples)`.
Pass angles in radians, even when a drawing labels them in degrees. Supply
other quantities in the model's engineering units; the drawings do not
reconcile inconsistent physical scales.

Several factories derive axis limits or display scales from the full
trajectory. Explicit scale options change the drawing, not the physical
model. A factory's displayed dimensions need not be a geometrically exact
representation of every model parameter.

## Animation scheduling

`AutoAnimation` returns `(animation, frames)`, where `animation` is a
Matplotlib `FuncAnimation` and `frames` is the tuple of displayed sample
indices. Keep a reference to `animation` while displaying or saving it.

Prefer `times=solution.t` to preserve simulation sample labels. `speed` is
an integer sample stride; `fps` controls playback rate. `times` determines
the label at each displayed sample, while the playback interval is constant:
an irregular time grid is not replayed with variable wall-clock delays.
`time_unit` changes the displayed suffix; supply already-scaled time values
when using minutes or hours.

```{eval-rst}
.. autofunction:: benchmark_systems.animations.AutoAnimation
```

## Mechanical and electrical scenes

### Pendulum

Draw a rod and bob at angle `theta` around `pivot=(x, y)`. Supply a
one-dimensional `theta` trajectory and drawing length `L`. Zero angle hangs
downward. The factory returns an initialized `update(frame)` callback.

```{figure} ../_static/models/pendulum.png
:alt: Pendulum rod of length L with a point mass m, angle theta, applied torque u and gravity.
:width: 137px
:align: center

Torque-driven point-mass pendulum.
```

```{eval-rst}
.. autofunction:: benchmark_systems.animations.Pendulum
```

### DC motor

Display a rotating shaft, angular velocity and armature current. `theta`,
`omega` and `current` are matching scalar trajectories, normally rows
0, 1 and 2 of the motor solution. Velocity and current indicator scales are
computed from their maximum absolute values over the supplied experiment.

```{figure} ../_static/models/dc-motor.png
:alt: DC motor shaft at 45 degrees with angular-velocity and armature-current indicators.
:width: 580px
:align: center

Shaft position, speed and armature-current indicators from the `DCMotor` animation helper.
```

```{eval-rst}
.. autofunction:: benchmark_systems.animations.DCMotor
```

### Cart pendulum

Draw a cart at `pos` with pendulum angle `theta` and drawing length `L`.
For `cart_pendulum`, use position row 0 and angle row 2. Optional `obstacles`
is an iterable of `((x, y), radius)` pairs. Obstacles are visual annotations;
the helper does not add collision dynamics to the simulation.

```{figure} ../_static/models/cart-pendulum.png
:alt: Force-driven cart carrying a point-mass pendulum, with position, velocity and angle coordinates.
:width: 188px
:align: center

Cart and point-mass pendulum with horizontal force input.
```

```{eval-rst}
.. autofunction:: benchmark_systems.animations.CartPendulum
```

### Multilevel cart pendulum

Draw a serial chain using the cart trajectory `pos`, link lengths `L` and
one scalar angle trajectory per link in `thetas`. Angles are absolute
orientations measured from the downward vertical, not angles relative to
the preceding link. Use this helper for both the double and multilevel
cart-pendulum models. `L` must contain one positive finite length per link.

```{figure} ../_static/models/multilevel-cart-pendulum.png
:alt: Three serial pendulum links on a cart, showing link lengths, masses, angles and joint friction.
:width: 288px
:align: center

Three-link example of the general multilevel cart-pendulum arrangement.
```

```{eval-rst}
.. autofunction:: benchmark_systems.animations.MultiCartPendulum
```

### Furuta pendulum

Provide a 3D axis plus `theta1`, `theta2` trajectories and drawing lengths
`L1`, `L2`. Optional time plots use `axis_th` and `axis_om`; `axis_ph` draws a
phase portrait. The velocity panel requires both `omega1` and `omega2`,
and the phase panel requires `omega2`. Optional `times` supplies the sample
coordinates for the side panels. `trail_len` sets the number of recent
samples in the bob trail; `fixed_view=(elevation, azimuth)` fixes the camera.

```{figure} ../_static/models/furuta-pendulum.png
:alt: Rotary motor arm carrying a pendulum, with two joint angles, centres of mass and motor torque K times u.
:width: 460px
:align: center

Furuta pendulum geometry, centre-of-mass distances and motor torque.
```

```{eval-rst}
.. autofunction:: benchmark_systems.animations.FurutaPendulum
```

### Multimass spring system

Provide `thetas` as one trajectory per disc and `u_true` as the two actual
motor-position trajectories. In a model result, these are the disc-position
block `y[2:2+N]` and motor block `y[:2]`, respectively. Supply `N+1` spring
constants in `K`; the helper draws the arrangement rather than evaluating
spring forces. Do not substitute motor commands for the filtered motor states
when comparing actuator dynamics.

```{figure} ../_static/models/multimass-spring.png
:alt: A chain of rotating discs coupled by springs, driven by stepper motors at both ends.
:width: 620px
:align: center

Spring-coupled rotating discs between two position-controlled motors.
```

```{eval-rst}
.. autofunction:: benchmark_systems.animations.MultimassSpring
```

## Process scenes

### Johansson tanks

Provide four height trajectories, two pump-command trajectories and two
valve-fraction trajectories, with four positive `h_max` values. Filled
fractions use those maximum heights. The displayed pipe routing follows the
same crossed upper-tank feed convention as the model.

Pump labels show the supplied commands in percent. Branch labels show
`(1 - gamma) * u` and `gamma * u` on the same percentage scale.

```{figure} ../_static/models/johansson.png
:alt: Four interconnected water tanks supplied by two pumps and two splitting valves, with upper tanks draining into lower tanks.
:width: 460px
:align: center

Four-tank process with crossed upper-tank pump feeds.
```

```{eval-rst}
.. autofunction:: benchmark_systems.animations.JohanssonTanks
```

### Batch bioreactor

Required trajectories are `X`, `S`, `P`, `V`. Optional `u` and `S_in` are
feed-flow and feed-concentration annotations and must be matching
trajectories when supplied. `V_max`, `X_max`, `S_max` and `P_max` override
the default scales derived from the experiment's maxima. Supply positive
scales suitable for the source units.

```{figure} ../_static/models/batch-bioreactor.png
:alt: Stirred bioreactor with a feed stream, biomass, substrate and product concentrations, and variable liquid volume.
:width: 266px
:align: center

Fed batch bioreactor with feed-flow input and variable liquid volume.
```

```{eval-rst}
.. autofunction:: benchmark_systems.animations.BatchBioreactor
```

### Continuous stirred-tank reactor

Provide concentrations `Ca`, `Cb` and temperatures `Tr`, `Tk`. Optional
`dQ`, `F`, `T_in` and `Ca_in` are matching annotation trajectories.
`Ca_max` and `Cb_max` set concentration scales; `Tr_min`, `Tr_max`,
`Tk_min`, `Tk_max` define temperature gauges; `dQ_max_abs` sets the heat-flow
colour scale. Pass ordered, nondegenerate ranges and a positive heat scale.
The drawing uses the numeric data as supplied and does not resolve the
[CSTR unit-scaling conventions](../models.md#continuous-stirred-tank-reactor).

```{figure} ../_static/models/cstr.png
:alt: Stirred tank with feed and outlet streams, species A and B, and a cooling jacket exchanging heat with the reactor.
:width: 410px
:align: center

Continuous stirred-tank reactor and cooling jacket.
```

```{eval-rst}
.. autofunction:: benchmark_systems.animations.CSTR
```

### Gas-lift oil well

Required trajectories are differential masses `x0`, `x1`, `x2`, tubing-top
pressure `P_tt`, production flow `w_out` and choke openings `u1`, `u2`.
Use the [oil-well state table](../models.md#gas-lift-oil-well) to select the
appropriate rows from a DAE result; `u1` means production and `u2` gas lift.

Optional trajectories show `w_G_in`, `P_at`, `P_ab`, `P_bh`, mixture density
`rho_m` and liquid fraction `alpha_L`. Note the annotation argument names:
the latter two correspond to model states `rho_m_tt` and `alpha_L_tt`.
Pressure inputs to the drawing are in bar. `w_G_in_max` and `w_out_max`
override the respective display scales. Bubbles are visual flow indicators,
not tracked material particles or additional model states.

```{figure} ../_static/models/oil-well.png
:alt: Gas-lift oil well showing annulus, tubing, injection valve, production and lift chokes, pressures and mixture flows.
:width: 621px
:align: center

Gas-lift well, manipulated choke openings and measured process quantities.
Source: [Jahanshahi et al., 2012](https://doi.org/10.3182/20120710-4-SG-2026.00110).
```

```{eval-rst}
.. autofunction:: benchmark_systems.animations.OilWell
```

### pH neutralization

Provide liquid-height `h` and algebraic `pH` trajectories, normally result
rows 2 and 3. `h_max` is a positive fill-height scale. Optional flow
annotations `q1`, `q2`, `q3` correspond to acid feed, buffer disturbance and
alkaline control. Repeat a constant flow over the sample grid before passing
it as an annotation. Colour represents pH; the helper does not recompute the
chemical constraint.

```{figure} ../_static/models/neutralization.png
:alt: Neutralization reactor with acid, buffer and alkaline inlet streams, liquid-height measurement and an outlet pH sensor.
:width: 560px
:align: center

Neutralization process with acid feed, buffer disturbance and controlled alkaline flow.
```

```{eval-rst}
.. autofunction:: benchmark_systems.animations.pHNeutralization
```

## Quadrotor scene

Provide a figure, one 3D axis and XY, XZ, YZ projection axes. Arrays `xyz`,
`angles` and optional `xyz_ref` have shape `(3, n_samples)`; `angles` is
ordered roll, pitch, yaw. For either quadrotor model, positions and Euler
angles are `y[:3]` and `y[3:6]`. The reference marks desired position.

`L` sets the drawing arm length, independently of the dynamics parameters.
`fixed_view=(elevation, azimuth)` fixes the 3D camera. Axis ranges are derived
from the supplied position trajectory, and all input arrays must have the
same sample count.

```{figure} ../_static/models/quadrotor.png
:alt: Four-rotor aircraft showing inertial and body axes, rotor forces, speeds and reaction torques.
:width: 426px
:align: center

Quadrotor geometry and reference frames used by both formulations.
Source: [MathWorks, quadrotor dynamics for nonlinear MPC](https://es.mathworks.com/help/symbolic/derive-quadrotor-dynamics-for-nonlinearMPC.html).
```

```{eval-rst}
.. autofunction:: benchmark_systems.animations.Quadrotor
```
