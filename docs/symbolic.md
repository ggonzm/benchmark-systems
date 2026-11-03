# Symbolic models and linearization

The symbolic API uses CasADi. The `sym` extra is required to build and
differentiate models; the plots and animation below also require the `anim`
extra. See [Installation](installation.md).

We will design an LQR controller for a quadrotor by constructing the
nonlinear model, linearizing around hover, computing a feedback gain and
simulating position tracking on the nonlinear plant.

## Construct a quadrotor model

A symbolic model function is a **factory**. Physical parameters are supplied once
to obtain a CasADi function with inputs `t`, `x`, `u`. Unlike the numerical
callable, the factory does not evaluate a derivative when it is constructed.

The ODE state is
`[x, y, z, phi, theta, psi, s, v, w, p, q, r]`: inertial position, Euler
angles, body-frame linear velocities and body-frame angular velocities.
Input contains four **squared rotor speeds**, not four forces or unsquared
rotor speeds. See the [model catalogue](models.md#quadrotor-ode-and-dae) for
coordinate conventions and units.

At hover, all four rotor inputs are equal and their total thrust balances
weight: $4k\omega_e^2=mg$. We will choose position `[0, 0, 5]` with zero angles and
velocities as the equilibrium state:

```{testcode} symbolic-quadrotor
import casadi as ca
import numpy as np
from benchmark_systems.symbolic import quadrotor_ode

quadrotor_parameters = {
    "Ixx": 1.2, "Iyy": 1.2, "Izz": 2.3,
    "k": 1.0, "L": 0.25, "m": 2.0, "drag": 0.2,
}
quadrotor_model = quadrotor_ode(**quadrotor_parameters)
x_equilibrium = np.array([0.0, 0.0, 5.0] + [0.0] * 9)
hover_squared_speed = (
    quadrotor_parameters["m"] * 9.81 / (4 * quadrotor_parameters["k"])
)
u_equilibrium = np.full(4, hover_squared_speed)

hover_derivative = quadrotor_model(
    t=0.0, x=x_equilibrium, u=u_equilibrium,
)["dx"]
print(hover_derivative.shape, hover_squared_speed)
```

```{testoutput} symbolic-quadrotor
(12, 1) 4.905
```

`Ixx`, `Iyy` and `Izz` set the body inertias; `m` and `L` set mass and arm
length. `k` converts squared rotor speed into thrust, while `drag` supplies
the rotor yaw-torque coefficient. These values are bound by
`quadrotor_ode(**quadrotor_parameters)` when it constructs the model.

`x_equilibrium` places the vehicle at a height of 5 m with zero attitude and
velocity. `np.full(4, hover_squared_speed)` gives every rotor the same hover
input. Named evaluation returns a dictionary, so `["dx"]` selects its
12-component derivative column. This numeric result is a CasADi `DM`;
symbolic arguments instead produce expressions for differentiation.

## Linearize around hover

The [CasADi naming conventions](api/symbolic.md#casadi-input-and-output-names)
define the input, output and Jacobian block names shared by the ODE models.

For $\dot{x}=f(t,x,u)$, the local state and input matrices are

$$ A=\left.\frac{\partial f}{\partial x}\right|_{t_e,x_e,u_e},\qquad
B=\left.\frac{\partial f}{\partial u}\right|_{t_e,x_e,u_e}. $$

CasADi's Jacobian function returns named blocks for the named model inputs
and output. We will evaluate them at the equilibrium:

```{testcode} symbolic-quadrotor
jacobian = quadrotor_model.jacobian()(
    t=0.0, x=x_equilibrium, u=u_equilibrium,
)
A = np.asarray(jacobian["jac_dx_x"], dtype=float)
B = np.asarray(jacobian["jac_dx_u"], dtype=float)
print(A.shape, B.shape)
```

```{testoutput} symbolic-quadrotor
(12, 12) (12, 4)
```

`jac_dx_x` is the derivative of the output named `dx` with respect to the
input named `x`; `jac_dx_u` differentiates it with respect to the four rotor
inputs. `np.asarray(..., dtype=float)` converts both blocks into NumPy
matrices for SciPy. Their shapes give the perturbation model
$\delta\dot{x}=A\delta x+B\delta u$, whose control increment is added to
the hover input rather than replacing it.

## Design an LQR controller

We will select the first six states—position and attitude—with
$C=[I_6\;0]$. Position errors will have weight 1 and attitude errors weight 100, with
an identity control-cost matrix. The state cost is $Q_x=C^TQC$:

$$
J=\int_0^\infty
\left(\delta x^TQ_x\delta x+\delta u^TR\delta u\right)\,dt.
$$

We will compute the feedback gain using SciPy's continuous-time algebraic Riccati
solver:

```{testcode} symbolic-quadrotor
from scipy.linalg import solve_continuous_are

C = np.zeros((6, 12))
C[:, :6] = np.eye(6)
Q = np.diag([1.0, 1.0, 1.0, 100.0, 100.0, 100.0])
R = np.eye(4)
Q_state = C.T @ Q @ C

P = solve_continuous_are(A, B, Q_state, R)
K = np.linalg.solve(R, B.T @ P)
print(K.shape)
```

```{testoutput} symbolic-quadrotor
(4, 12)
```

`C[:, :6] = np.eye(6)` selects position and attitude, and `C.T @ Q @ C`
embeds their six weights in the 12-state cost matrix. `R` weights the four
control increments equally. `solve_continuous_are` returns the Riccati
solution `P`; `np.linalg.solve(R, B.T @ P)` then computes the feedback gain
without explicitly forming an inverse. The resulting `K` maps 12 state
errors to four rotor-input corrections.

The cost selects six outputs, but the feedback still needs all 12 states.
If only position and attitude are measured, supply suitable state estimates;
the LQR calculation itself does not construct an estimator.

For a position reference $r$, we will form $x_{ref}=[r^T\;0\;\cdots\;0]^T$ and apply

$$ u=u_e-K(x-x_{ref}). $$

We will compose this law with the **nonlinear** model. The third input of the new
closed-loop function is now the three-dimensional position reference; the
four physical rotor commands are calculated inside it:

```{testcode} symbolic-quadrotor
t_symbol = ca.SX.sym("t")
x_symbol = ca.SX.sym("x", 12)
reference_symbol = ca.SX.sym("reference", 3)
reference_state = ca.vertcat(reference_symbol, ca.DM.zeros(9))
rotor_input = ca.DM(u_equilibrium) - ca.DM(K) @ (x_symbol - reference_state)

closed_loop_model = ca.Function(
    "quadrotor_lqr", [t_symbol, x_symbol, reference_symbol],
    [quadrotor_model(t_symbol, x_symbol, rotor_input)],
    ["t", "x", "u"], ["dx"],
)
```

`ca.SX.sym` declares placeholders for time, state and reference.
`ca.vertcat` appends nine zeros to the position reference, requesting zero
attitude and velocity. `ca.DM` inserts the numeric hover input and gain into
the expression. The last two argument lists in `ca.Function` name its inputs
and output: the new input named `u` carries three reference coordinates,
while the four physical commands remain inside `rotor_input`.

This is a local hover design. The model uses Euler angles and has singular
coordinates at a pitch of ±π/2. LQR stability of the linearized model is
not a guarantee for arbitrary nonlinear initial conditions, reference steps
or actuator limits. This example does not clip rotor commands; a physical
implementation must respect nonnegative squared speeds and actual motor
limits. The parameters define a benchmark model.

## Integrate the nonlinear closed loop

We will use a sample period of 0.01 s and hold each reference for 10 s:
`[0, 0, 8]`, `[3, 2, 5]`, then `[1, 1, 8]`. Small disturbances will be added to all
initial states with a fixed seed.

`solve_ivp_casadi` holds its input constant for each call. Here that input is
the position reference, while the composed feedback law continuously
recomputes the rotor commands from the simulated state. The final state will be carried between intervals. Absolute interval times
and output grids that omit each starting sample will avoid duplicates:

```{testcode} symbolic-quadrotor
from benchmark_systems.symbolic import solve_ivp_casadi

rng = np.random.default_rng(42)
initial_state = x_equilibrium + 0.05 * rng.standard_normal(12)
dt = 0.01
reference_duration = 10.0
position_references = np.array([
    [0.0, 0.0, 8.0],
    [3.0, 2.0, 5.0],
    [1.0, 1.0, 8.0],
])
samples_per_reference = int(round(reference_duration / dt))
time_segments = []
state_segments = []

for index, reference in enumerate(position_references):
    start = index * reference_duration
    stop = start + reference_duration
    output_times = np.linspace(start + dt, stop, samples_per_reference)
    segment = solve_ivp_casadi(
        closed_loop_model, (start, stop), initial_state,
        t_eval=output_times, u=reference,
    )
    time_segments.append(segment.t)
    state_segments.append(segment.y)
    initial_state = segment.y[:, -1]

quadrotor_times = np.concatenate(time_segments)
quadrotor_states = np.hstack(state_segments)
reference_positions = np.repeat(
    position_references.T, samples_per_reference, axis=1,
)
reference_states = np.vstack([
    reference_positions, np.zeros((9, quadrotor_times.size)),
])
rotor_commands = (
    u_equilibrium[:, None] - K @ (quadrotor_states - reference_states)
)

print(quadrotor_states.shape, quadrotor_times[-1])
```

```{testoutput} symbolic-quadrotor
(12, 3000) 30.0
```

`default_rng(42)` fixes the initial disturbance, and the factor `0.05` scales
it before adding it to the equilibrium. `dt` sets the requested output
spacing; the integrator can take finer internal steps. Each loop uses
absolute `start` and `stop` times, requests 1,000 samples after `start`, and
carries `segment.y[:, -1]` into the next interval. Concatenation therefore
produces 3,000 samples without duplicate boundary times.

`np.repeat` holds each position reference over its 1,000 columns, and
`np.vstack` adds the nine zero attitude/velocity targets. Subtracting these
targets from the simulated states reconstructs all four physical
`rotor_commands` using the same feedback law. For this seeded trajectory,
the final position error is below 2 cm and the sampled commands remain
nonnegative; the code does not impose saturation or guarantee constraints
between samples.

`t_eval` must be finite and strictly increasing within an increasing
`t_span`. Each interval's endpoint must be included when its state will
initialize the next interval. The default ODE backend is CasADi `rk`; alternatives include
`cvodes` and `collocation`. Backend-specific options are passed with `opts`.
The returned ODE `.yp` array contains zeros as a placeholder; evaluate the
model separately when state derivatives are required.

### Plot position, attitude and rotor commands

We will plot positions and angles in separate panels to keep their units clear.
Dashed lines show the position references:

```{testcode} symbolic-quadrotor
import matplotlib.pyplot as plt

quadrotor_figure, state_axes = plt.subplots(
    2, 1, figsize=(9, 6.5), sharex=True, constrained_layout=True,
)
colours = ["#0072B2", "#D55E00", "#009E73"]
for component, (label, colour) in enumerate(zip(["x", "y", "z"], colours)):
    state_axes[0].plot(
        quadrotor_times, quadrotor_states[component], color=colour, label=label,
    )
    state_axes[0].plot(
        quadrotor_times, reference_positions[component], "--", color=colour,
        alpha=0.65, label=f"{label} reference",
    )
for component, (label, colour) in enumerate(
    zip(["roll", "pitch", "yaw"], colours), start=3,
):
    state_axes[1].plot(
        quadrotor_times, quadrotor_states[component], color=colour, label=label,
    )
state_axes[0].set_title("Nonlinear quadrotor tracking with hover LQR")
state_axes[0].set_ylabel("Position (m)")
state_axes[1].set_ylabel("Euler angle (rad)")
state_axes[1].set_xlabel("Time (s)")
for axis in state_axes:
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=3, loc="upper right")

quadrotor_control_figure, control_axis = plt.subplots(
    figsize=(9, 3.5), constrained_layout=True,
)
for rotor in range(4):
    control_axis.plot(
        quadrotor_times, rotor_commands[rotor], label=f"Rotor {rotor + 1}",
    )
control_axis.axhline(
    hover_squared_speed, color="0.35", linestyle="--", label="Hover input",
)
control_axis.set_title("Squared rotor-speed commands from the LQR law")
control_axis.set_xlabel("Time (s)")
control_axis.set_ylabel(r"Squared rotor speed $\omega_i^2$ ((rad/s)$^2$)")
control_axis.grid(True, alpha=0.25)
control_axis.legend(ncol=3)
```

The first loop plots state rows 0–2 as positions, then draws their references
with matching `colours`, dashed lines and `alpha=0.65`. Starting the second
loop at row 3 selects roll, pitch and yaw. Shared horizontal axes align both
panels in time, while their labels distinguish metres from radians.

The separate control plot draws one curve per rotor. Its dashed horizontal
line marks `hover_squared_speed`, making the hover offset and transient
control corrections visible in squared-speed units.

```{figure} _static/quadrotor-states.svg
:alt: Quadrotor positions tracking three piecewise references, with roll, pitch and yaw shown in a separate panel.

Position tracking and attitude transients for the nonlinear plant.
```

```{figure} _static/quadrotor-inputs.svg
:alt: Four nonnegative squared-rotor-speed commands changing at reference transitions and settling toward the hover input.

Control effort in the physical input coordinates, including the hover offset.
```

### Animate the same trajectory

The `Quadrotor` helper takes inertial position `y[:3]`, Euler angles
`y[3:6]`, and optional reference positions, all component-major arrays.
We will create a 3D axis and three projection axes:

```{testcode} symbolic-quadrotor
from benchmark_systems.animations import AutoAnimation, Quadrotor

quadrotor_animation_figure = plt.figure(figsize=(9, 7))
quadrotor_animation_figure.subplots_adjust(
    left=0.09, right=0.97, bottom=0.18, top=0.95, hspace=0.35, wspace=0.35,
)
axis_3d = quadrotor_animation_figure.add_subplot(221, projection="3d")
axis_xy = quadrotor_animation_figure.add_subplot(222)
axis_xz = quadrotor_animation_figure.add_subplot(223)
axis_yz = quadrotor_animation_figure.add_subplot(224)
axis_3d.set_xlabel("x (m)")
axis_3d.set_ylabel("y (m)")
axis_3d.set_zlabel("z (m)")
for axis, horizontal, vertical in [
    (axis_xy, "x (m)", "y (m)"),
    (axis_xz, "x (m)", "z (m)"),
    (axis_yz, "y (m)", "z (m)"),
]:
    axis.set_xlabel(horizontal)
    axis.set_ylabel(vertical)
    axis.grid(True, alpha=0.25)

quadrotor_update = Quadrotor(
    quadrotor_animation_figure, axis_3d, axis_xy, axis_xz, axis_yz,
    L=quadrotor_parameters["L"], xyz=quadrotor_states[:3],
    angles=quadrotor_states[3:6], xyz_ref=reference_positions,
    fixed_view=(22, -55),
)
quadrotor_animation, quadrotor_frames = AutoAnimation(
    quadrotor_animation_figure, quadrotor_update,
    times=quadrotor_times, speed=20, fps=20, repeat=True,
)
quadrotor_animation_figure.canvas.draw()
```

`221` through `224` arrange a two-by-two grid; only the first axis uses
`projection="3d"`. The margins reserve room for the animation's text.
`xyz=quadrotor_states[:3]` supplies positions, `angles=quadrotor_states[3:6]`
supplies attitude, and `xyz_ref` supplies the targets. Here `L` sets the
drawing's arm length; `fixed_view` fixes camera elevation and azimuth.

`times` supplies simulation-time labels, `speed=20` selects every twentieth
sample, and `fps=20` sets playback rate. `repeat=True` loops the replay;
`canvas.draw()` renders its initial view. A reference to `quadrotor_animation` must remain alive
while displaying or saving it. In an interactive desktop session, call
`plt.show()` to open the figures; see the [animation guide](animations.md)
for inline display and GIF/video export.

```{figure} _static/quadrotor.gif
:alt: Animated quadrotor moving through three position references, with a 3D view and XY, XZ and YZ projections.

The same 30-second nonlinear experiment replayed with a stride of 20 samples
at 20 frames per second. The displayed time remains simulation time.
```

## Symbolic DAEs

DAE factories return `DAEFunction` models, which can be simulated with
`solve_dae_casadi`. Initial states and derivatives must be consistent with
the model. This example starts at the hanging equilibrium:

```{testcode} symbolic-dae
import numpy as np
from benchmark_systems.symbolic import double_cart_pendulum, solve_dae_casadi

dae = double_cart_pendulum(M=0.6, m1=0.2, m2=0.2, L1=0.5, L2=0.5)
y0 = np.zeros(6)
yp0 = np.zeros(6)
result = solve_dae_casadi(
    dae, (0.0, 0.2), y0, yp0, u=0.0,
    t_eval=np.linspace(0.0, 0.2, 11),
)
print(dae.state_size, dae.input_size, result.y.shape)
```

```{testoutput} symbolic-dae
6 1 (6, 11)
```

`M`, `m1` and `m2` bind the cart/link masses in kg; `L1` and `L2` set link
lengths in m.
The six entries of `y0` set cart position, two angles and their velocities;
`yp0` supplies their initial derivatives. Zeros describe the hanging,
stationary configuration with cart force `u=0.0`. The time span lasts 0.2 s,
and `np.linspace` requests 11 output samples. `state_size` and `input_size`
report six states and one input. `result.y` has one row per state and one
column per output time. Other starting configurations need suitable initial values; see
[DAE initialization](numerical.md#differential-algebraic-equations).

`solve_dae_casadi` uses CasADi's IDAS integrator. See the
[integrator reference](api/symbolic.md#dae-integration) for output fields and
initialization details.

See the [symbolic API reference](api/symbolic.md) for factories, wrappers and
integrator options. Some input conventions, notably the Johansson model's
flow-split parameters, differ between numerical and symbolic interfaces; these
are listed in the [catalogue](models.md).
