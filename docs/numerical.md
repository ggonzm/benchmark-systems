# Numerical simulation

The numerical API is available through `benchmark_systems.models` and the
package root. Physical parameters are keyword arguments; the control input
is supplied through `u`.

## Ordinary differential equations

An ODE model returns $\dot{x}=f(t,x,u)$. We will bind parameters and the input in a
callable with the `(t, x)` signature expected by SciPy:

```{testcode} numerical-ode
import numpy as np
from scipy.integrate import solve_ivp
from benchmark_systems.models import pendulum

def rhs(t, x):
    return pendulum(t, x, m=1.0, L=1.0, drag=0.1, u=0.5)

result = solve_ivp(
    rhs, (0.0, 4.0), [0.2, 0.0],
    t_eval=np.linspace(0.0, 4.0, 201),
    rtol=1e-8, atol=1e-10,
)
print(result.y.shape)
```

```{testoutput} numerical-ode
(2, 201)
```

`rhs` fixes the pendulum parameters and applies a constant torque
`u=0.5` N m. The interval `(0.0, 4.0)` and initial state `[0.2, 0.0]` start a
four-second experiment at 0.2 rad with no angular velocity. `t_eval` requests
201 equally spaced output samples; `rtol` and `atol` set the solver's
relative and absolute error tolerances.

The state vector is one-dimensional, with its entries in the order listed in the
[model catalog](models.md). Returned trajectories follow the solver's
`(number_of_states, number_of_output_times)` convention. These callables are
not automatically vectorized over a batch of states.

We will plot the two state trajectories on separate axes to keep their units explicit:

```{testcode} numerical-ode
import matplotlib.pyplot as plt

ode_figure, axes = plt.subplots(2, 1, sharex=True, figsize=(6.4, 4.8))
axes[0].plot(result.t, result.y[0])
axes[0].set_ylabel("Angle (rad)")
axes[1].plot(result.t, result.y[1])
axes[1].set_ylabel("Angular velocity (rad/s)")
axes[1].set_xlabel("Time (s)")
ode_figure.tight_layout()
plt.show()
```

The first `plot` uses row 0 of `result.y` for angle; the second uses row 1
for angular velocity. Both use `result.t` as their horizontal coordinate.
`sharex=True` aligns the time axes, while separate `set_ylabel` calls
identify the units. `tight_layout()` reserves space for these labels before
`plt.show()` displays the figure.

```{figure} _static/numerical-ode.svg
:alt: Pendulum angle and angular velocity under a constant applied torque.

Response to the constant 0.5 N m torque used above.
```

## Piecewise-constant inputs

For a stepped input, we will integrate each interval with its own fixed `u`,
carry the last state into the next interval and preserve absolute time.
Including each interval's endpoint gives the state at that boundary.

```{testcode} numerical-piecewise
import numpy as np
from scipy.integrate import solve_ivp
from benchmark_systems.models import pendulum

state = np.array([0.2, 0.0])
time_parts = [np.array([0.0])]
state_parts = [state[:, None]]

for start, end, torque in [(0.0, 2.0, 0.0), (2.0, 4.0, 1.0)]:
    output_times = np.linspace(start, end, 101)[1:]
    segment = solve_ivp(
        lambda t, x: pendulum(t, x, m=1.0, L=1.0, drag=0.1, u=torque),
        (start, end), state, t_eval=output_times,
        rtol=1e-8, atol=1e-10,
    )
    time_parts.append(segment.t)
    state_parts.append(segment.y)
    state = segment.y[:, -1]

t = np.concatenate(time_parts)
x = np.hstack(state_parts)
print(x.shape)
```

```{testoutput} numerical-piecewise
(2, 201)
```

The tuples in the loop specify `(start, end, torque)`: zero torque for the
first two seconds, then 1 N m for the next two. On each iteration, `u=torque`
sets that interval's input. `segment.y[:, -1]` becomes the next initial
state, so the state remains continuous when the input changes.

`state[:, None]` stores the initial state as one column. The slice `[1:]`
removes each interval's starting sample because that boundary is already
stored. `np.concatenate(time_parts)` joins the time vectors and
`np.hstack(state_parts)` joins the state columns, giving one trajectory with
201 distinct times.

We will plot the applied torque alongside the states:

```{testcode} numerical-piecewise
import matplotlib.pyplot as plt

piecewise_figure, axes = plt.subplots(3, 1, sharex=True, figsize=(6.4, 6.0))
axes[0].plot(t, x[0])
axes[0].set_ylabel("Angle (rad)")
axes[1].plot(t, x[1])
axes[1].set_ylabel("Angular velocity (rad/s)")
axes[2].step(t, np.where(t < 2.0, 0.0, 1.0), where="post")
axes[2].set_ylabel("Torque (N m)")
axes[2].set_xlabel("Time (s)")
piecewise_figure.tight_layout()
plt.show()
```

`np.where(t < 2.0, 0.0, 1.0)` reconstructs the torque command at each output
time. `step(..., where="post")` draws each command as constant until the
next sample, placing the jump at 2 s. The upper panels use the assembled
state array `x`; plotting all three against the same `t` makes the response
to the input switch visible.

```{figure} _static/numerical-piecewise.svg
:alt: Continuous pendulum states with a torque step at two seconds.

A piecewise-constant input and the complete trajectory assembled from both intervals.
```

## Differential-algebraic equations

DAE models describe implicit equations involving states and their
derivatives. A DAE solver requires both an initial state and an initial
derivative consistent with the model.

We will use `solve_dae` from the `solve-dae` distribution. Its Python import is
`solve_dae.integrate`, and its solver receives both an initial state `y0` and
an initial derivative `yp0`.

This example starts a double cart-pendulum near its hanging configuration.
For this model, `root` computes the initial derivative from the initial state:

```{testcode} numerical-dae
import numpy as np
from scipy.optimize import root
from solve_dae.integrate import solve_dae
from benchmark_systems.models import double_cart_pendulum

parameters = {"M": 0.6, "m1": 0.2, "m2": 0.2, "L1": 0.5, "L2": 0.5}

def model(t, x, xp):
    return double_cart_pendulum(t, x, xp, **parameters, u=0.0)

y0 = np.array([0.0, 0.15, 0.1, 0.0, 0.0, 0.0])
yp0 = root(lambda xp: model(0.0, y0, xp), np.zeros_like(y0)).x

dae_result = solve_dae(
    model, (0.0, 2.0), y0=y0, yp0=yp0,
    t_eval=np.linspace(0.0, 2.0, 101),
    rtol=1e-9, atol=1e-11,
)
print(dae_result.y.shape)
```

```{testoutput} numerical-dae
(6, 101)
```

Here `M` is the cart mass, `m1` and `m2` are the link masses, and `L1` and
`L2` are their lengths. `model(t, x, xp)` binds these parameters and a zero
cart force. The six entries of `y0` are cart position, two link angles,
cart velocity and two angular velocities; only the angles are initially
nonzero.

`root` keeps `y0` fixed and solves for a compatible initial derivative.
`np.zeros_like(y0)` supplies its starting guess, and `.x` extracts the
computed derivative vector as `yp0`. `solve_dae` then receives both `y0`
and `yp0`, integrates over two seconds and returns 101 requested samples.
The `rtol` and `atol` arguments control integration accuracy.

For models with algebraic variables, such as `oil_well` and `neutralization`,
the initial state must also satisfy their algebraic equations. See the
[API reference](api/numerical.md) for each model's initialization requirements.

The same sampled-state layout can be plotted for a DAE solution:

```{testcode} numerical-dae
import matplotlib.pyplot as plt

dae_figure, axes = plt.subplots(4, 1, sharex=True, figsize=(6.4, 7.2))
axes[0].plot(dae_result.t, dae_result.y[0])
axes[0].set_ylabel("Cart position (m)")
axes[1].plot(dae_result.t, dae_result.y[1], label="Pendulum 1")
axes[1].plot(dae_result.t, dae_result.y[2], label="Pendulum 2")
axes[1].set_ylabel("Angle (rad)")
axes[1].legend()
axes[2].plot(dae_result.t, dae_result.y[3])
axes[2].set_ylabel("Cart velocity (m/s)")
axes[3].plot(dae_result.t, dae_result.y[4], label="Pendulum 1")
axes[3].plot(dae_result.t, dae_result.y[5], label="Pendulum 2")
axes[3].set_ylabel("Angular velocity (rad/s)")
axes[3].set_xlabel("Time (s)")
dae_figure.tight_layout()
plt.show()
```

The four panels select rows `0`, `1:3`, `3` and `4:6` of the solution:
cart position, link angles, cart velocity and link angular velocities.
The `label` arguments identify the two links, and `legend()` displays their
names. Positions and angles have separate panels so metres and radians
are not mixed on one vertical axis.

```{figure} _static/numerical-dae.svg
:alt: Cart position, pendulum angles and velocities from the implicit double cart-pendulum simulation.

The simulated double cart-pendulum trajectory.
```

## Model conventions

`quadrotor_ode` and `quadrotor_dae` use different velocity coordinates.
Their last six state entries cannot be exchanged directly. The
[catalog](models.md) specifies both orderings, process-model unit conventions
and the distinction between differential and algebraic states.

See the [numerical API reference](api/numerical.md) for complete signatures
and parameters, and [symbolic models](symbolic.md) for CasADi integration.
