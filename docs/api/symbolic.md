# Symbolic models and integrators

Import this API from `benchmark_systems.symbolic`; it requires the `sym`
extra. Model functions are **factories**: pass physical parameters once to
build a CasADi function, then evaluate or differentiate that function with
respect to state and input. Physical parameters are bound at construction;
the returned model does not expose a separate tunable parameter vector.

The [model catalogue](../models.md) defines state orders, units and domains.
For examples of Jacobians and operating-point linearization, see the
[symbolic guide](../symbolic.md).

## CasADi input and output names

All built-in symbolic ODE models have named inputs `t`, `x`, `u` and output
`dx`. These names determine the keys returned by named evaluation and by
CasADi's `jacobian()` function.

`x` is the state, `u` the control input, and `dx` the computed state
derivative. The Jacobian naming rule is `jac_<output>_<input>`:

| Key | Derivative |
| --- | --- |
| `jac_dx_x` | State derivative with respect to the state |
| `jac_dx_u` | State derivative with respect to the control input |
| `jac_dx_t` | State derivative with respect to time |

The returned function's `name_in()` and `name_out()` methods list its input
and output names; `f.jacobian().name_out()` lists the derivative blocks.
The Jacobian function also has an auxiliary input named `out_dx`,
representing the original function output rather than an additional model
input.

The [quadrotor example](../symbolic.md#linearize-around-hover) demonstrates
how to evaluate `jac_dx_x` and `jac_dx_u` to obtain the local matrices
`A` and `B`.

## ODE factories

Each factory returns a CasADi function with named inputs `t`, `x`, `u` and
named output `dx`. Positional evaluation `f(t, x, u)` returns a CasADi column
vector; keyword evaluation returns a dictionary containing `dx`. A model
with `nx` states and `nu` inputs expects sizes `1`, `nx` and `nu` respectively.
The function remains symbolic when called with CasADi symbolic arguments.

### Pendulum

Build a two-state pendulum function. Input is one pivot torque.

```{figure} ../_static/models/pendulum.png
:alt: Pendulum rod of length L with a point mass m, angle theta, applied torque u and gravity.
:width: 137px
:align: center

Torque-driven point-mass pendulum.
```

```{eval-rst}
.. autofunction:: benchmark_systems.symbolic.pendulum
```

### DC motor

Build a three-state motor function. Input is one armature voltage.

```{figure} ../_static/models/dc-motor.png
:alt: DC motor shaft at 45 degrees with angular-velocity and armature-current indicators.
:width: 580px
:align: center

Shaft position, speed and armature-current indicators from the `DCMotor` animation helper.
```

```{eval-rst}
.. autofunction:: benchmark_systems.symbolic.dc_motor
```

### Cart pendulum

Build a four-state cart-pendulum function. Input is one cart force.

```{figure} ../_static/models/cart-pendulum.png
:alt: Force-driven cart carrying a point-mass pendulum, with position, velocity and angle coordinates.
:width: 188px
:align: center

Cart and point-mass pendulum with horizontal force input.
```

```{eval-rst}
.. autofunction:: benchmark_systems.symbolic.cart_pendulum
```

### Furuta pendulum

Build a four-state rotary-pendulum function. Input is one motor voltage.
Centre-of-mass distances are explicit factory parameters.

```{figure} ../_static/models/furuta-pendulum.png
:alt: Rotary motor arm carrying a pendulum, with two joint angles, centres of mass and motor torque K times u.
:width: 460px
:align: center

Furuta pendulum geometry, centre-of-mass distances and motor torque.
```

```{eval-rst}
.. autofunction:: benchmark_systems.symbolic.furuta_pendulum
```

### Multimass spring system

Build a `2+2*N`-state function for `N>=2` discs. The first two states are
filtered motor positions; input is the two commanded positions.

```{figure} ../_static/models/multimass-spring.png
:alt: A chain of rotating discs coupled by springs, driven by stepper motors at both ends.
:width: 620px
:align: center

Spring-coupled rotating discs between two position-controlled motors.
```

```{eval-rst}
.. autofunction:: benchmark_systems.symbolic.multimass_spring
```

### Johansson four-tank process

Build a four-state function with **four inputs**, ordered as
`[pump1, pump2, valve1, valve2]`. Pump commands are in the 0–100% scale;
valve values are fractions in the 0–1 scale. Unlike the numerical callable,
this function has no separate runtime `gamma` argument. `gamma` is formed
from the last two elements of `u`.

```{figure} ../_static/models/johansson.png
:alt: Four interconnected water tanks supplied by two pumps and two splitting valves, with upper tanks draining into lower tanks.
:width: 460px
:align: center

Four-tank process with crossed upper-tank pump feeds.
```

```{eval-rst}
.. autofunction:: benchmark_systems.symbolic.johansson
```

### Batch bioreactor

Build a four-state Haldane-kinetics function. Input is one feed flow.

```{figure} ../_static/models/batch-bioreactor.png
:alt: Stirred bioreactor with a feed stream, biomass, substrate and product concentrations, and variable liquid volume.
:width: 266px
:align: center

Fed batch bioreactor with feed-flow input and variable liquid volume.
```

```{eval-rst}
.. autofunction:: benchmark_systems.symbolic.batch_bioreactor
```

### Continuous stirred-tank reactor

Build a four-state function with inputs `[F, dQ]`. Both backends use the
same balance-equation scaling; see the
[CSTR catalogue entry](../models.md#continuous-stirred-tank-reactor).

```{figure} ../_static/models/cstr.png
:alt: Stirred tank with feed and outlet streams, species A and B, and a cooling jacket exchanging heat with the reactor.
:width: 410px
:align: center

Continuous stirred-tank reactor and cooling jacket.
```

```{eval-rst}
.. autofunction:: benchmark_systems.symbolic.cstr
```

### Quadrotor ODE

Build a twelve-state Newton–Euler function. Input consists of four squared
rotor speeds; the velocity states use body coordinates.

```{figure} ../_static/models/quadrotor.png
:alt: Four-rotor aircraft showing inertial and body axes, rotor forces, speeds and reaction torques.
:width: 426px
:align: center

Quadrotor geometry and reference frames used by both formulations.
Source: [MathWorks, quadrotor dynamics for nonlinear MPC](https://es.mathworks.com/help/symbolic/derive-quadrotor-dynamics-for-nonlinearMPC.html).
```

```{eval-rst}
.. autofunction:: benchmark_systems.symbolic.quadrotor_ode
```

## DAE factories

These factories return `DAEFunction` models. Simulate them with
`solve_dae_casadi`, using consistent initial states and derivatives.

### Double cart pendulum

Six states, all differential; one cart-force input.

```{figure} ../_static/models/double-cart-pendulum.png
:alt: A force-driven cart with two serial pendulum links and their absolute angular coordinates.
:width: 195px
:align: center

Two serial pendulum links mounted on a translating cart.
```

```{eval-rst}
.. autofunction:: benchmark_systems.symbolic.double_cart_pendulum
```

### Multilevel cart pendulum

`2*(N+1)` states, all differential; one cart-force input. Parameter `deltas`
has `N+1` entries when friction is enabled.

```{figure} ../_static/models/multilevel-cart-pendulum.png
:alt: Three serial pendulum links on a cart, showing link lengths, masses, angles and joint friction.
:width: 288px
:align: center

Three-link example of the general multilevel cart-pendulum arrangement.
```

```{eval-rst}
.. autofunction:: benchmark_systems.symbolic.multilevel_cart_pendulum
```

### Quadrotor DAE

Twelve states, all differential; four squared-rotor-speed inputs. Velocity
coordinates differ from the ODE formulation.

```{figure} ../_static/models/quadrotor.png
:alt: Four-rotor aircraft showing inertial and body axes, rotor forces, speeds and reaction torques.
:width: 426px
:align: center

Quadrotor geometry and reference frames used by both formulations.
Source: [MathWorks, quadrotor dynamics for nonlinear MPC](https://es.mathworks.com/help/symbolic/derive-quadrotor-dynamics-for-nonlinearMPC.html).
```

```{eval-rst}
.. autofunction:: benchmark_systems.symbolic.quadrotor_dae
```

### Gas-lift oil well

Eleven states, with the first three masses differential and the final eight
quantities algebraic. Input is `[production_choke, gas_lift_choke]`.

```{figure} ../_static/models/oil-well.png
:alt: Gas-lift oil well showing annulus, tubing, injection valve, production and lift chokes, pressures and mixture flows.
:width: 621px
:align: center

Gas-lift well, manipulated choke openings and measured process quantities.
Source: [Jahanshahi et al., 2012](https://doi.org/10.3182/20120710-4-SG-2026.00110).
```

```{eval-rst}
.. autofunction:: benchmark_systems.symbolic.oil_well
```

### pH neutralization

Four states, with `[Wa4, Wb4, h]` differential and `pH` algebraic.
Input is the alkaline flow; buffer disturbance `d` is bound by the factory.

```{figure} ../_static/models/neutralization.png
:alt: Neutralization reactor with acid, buffer and alkaline inlet streams, liquid-height measurement and an outlet pH sensor.
:width: 560px
:align: center

Neutralization process with acid feed, buffer disturbance and controlled alkaline flow.
```

```{eval-rst}
.. autofunction:: benchmark_systems.symbolic.neutralization
```

### DAE wrapper

Factories construct this class for you. A wrapper preserves both the
original residual function and the differential/algebraic split. Its
`state_size` includes algebraic states; `input_size` counts control entries.

```{eval-rst}
.. autoclass:: benchmark_systems.symbolic.DAEFunction
   :members: state_size, input_size, dae_dict
   :special-members: __call__
```

## Integration helpers

Supply output times through `t_eval`. Inputs are held constant during each
call; use successive intervals for changing inputs, as shown in the
[numerical guide](../numerical.md#piecewise-constant-inputs).

### ODE integration

The default integrator is CasADi `rk`. Alternatives include `cvodes` and
`collocation`, subject to the available CasADi plugins and valid options.
The local-time integration formulation preserves the model's absolute `t`.

```{eval-rst}
.. autofunction:: benchmark_systems.symbolic.solve_ivp_casadi
```

### DAE integration

This helper uses CasADi IDAS and the wrapper's differential/algebraic split.
Pass full-length `y0` and `yp0`. Differential entries of `yp0` provide
derivative guesses; algebraic state guesses come from the algebraic part of
`y0`. Algebraic entries of `yp0` are not integrated derivatives.

```{eval-rst}
.. autofunction:: benchmark_systems.symbolic.solve_dae_casadi
```
