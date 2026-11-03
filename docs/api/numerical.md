# Numerical models

Import from `benchmark_systems.models` or directly from `benchmark_systems`.
The same physical parameters and state orders are used by the
[symbolic factories](symbolic.md), except that the symbolic Johansson model
combines pump commands and valve fractions into one four-element input.
The [model catalogue](../models.md) gives the complete state and unit tables.

## ODE callables

An ODE callable accepts time, a one-dimensional state vector and keyword
parameters, and returns a NumPy derivative vector of the same length.
Bind the parameters and input to obtain a SciPy-style `fun(t, x)` callable.

### Pendulum

Two states `[theta, omega]`; scalar input is pivot torque.

```{figure} ../_static/models/pendulum.png
:alt: Pendulum rod of length L with a point mass m, angle theta, applied torque u and gravity.
:width: 137px
:align: center

Torque-driven point-mass pendulum.
```

```{eval-rst}
.. autofunction:: benchmark_systems.models.pendulum
```

### DC motor

Three states `[theta, omega, current]`; scalar input is armature voltage.

```{figure} ../_static/models/dc-motor.png
:alt: DC motor shaft at 45 degrees with angular-velocity and armature-current indicators.
:width: 580px
:align: center

Shaft position, speed and armature-current indicators from the `DCMotor` animation helper.
```

```{eval-rst}
.. autofunction:: benchmark_systems.models.dc_motor
```

### Cart pendulum

Four states `[position, velocity, theta, omega]`; scalar input is cart force.

```{figure} ../_static/models/cart-pendulum.png
:alt: Force-driven cart carrying a point-mass pendulum, with position, velocity and angle coordinates.
:width: 188px
:align: center

Cart and point-mass pendulum with horizontal force input.
```

```{eval-rst}
.. autofunction:: benchmark_systems.models.cart_pendulum
```

### Furuta pendulum

Four states `[theta1, theta2, omega1, omega2]`; scalar input is motor voltage.
Supply explicit centre-of-mass distances `l1` and `l2`.

```{figure} ../_static/models/furuta-pendulum.png
:alt: Rotary motor arm carrying a pendulum, with two joint angles, centres of mass and motor torque K times u.
:width: 460px
:align: center

Furuta pendulum geometry, centre-of-mass distances and motor torque.
```

```{eval-rst}
.. autofunction:: benchmark_systems.models.furuta_pendulum
```

### Multimass spring system

The `2+2*N` states include two actual motor positions before disc positions
and velocities. Input contains two motor position setpoints.

```{figure} ../_static/models/multimass-spring.png
:alt: A chain of rotating discs coupled by springs, driven by stepper motors at both ends.
:width: 620px
:align: center

Spring-coupled rotating discs between two position-controlled motors.
```

```{eval-rst}
.. autofunction:: benchmark_systems.models.multimass_spring
```

### Johansson four-tank process

Four height states. Pass two pump commands in `u` and two valve fractions
separately in `gamma`.

```{figure} ../_static/models/johansson.png
:alt: Four interconnected water tanks supplied by two pumps and two splitting valves, with upper tanks draining into lower tanks.
:width: 460px
:align: center

Four-tank process with crossed upper-tank pump feeds.
```

```{eval-rst}
.. autofunction:: benchmark_systems.models.johansson
```

### Batch bioreactor

Four states `[X, S, P, V]`; scalar input is feed flow.

```{figure} ../_static/models/batch-bioreactor.png
:alt: Stirred bioreactor with a feed stream, biomass, substrate and product concentrations, and variable liquid volume.
:width: 266px
:align: center

Fed batch bioreactor with feed-flow input and variable liquid volume.
```

```{eval-rst}
.. autofunction:: benchmark_systems.models.batch_bioreactor
```

### Continuous stirred-tank reactor

Four states `[Ca, Cb, Tr, Tk]`; input is `[F, dQ]`. See the
[catalogue's scaling notes](../models.md#continuous-stirred-tank-reactor)
before converting volumetric flow or thermal-power data.

```{figure} ../_static/models/cstr.png
:alt: Stirred tank with feed and outlet streams, species A and B, and a cooling jacket exchanging heat with the reactor.
:width: 410px
:align: center

Continuous stirred-tank reactor and cooling jacket.
```

```{eval-rst}
.. autofunction:: benchmark_systems.models.cstr
```

### Quadrotor ODE

Twelve states with body-frame velocity components. Use the public name
`quadrotor_ode` to distinguish it from the DAE formulation.

```{figure} ../_static/models/quadrotor.png
:alt: Four-rotor aircraft showing inertial and body axes, rotor forces, speeds and reaction torques.
:width: 426px
:align: center

Quadrotor geometry and reference frames used by both formulations.
Source: [MathWorks, quadrotor dynamics for nonlinear MPC](https://es.mathworks.com/help/symbolic/derive-quadrotor-dynamics-for-nonlinearMPC.html).
```

```{eval-rst}
.. autofunction:: benchmark_systems.models.quadrotor_ode
```

## DAE models

DAE callables take both states and state derivatives. Use them with a
DAE solver such as `solve_dae`, supplying consistent initial values for both.

### Double cart pendulum

Six differential states describing cart position, link angles and their
velocities.

```{figure} ../_static/models/double-cart-pendulum.png
:alt: A force-driven cart with two serial pendulum links and their absolute angular coordinates.
:width: 195px
:align: center

Two serial pendulum links mounted on a translating cart.
```

```{eval-rst}
.. autofunction:: benchmark_systems.models.double_cart_pendulum
```

### Multilevel cart pendulum

`2*(N+1)` differential states. When supplying friction, `deltas` contains
one coefficient for the cart and one for each pendulum link.

```{figure} ../_static/models/multilevel-cart-pendulum.png
:alt: Three serial pendulum links on a cart, showing link lengths, masses, angles and joint friction.
:width: 288px
:align: center

Three-link example of the general multilevel cart-pendulum arrangement.
```

```{eval-rst}
.. autofunction:: benchmark_systems.models.multilevel_cart_pendulum
```

### Quadrotor DAE

Twelve differential states with inertial linear velocities and Euler-angle
rates. They are not the same velocity coordinates as `quadrotor_ode`.

```{figure} ../_static/models/quadrotor.png
:alt: Four-rotor aircraft showing inertial and body axes, rotor forces, speeds and reaction torques.
:width: 426px
:align: center

Quadrotor geometry and reference frames used by both formulations.
Source: [MathWorks, quadrotor dynamics for nonlinear MPC](https://es.mathworks.com/help/symbolic/derive-quadrotor-dynamics-for-nonlinearMPC.html).
```

```{eval-rst}
.. autofunction:: benchmark_systems.models.quadrotor_dae
```

### Gas-lift oil well

Eleven states: three differential masses and eight algebraic intermediate
quantities. Pressure states are expressed in bar.

```{figure} ../_static/models/oil-well.png
:alt: Gas-lift oil well showing annulus, tubing, injection valve, production and lift chokes, pressures and mixture flows.
:width: 621px
:align: center

Gas-lift well, manipulated choke openings and measured process quantities.
Source: [Jahanshahi et al., 2012](https://doi.org/10.3182/20120710-4-SG-2026.00110).
```

```{eval-rst}
.. autofunction:: benchmark_systems.models.oil_well
```

### pH neutralization

Four states: three differential variables and algebraic pH. The fourth
derivative argument has no role in the pH constraint.

```{figure} ../_static/models/neutralization.png
:alt: Neutralization reactor with acid, buffer and alkaline inlet streams, liquid-height measurement and an outlet pH sensor.
:width: 560px
:align: center

Neutralization process with acid feed, buffer disturbance and controlled alkaline flow.
```

```{eval-rst}
.. autofunction:: benchmark_systems.models.neutralization
```
