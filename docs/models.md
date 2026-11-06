# Model catalogue

Each model has a numerical callable in `benchmark_systems.models` and a
symbolic factory of the same public name in `benchmark_systems.symbolic`.
Numerical callables are also available directly from `benchmark_systems`.

ODE models are integrated with an ODE solver; implicit DAE models use a
DAE solver with initial states and derivatives.
The symbolic factories bind physical parameters and return a CasADi function
or a `DAEFunction` wrapper; they do not run a simulation.

The [numerical API](api/numerical.md) lists callable signatures, while the
[symbolic API](api/symbolic.md) lists factory signatures. Parameters are
keyword-only. Unless noted below, these models do not enforce actuator limits
or validate every physical parameter for you.

## Mechanical and electrical systems

Mechanical models use radians for angles. With lengths in metres, masses in
kilograms and moments of inertia in kg m², their time unit is seconds and
their force and torque units are N and N m. Do not mix degrees and radians.

### Pendulum

```{figure} _static/models/pendulum.png
:alt: Pendulum rod of length L with a point mass m, angle theta, applied torque u and gravity.
:width: 137px
:align: center

Torque-driven point-mass pendulum.
```

`pendulum` models a point mass at distance `L` from its pivot, driven by torque
`u`. Its states are angle `theta` in rad and angular velocity `omega` in rad/s.
`theta=0` points downward; `theta=pi` points upward. The inertia is `m*L**2`,
with rotational damping controlled by `drag`.

Mass and length must be positive. The implementation divides by both and does
not include a torque limit or a stop at a specified angle.

### DC motor

```{figure} _static/models/dc-motor.png
:alt: DC motor shaft at 45 degrees with angular-velocity and armature-current indicators.
:width: 580px
:align: center

Shaft position, speed and armature-current indicators from the `DCMotor` animation helper.
```

`dc_motor` has shaft position `theta` (rad), speed `omega` (rad/s) and armature
`current` (A). Input `u` is the armature voltage (V). Parameters specify rotor
inertia `J`, viscous friction `b`, back-EMF constant `Ke`, torque constant `Kt`,
resistance `R` and inductance `L`.

The equations include electrical and mechanical dynamics. Inertia and
inductance must be nonzero; voltage, current and speed are not saturated.
Here, `L` means inductance, not a mechanical length.

### Cart pendulum

```{figure} _static/models/cart-pendulum.png
:alt: Force-driven cart carrying a point-mass pendulum, with position, velocity and angle coordinates.
:width: 188px
:align: center

Cart and point-mass pendulum with horizontal force input.
```

`cart_pendulum` orders its states as cart position (m), cart velocity (m/s),
pendulum angle (rad) and angular velocity (rad/s). Input `u` is cart force (N).
`M` is the cart mass, `m` the pendulum mass and `L` the pendulum length.
The pendulum is a point mass and `drag` acts through cart velocity.
`theta=0` is downward and `theta=pi` upward.

The model has no track boundary, obstacle response or force saturation. Use
positive masses and length to avoid singular denominators.

### Furuta pendulum

```{figure} _static/models/furuta-pendulum.png
:alt: Rotary motor arm carrying a pendulum, with two joint angles, centres of mass and motor torque K times u.
:width: 460px
:align: center

Furuta pendulum geometry, centre-of-mass distances and motor torque.
```

`furuta_pendulum` orders its states as horizontal motor-arm angle `theta1`,
pendulum-arm angle `theta2` (rad), then their velocities `omega1`, `omega2`
(rad/s). `theta2=0` is downward and `theta2=pi` upward. Input `u` is a
dimensionless signed motor command with a nominal range of `[-1, 1]`.
The applied motor torque is `K*u`, where `K` is the torque in N·m at unit
command. With the simplified voltage-to-torque approximation, use
`u = V/Vmax` and `K = (Kt/R)*Vmax`, where `Kt` is the motor torque constant,
`R` is the armature resistance, and `Vmax` is the voltage magnitude at full command.

The model requires both arm masses, inertias about their centres of mass,
friction coefficients, and explicit pivot-to-centre-of-mass distances `l1` and `l2`.
These distances are not inferred as half of `L1` or `L2`. In the current
equations, `L1` enters the coupling while `L2` is accepted but does not appear
in the derivative expressions. Changing `L2` alone therefore does not change
the simulated dynamics.

The inertia terms and angle-dependent denominator must remain nonsingular.
Motor electrical dynamics, voltage saturation and mechanical travel limits
are not included.

### Multimass spring system

```{figure} _static/models/multimass-spring.png
:alt: A chain of rotating discs coupled by springs, driven by stepper motors at both ends.
:width: 620px
:align: center

Spring-coupled rotating discs between two position-controlled motors.
```

`multimass_spring` connects `N` rotating discs between two position-controlled
stepper motors. The first two states are the **actual motor positions**
`u1_true` and `u2_true` (rad), followed by `N` disc angles (rad) and `N` disc
velocities (rad/s). Input `u=[setpoint_left, setpoint_right]` contains motor
position commands (rad), not applied torques.

The actual motor positions follow first-order filters,
`du_true[i]/dt = (u[i] - u_true[i])/tau[i]`. Supply two positive time constants
`tau`, `N` inertias `I`, `N+1` spring constants `K` and `N` damping
coefficients `d`. With the mechanical units above, stiffness is in N m/rad
and damping in N m s/rad.

The model requires `N>=2`: the first-disc equation accesses the neighbouring disc. Inertias
and time constants must be nonzero. The model does not include spring travel
limits or motor rate saturation.

### Double cart pendulum

```{figure} _static/models/double-cart-pendulum.png
:alt: A force-driven cart with two serial pendulum links and their absolute angular coordinates.
:width: 195px
:align: center

Two serial pendulum links mounted on a translating cart.
```

`double_cart_pendulum` has six states ordered as
`[position, theta1, theta2, velocity, omega1, omega2]`. Angles are absolute
link angles measured from the downward vertical. Positions, velocities,
angles and angular velocities have units m, m/s, rad and rad/s respectively.
Input `u` is cart force (N).

The equations use each link's centre of mass at `L/2` and a corresponding
rod inertia; those quantities are computed from the supplied lengths and
masses. Friction is omitted. All six states are differential
(`DAEFunction.n_diff == 6`); the implicit form retains acceleration coupling.
The state derivative argument contains all six derivatives in the same
order and must be consistent with the initial state and input.

### Multilevel cart pendulum

```{figure} _static/models/multilevel-cart-pendulum.png
:alt: Three serial pendulum links on a cart, showing link lengths, masses, angles and joint friction.
:width: 288px
:align: center

Three-link example of the general multilevel cart-pendulum arrangement.
```

`multilevel_cart_pendulum` extends the serial cart-pendulum to `N` links.
All positions and angles come first, followed by all velocities:
`[position, theta1, ..., thetaN, velocity, omega1, ..., omegaN]`.
Angles are absolute, with `pi` pointing up. Input `u` is cart force (N).
The link centres of mass and inertias are calculated from `L/2` as in the
double-pendulum model. All `2*(N+1)` states are differential.

The `m` and `L` sequences must have matching lengths. When friction is
enabled, `deltas` contains `N+1` linear viscous friction coefficients.
The first is for cart motion relative to the ground, in N·s/m. The rest
are for the rotational joints in order from the cart, in N·m·s/rad.
The force on the cart is proportional to its velocity, and each joint
torque opposes the relative angular velocity across the joint. With
`deltas=None`, friction contributions are zero.

### Quadrotor: ODE and DAE

```{figure} _static/models/quadrotor.png
:alt: Four-rotor aircraft showing inertial and body axes, rotor forces, speeds and reaction torques.
:width: 426px
:align: center

Quadrotor geometry and reference frames used by both formulations.
Source: [MathWorks, quadrotor dynamics for nonlinear MPC](https://es.mathworks.com/help/symbolic/derive-quadrotor-dynamics-for-nonlinearMPC.html).
```

Both quadrotor models use inertial positions `[x, y, z]` (m), Euler angles
`[phi, theta, psi]` (rad: roll, pitch, yaw), body principal inertias, thrust
coefficient `k`, arm length `L` and mass `m`. Input
`u=[omega1**2, omega2**2, omega3**2, omega4**2]` contains **squared** rotor
angular velocities. With rotor speeds in rad/s, choose `k` and `drag` so
that `k*u` is thrust (N) and `drag*u` is yaw torque (N m).

The last six states differ:

| Model | Linear velocity states | Angular velocity states |
| --- | --- | --- |
| `quadrotor_ode` | `[s, v, w]`, body-frame m/s | `[p, q, r]`, body-frame rad/s |
| `quadrotor_dae` | `[dx, dy, dz]`, inertial-frame m/s | `[dphi, dtheta, dpsi]`, Euler-angle rates in rad/s |

The ODE uses Newton–Euler equations; the DAE uses an implicit Euler–Lagrange
form and has 12 differential states. Do not pass one model's velocity states
unchanged into the other. In particular, body angular rates are related to
Euler-angle rates by an angle-dependent transformation.

Euler-angle coordinates are singular when `cos(theta)=0`. Aerodynamic drag,
rotor dynamics and rotor-speed saturation are not included. Despite its
name, the parameter `drag` is the rotor yaw-torque coefficient, not a
translational air-resistance term.

## Process systems

Process examples use their own engineering scales. The functions operate on
numeric values and do not use a unit-conversion library. Preserve the scale
of the chosen parameter set and input values, including the time unit.

### Johansson four-tank process

```{figure} _static/models/johansson.png
:alt: Four interconnected water tanks supplied by two pumps and two splitting valves, with upper tanks draining into lower tanks.
:width: 460px
:align: center

Four-tank process with crossed upper-tank pump feeds.
```

`johansson` has four liquid heights `[h1, h2, h3, h4]` in cm. Gravity is
converted internally to cm/s². Use compatible tank areas `A`, outlet areas
`a` and pump gains `K` with this centimetre/second scale.

The numerical interface takes `u=[pump1, pump2]` (0–100%) and a separate
`gamma=[valve1, valve2]` (fractions 0–1). The symbolic function takes a
single four-element vector:

```text
u = [pump1, pump2, valve1, valve2]
```

Pump 1 sends `(1-gamma1)` to tank 1 and `gamma1` to tank 4; pump 2 sends
`(1-gamma2)` to tank 2 and `gamma2` to tank 3. Tank 3 drains into tank 1 and
tank 4 into tank 2. Check this valve convention when adapting an experiment
from a different reference.

Heights used in the flow equations are clipped between a small positive
value and `h_max`. At a full tank, a positive derivative is suppressed.
This does not project the solver's state back into the permitted interval;
choose integration tolerances appropriately. The model docstring recommends
`rtol <= 1e-6` for saturation behaviour. Input and valve ranges are not
automatically enforced.

### Batch bioreactor

```{figure} _static/models/batch-bioreactor.png
:alt: Stirred bioreactor with a feed stream, biomass, substrate and product concentrations, and variable liquid volume.
:width: 266px
:align: center

Fed batch bioreactor with feed-flow input and variable liquid volume.
```

`batch_bioreactor` has biomass `X`, substrate `S` and product `P`
concentrations, documented in mol/L, followed by liquid volume `V`,
documented in m³. Input `u` is feed flow in m³/min. Use a minute-based
parameter set for the kinetic rates. `S_in` is feed substrate concentration.
Growth follows the substrate-inhibited Haldane expression using `mu_max`,
`K_m` and `K_i`; `v`, `Y_x` and `Y_p` control production and yields.

The equations contain `u/V`, so a physically positive volume is necessary.
Concentrations and volume are clipped to a small positive value for evaluating
the equations, and outward derivatives at the lower boundary are suppressed.
These guards do not make an arbitrary initial condition physical or remove
the need for appropriate solver tolerances.

### Continuous stirred-tank reactor

```{figure} _static/models/cstr.png
:alt: Stirred tank with feed and outlet streams, species A and B, and a cooling jacket exchanging heat with the reactor.
:width: 410px
:align: center

Continuous stirred-tank reactor and cooling jacket.
```

`cstr` has concentrations `[Ca, Cb]`, documented in mol/L, and reactor/jacket
temperatures `[Tr, Tk]` in °C. The Arrhenius expressions add 273.15 internally.
Time is in hours. Input `u=[F, dQ]` contains the dilution rate `F` in h⁻¹
and the signed jacket heat flow `dQ` in kJ/h. To obtain `F`, divide the
volumetric inlet flow in L/h by reactor volume `V` in L. To obtain `dQ`
from a thermal power in kW, multiply by 3600. Positive `dQ` heats the jacket
and negative `dQ` cools it. The second-order reaction coefficient `K0_ad`
is in L/(mol·h).

For the CSTR animation, pass the volumetric flow `F*V` in L/h and the
model heat flow `dQ` in kJ/h. The viewer displays the heat flow in kW.

Concentration evaluations use a small positive floor, with lower-bound
derivative guards. Temperatures are not clipped; keep absolute temperature
away from zero and use positive physical volumes, masses and heat capacities.
Boundary clipping creates nonsmooth regions, which matter when linearizing.

### Gas-lift oil well

```{figure} _static/models/oil-well.png
:alt: Gas-lift oil well showing annulus, tubing, injection valve, production and lift chokes, pressures and mixture flows.
:width: 621px
:align: center

Gas-lift well, manipulated choke openings and measured process quantities.
Source: [Jahanshahi et al., 2012](https://doi.org/10.3182/20120710-4-SG-2026.00110).
```

`oil_well` has 11 states. The first three states are differential
masses; the remaining eight are algebraic intermediate quantities:

| Index | State | Meaning | Unit |
| --- | --- | --- | --- |
| 0 | `x0` | Gas mass in annulus | kg |
| 1 | `x1` | Gas mass in tubing | kg |
| 2 | `x2` | Liquid mass in tubing | kg |
| 3 | `w_G_in` | Gas inflow into annulus | kg/s |
| 4 | `P_at` | Annulus top pressure | bar |
| 5 | `P_ab` | Annulus bottom pressure | bar |
| 6 | `P_bh` | Bottom-hole pressure | bar |
| 7 | `P_tt` | Tubing top pressure | bar |
| 8 | `w_out` | Produced mixture flow | kg/s |
| 9 | `rho_m_tt` | Tubing-top mixture density | kg/m³ |
| 10 | `alpha_L_tt` | Tubing-top liquid volume fraction | dimensionless |

Input `u=[production_choke, gas_lift_choke]` contains opening fractions
(0–1). `P_gs`, `P_out` and `P_res` are supplied in bar and converted
internally to Pa; algebraic pressure states remain in bar. Other documented
units include K for temperatures, m and m³ for lengths/volumes, kg/kmol for
gas molecular weight and Pa s for liquid viscosity.

`DAEFunction.n_diff == 3`; derivatives of the eight algebraic states are not
independent dynamics. A DAE initial condition needs consistent masses,
algebraic values and mass derivatives, or suitable guesses for a solver that
computes consistency.

The correlations require physically meaningful gas volumes, densities and
Reynolds numbers. The symbolic implementation floors Reynolds numbers at 1
in the friction-factor expressions; the numerical implementation does not.
Avoid extrapolating either backend to invalid or stagnant-flow states. The
numerical model docstring recommends `rtol <= 1e-8` near equilibrium when
algebraic variables require high accuracy.

### pH neutralization

```{figure} _static/models/neutralization.png
:alt: Neutralization reactor with acid, buffer and alkaline inlet streams, liquid-height measurement and an outlet pH sensor.
:width: 560px
:align: center

Neutralization process with acid feed, buffer disturbance and controlled alkaline flow.
```

`neutralization` has `[Wa4, Wb4, h, pH]`: two outlet reaction-invariant
concentrations, liquid height and outlet pH. The first three states are
differential and pH is algebraic (`DAEFunction.n_diff == 3`).
`x_dot[3]` does not enter the model equations. Reaction invariants may be negative;
do not clip them as if they were ordinary nonnegative species concentrations.

Input `u=q3` is the alkaline flow. Acid flow `q1` is a fixed parameter and
buffer flow `d=q2` is a disturbance parameter. Supply all flows in mL/s,
equivalent to cm³/s. Time is in seconds, invariant concentrations in mol/L,
`h` and `z` in cm, and area `A` in cm². The coefficient `Cv4` has units
of mL/(s·cmⁿ), so the outlet flow `Cv4*(h+z)**n` is also in mL/s.

The model requires positive liquid height and a valid real-valued outlet-flow expression
`Cv4*(h+z)**n`. The initial pH must satisfy the nonlinear algebraic
constraint; choosing a zero derivative for pH does not by itself satisfy that
constraint.

## Illustration sources

The schematics above are reproduced from the bundled
[numerical example notebook](https://github.com/ggonzm/benchmark-systems/blob/main/examples/simulations.ipynb),
except for the DC motor illustration, which uses the library's animation
helper. External sources credited in the notebook are preserved in the
quadrotor and oil-well captions.
