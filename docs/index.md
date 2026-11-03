# benchmark-systems

**Nonlinear benchmark models for simulation and control design.**

`benchmark-systems` provides numerical and symbolic models of mechanical
systems, electrical drives and process systems. Use NumPy-based functions for
simulation, CasADi factories for derivatives and linearization, and Matplotlib
helpers to animate complete trajectories.

The library is developed by the
[SUPPRESS Research Group](https://suppress.unileon.es/en/) at the
[Universidad de León](https://www.unileon.es/).

## Start with a complete experiment

1. [Install the package](installation.md) and the optional dependencies you need.
2. [Simulate a pendulum](quickstart.md), inspect its states and plot the result.
3. [Animate the trajectory](animations.md) or
   [linearize a symbolic model](symbolic.md).

## Choose a model and a workflow

| Goal | Start here |
| --- | --- |
| Find state order, inputs and conventions | [Model catalog](models.md) |
| Integrate an ODE or an implicit DAE | [Numerical simulation](numerical.md) |
| Compute Jacobians or use CasADi integrators | [Symbolic models](symbolic.md) |
| Visualize a completed experiment | [Trajectory animations](animations.md) |
| Look up a function signature or parameter | [API reference](api/index.md) |

Numerical models are ordinary Python callables. Symbolic models are factories
that return CasADi functions or DAE wrappers. These interfaces share model
equations, but their calling conventions differ; the guides show both.

```{toctree}
:maxdepth: 2
:caption: User guide
:hidden:

installation
quickstart
models
numerical
symbolic
animations
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

api/index
changelog
```
