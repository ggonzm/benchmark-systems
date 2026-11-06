# Changelog

## 1.2.0

- Correct the rotor torque projection and Coriolis coupling in the numerical
  and symbolic quadrotor DAEs.
- Correct Furuta pendulum dynamics and document its normalized motor command.
- Use linear viscous damping for the cart and joints in the multilevel
  cart-pendulum, with each joint acting on its relative angular velocity.
- Clarify CSTR dilution and heat-flow units, and correct the animation labels.
- Use mL/s for neutralization flow labels and correct the base-stream
  composition in the neutralization examples.
- Label Johansson pump commands and their branch shares as percentages.

## 1.1.2

- Update the publishing toolchain for compatibility with its HTTP dependencies.
- Support manually retrying publication of an existing GitHub release.

## 1.1.1

- Link the README and package metadata to the online Read the Docs documentation.

## 1.1.0

- Add installation instructions, runnable user guides, a model catalog and a
  reference for the numerical, symbolic and animation APIs.
- Add Sphinx configuration for local documentation builds and Read the Docs.

- Document installation from PyPI, including the `sym`, `anim` and `all` extras.
- Expand symbolic and animation API descriptions with units and data conventions.
- Add model illustrations and reproducible simulation plots and animations.

## 1.0.7

- Reorganize animation implementations into dedicated modules while preserving
  their public imports through `benchmark_systems.animations`.
- Support explicit sample times in `AutoAnimation` and retain the final frame
  when frames are skipped for playback.
- Validate animation trajectories and timing configuration.
- Update animation examples to use explicit timing.

See the [repository history](https://github.com/ggonzm/benchmark-systems/commits/main/)
for the complete record of changes.
