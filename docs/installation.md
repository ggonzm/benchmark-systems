# Installation

## Requirements

The package requires **Python 3.13 or later**, preferably in a virtual environment. The numerical package
depends on NumPy, SciPy and `solve-dae`.

```shell
python -m venv .venv
```

`-m venv` creates an isolated Python environment in the `.venv` directory.
The environment is activated before installing packages:

```shell
# Linux and macOS
source .venv/bin/activate
```

`source` activates the environment in Linux and macOS shells. On Windows,
use the PowerShell activation script instead:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

After activation, `python` and `pip` refer to this environment rather than
the system installation.

## Install from PyPI

The package is available on the
[Python Package Index (PyPI)](https://pypi.org/project/benchmark-systems/).
The numerical package can be installed with:

```shell
python -m pip install benchmark-systems
```

`python -m pip` runs the installer for the active Python environment. This
command installs the numerical models and resolves their NumPy, SciPy and
`solve-dae` dependencies automatically from PyPI; no separate dependency
installation or alternate index is needed. Check the installed version when
reproducing a published experiment:

```python
import benchmark_systems

print(benchmark_systems.__version__)
```

`__version__` reports the version of the package imported by the active
Python environment.

## Optional dependencies

| Feature | Extra | Additional dependency |
| --- | --- | --- |
| Symbolic model factories and CasADi integration | `sym` | CasADi |
| Trajectory animations | `anim` | Matplotlib |
| Both features | `all` | CasADi and Matplotlib |

An extra includes its optional dependencies in the same installation. For symbolic models and CasADi integration:

```shell
python -m pip install "benchmark-systems[sym]"
```

`[sym]` installs CasADi alongside the numerical package. Quoting the
requirement protects its square brackets from shell expansion.

For trajectory animations:

```shell
python -m pip install "benchmark-systems[anim]"
```

`[anim]` adds Matplotlib for the animation helpers. Symbolic models remain
optional when only numerical simulation and animation are needed.

For both features:

```shell
python -m pip install "benchmark-systems[all]"
```

`[all]` includes CasADi and Matplotlib. This single command is sufficient for
the numerical, symbolic and animated examples in these guides.

The numerical package can be imported without either optional dependency.
For GIF export, install Pillow; MP4 export requires an available video writer
such as FFmpeg.

## Install the current source

For a source checkout, the package is installed from the repository root. This
installs dependencies through the configured package index:

```shell
python -m pip install ".[all]"
```

The dot in `.[all]` selects the current directory; `[all]` includes the
optional dependencies. An editable installation is useful when changing the
package:

```shell
python -m pip install -e ".[all]"
```

`-e` links the installed package to this checkout, so edits to its Python
source are picked up without reinstalling it.

The [repository](https://github.com/ggonzm/benchmark-systems) also contains
Jupyter notebooks under `examples/` for numerical, symbolic and animated
experiments. A notebook environment is separate from the package's runtime
dependencies.
