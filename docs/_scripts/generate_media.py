"""Regenerate guide illustrations by executing the examples printed in the guides.

Run from the repository root:
    pdm run python docs/_scripts/generate_media.py
"""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

DOCS = Path(__file__).resolve().parents[1]
BLOCK = re.compile(r"^```\{testcode\} +([^\n]+)\n(.*?)^```[ \t]*$", re.M | re.S)


def execute_examples():
    """Keep each doctest group's namespace, just as Sphinx does."""
    groups = {}
    for filename in ("quickstart.md", "numerical.md", "symbolic.md", "animations.md"):
        source = (DOCS / filename).read_text(encoding="utf-8")
        for match in BLOCK.finditer(source):
            name, code = match.groups()
            namespace = groups.setdefault(name.strip(), {"__name__": "__main__"})
            with redirect_stdout(StringIO()):
                exec(compile(code, str(DOCS / filename), "exec"), namespace)
    return groups


def main():
    output = DOCS / "_static"
    output.mkdir(exist_ok=True)
    matplotlib.rcParams["svg.hashsalt"] = "benchmark-systems-docs"
    warnings.filterwarnings(
        "ignore", message="FigureCanvasAgg is non-interactive, and thus cannot be shown",
        category=UserWarning,
    )
    groups = execute_examples()
    figures = [
        ("quickstart", "fig", "pendulum-states.svg"),
        ("numerical-ode", "ode_figure", "numerical-ode.svg"),
        ("numerical-piecewise", "piecewise_figure", "numerical-piecewise.svg"),
        ("numerical-dae", "dae_figure", "numerical-dae.svg"),
        ("symbolic-quadrotor", "quadrotor_figure", "quadrotor-states.svg"),
        ("symbolic-quadrotor", "quadrotor_control_figure", "quadrotor-inputs.svg"),
    ]
    for group, variable, filename in figures:
        groups[group][variable].savefig(output / filename, metadata={"Date": None})
    for group, variable, filename, fps in [
        ("pendulum-animation", "animation", "pendulum.gif", 25),
        ("symbolic-quadrotor", "quadrotor_animation", "quadrotor.gif", 20),
    ]:
        groups[group][variable].save(str(output / filename), writer=PillowWriter(fps=fps), dpi=80)
    plt.close("all")
    print(f"Generated {len(figures)} plots and 2 animations in {output}")


if __name__ == "__main__":
    main()
