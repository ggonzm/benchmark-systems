from pathlib import Path
import re


def test_release_workflow_pins_supported_pdm_version():
    workflow = (
        Path(__file__).parents[1] / ".github" / "workflows" / "python-publish.yml"
    ).read_text()

    match = re.search(r"pdm==(\d+)\.(\d+)\.(\d+)", workflow)
    assert match is not None, "Release workflow must pin an explicit PDM version"

    version = tuple(int(part) for part in match.groups())
    assert version >= (2, 29, 7)
