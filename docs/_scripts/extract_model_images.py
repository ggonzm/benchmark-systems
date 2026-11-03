"""Reproduce catalogue illustrations from existing repository material.

Run from any directory with benchmark-systems and its anim extra installed:
    python docs/_scripts/extract_model_images.py

Notebook attachments are copied byte for byte. The DC motor illustration is
rendered from the public Matplotlib animator because the numerical example
notebook does not contain a separate motor attachment.
"""

import base64
import json
from math import pi
from pathlib import Path


DOCS = Path(__file__).resolve().parents[1]
NOTEBOOK = DOCS.parent / 'examples' / 'simulations.ipynb'
OUTPUT = DOCS / '_static' / 'models'

# Keys are catalogue asset names; values identify the notebook cell and its
# attached image. Indices are zero-based Jupyter cell indices.
ATTACHMENTS = {
    'pendulum': (6, 'image-2.png'),
    'cart-pendulum': (9, 'image-4.png'),
    'furuta-pendulum': (11, 'image.png'),
    'double-cart-pendulum': (14, 'image-3.png'),
    'multilevel-cart-pendulum': (17, 'image-3.png'),
    'multimass-spring': (20, 'image-2.png'),
    'johansson': (24, 'image.png'),
    'batch-bioreactor': (28, 'image.png'),
    'cstr': (31, 'image.png'),
    'quadrotor': (35, 'image.png'),
    'oil-well': (39, 'image.png'),
    'neutralization': (44, 'image-2.png'),
}


def render_motor(path):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from benchmark_systems.animations import DCMotor

    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11}):
        fig, axis = plt.subplots(figsize=(6, 3.2), dpi=180)
        DCMotor(axis, theta=[pi / 4], omega=[2.0], current=[0.5])
        fig.savefig(path, bbox_inches='tight', pad_inches=0.12, facecolor='white')
        plt.close(fig)


def main():
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, (cell_index, filename) in ATTACHMENTS.items():
        try:
            payload = notebook['cells'][cell_index]['attachments'][filename]['image/png']
        except (IndexError, KeyError) as exc:
            raise ValueError(f'Expected attachment {filename!r} in cell {cell_index}') from exc
        data = base64.b64decode(payload, validate=True)
        if not data.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ValueError(f'Attachment for {name} is not a PNG image')
        (OUTPUT / f'{name}.png').write_bytes(data)
    render_motor(OUTPUT / 'dc-motor.png')
    print(f'Wrote {len(ATTACHMENTS) + 1} catalogue illustrations to {OUTPUT}')


if __name__ == '__main__':
    main()
