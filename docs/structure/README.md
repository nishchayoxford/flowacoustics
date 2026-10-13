# Editable project workflow

Open [flowacoustics-map.svg](flowacoustics-map.svg) in a browser or SVG editor. It is a standalone vector image with editable text, no linked fonts/images and no JavaScript.

The diagram separates the completed 459,264-cell, eight-rank, 10 s LES; saved and reconstructed fields; native Curle and FW-H alternatives; independent compact-force calculations; and analytical teaching examples. Curle computes its own loads. The independent force export feeds the Python comparison. Curle and FW-H are not added.

Reconstructed times are 2, 4, 6, 8 and 10 s. These completion labels describe execution, not experimental validation, resolved stationary shedding or a successful website deployment. See [methods](../CYLINDER_ACOUSTICS.md) and [limits](../OUTPUTS_AND_LIMITS.md).

Stable SVG IDs and `data-node`, `data-from` and `data-to` attributes make the graph auditable. Keep edges behind cards and preserve the physical/evidence distinctions when editing. The canonical SVG and the website copy must remain byte-identical.

From the repository root, run:

```sh
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/test_project_flowchart.py -q -p no:cacheprovider
```

The tests check graph structure and scientific boundaries and render the actual SVG in isolated Chromium, measuring every label against its card. Inspect the resulting `.cache/flowchart-verification/flowacoustics-map.png` after changes. Browser prerequisites are listed in [reproduction](../REPRODUCE.md).
