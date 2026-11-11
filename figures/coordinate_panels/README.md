# Orthogonal coordinate panels

Small NumPy/Matplotlib functions for a rectangular source grid, its
polar-to-Cartesian image, and a Swiss-roll orthogonal coordinate transformation.

The first two panels reproduce the plotting choices in
[`figures/Figure_radial_map.ipynb`](https://github.com/lgresele/independent-mechanism-analysis/blob/master/figures/Figure_radial_map.ipynb).
The source grid's negative angular tick is corrected. The original notebook
does not need to be modified, and JAX is not required by this script.

## Run

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python coordinate_panels.py
```

The script writes four vector PDFs and corresponding PNGs to `output/pdf/`:
`source_grid`, `observations_grid`, `swiss_roll`, and the assembled
`coordinate_panels`. Individual panels have no headings; headings in the
assembled preview identify the three examples.

The default view places the Swiss roll's near end toward the lower left and
its far end toward the upper right. The camera is adjustable:

```sh
python coordinate_panels.py --elev 22 --azim -62 --roll 0 --output output/pdf
```

## Geometry and rendering

`swiss_roll(s)` implements the global map
`f(s1, s2) = (r*cos(r), s2, r*sin(r))`, where `r = exp(s1)`.
Its Jacobian Gram matrix is `diag(r**2 * (1 + r**2), 1)`, and its smooth inverse
on the image is `(log(hypot(x1, x3)), x2)`. It is therefore an OCT; independent
latent coordinates make it an OCT-IC model.

The picture shows `3*pi/2 <= r <= 9*pi/2` and `-5 <= s2 <= 5`.
This is a separate source domain from the polar example on the left.
Blue curves vary the first source coordinate; magenta curves vary the second.
The Swiss-roll grid is sampled more densely along its long spiral than along
its depth, with rulings spaced by arc length to balance the visible cells.
Curve spacing is an illustration choice, not a specification of a latent density.

Orthographic ray intersections suppress curves hidden behind the surface.
The visibility calculation uses a dense polygonal approximation of the
spiral; displayed points are evaluated on the exact smooth surface. All
curves remain vector paths. Orthogonality holds in 3D, not generally in the
2D projection. Avoid a camera exactly along the extrusion axis.

Shared styling follows the original figures: DejaVu Sans, 13 pt tick labels,
14 pt axis labels, 1.5 pt coordinate lines, blue `#1A85FF`, magenta `#D41159`,
and white backgrounds. The preview was generated with Python 3.12,
NumPy 2.3.5 and Matplotlib 3.10.9.
