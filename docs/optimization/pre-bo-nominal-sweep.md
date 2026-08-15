# Pre-BO nominal morphology sweep

## Purpose and provenance

This exploratory sweep was run on 2026-08-15 from repository revision
`117723b`. It evaluates the current nominal LIT morphology separately and
then samples a fixed six-dimensional morphology space before any Bayesian
optimization campaign. The sweep does not change `FingertipParameters()`
defaults, optimization bounds, or optimizer state.

The nominal reference trial used `FingertipParameters()`:

| Parameter | Value (mm) |
| --- | ---: |
| `flat_pad_width` | 30.0 |
| `flat_pad_height` | 5.0 |
| `semielliptical_pad_height` | 9.0 |
| `stem_width` | 7.6 |
| `stem_height` | 6.0 |
| `void_width` | 1.0 |
| `void_height` | 0.0 |

Other mechanical and numerical defaults were unchanged. The nominal trial
had minimum separability `0.0751066215`, side ligament `10.2 mm`, distal
ligament `7.5267578833 mm`, and minimum conservative ligament
`7.5267578833 mm`.

## Sampling and admissibility

The flat-pad width was fixed at `30.0 mm`. The six swept parameters were
sampled in canonical order using scrambled SciPy Sobol points with seed
`20260815`, `random_base2(m=6)`, for exactly 64 points without replacement.

| Parameter | Lower (mm) | Upper (mm) |
| --- | ---: | ---: |
| `flat_pad_height` | 3.5 | 6.5 |
| `semielliptical_pad_height` | 7.0 | 11.0 |
| `stem_width` | 6.5 | 9.0 |
| `stem_height` | 5.0 | 7.5 |
| `void_width` | 0.5 | 2.0 |
| `void_height` | 0.0 | 1.5 |

The shared conservative geometry rule computes:

```text
half_width = flat_pad_width / 2
cutout_half_width = stem_width / 2 + void_width
cutout_depth = stem_height + void_height
ellipse_depth_at_cutout = semielliptical_pad_height
    * sqrt(1 - (cutout_half_width / half_width)^2)
side = half_width - cutout_half_width
distal = flat_pad_height + ellipse_depth_at_cutout - cutout_depth
minimum = min(side, distal)
```

A candidate is admissible only when both `side` and `distal` are at least
`2.0 mm`. These are conservative design-space measures, not an exact
minimum Euclidean wall-thickness calculation. Invalid candidates are
checkpointed and never sent to FEM.

## Evaluation protocol

Each candidate used the direct `DesignEvaluator` with the existing protocol:

- scenario locations `(-3.0, +3.0) mm`;
- indentations `(0.5, 1.0) mm`;
- indenter radius `4.0 mm`;
- four scenario combinations and four internal-contact comparison pairs;
- medium mesh;
- default `TraceSettings`, LED, and optical material;
- `fem_steps=48` and `internal_contact="three_pairs"`.

Candidates were evaluated serially, each in its own child process with a
`1800 s` timeout and no retry. The parent process owned progress and
checkpoint updates. The checkpoint was written after the nominal trial and
after every attempted candidate. A same-configuration rerun regenerated the
same Sobol points and skipped all completed indexes.

## Results

The run recorded `64` candidate proposals and `19,639.0170 s` total wall time
(approximately 5 h 27 min, including the separate nominal trial).

| Result | Count |
| --- | ---: |
| Successful candidates | 63 |
| Geometry-rejected candidates | 0 |
| FEM failures | 1 |
| Optics failures | 0 |
| Timeouts | 0 |
| Other process failures | 0 |
| Successful candidates better than nominal | 50 |

For successful candidates, the minimum-separability distribution was:

| Statistic | Value |
| --- | ---: |
| Minimum | 0.0586946895 |
| Median | 0.0945019999 |
| Maximum | 0.1273767967 |

The limiting axis was `location` for 61 successful candidates and
`indentation` for 2. Candidate 50 was geometrically admissible but its first
scenario, `(-3.0 mm, 0.5 mm, 4.0 mm)`, did not converge in FEM. It was recorded
as a FEM failure without an objective or retry, and the remaining candidates
continued.

### Top ten successful candidates

Parameters are shown as `(flat_pad_height, semielliptical_pad_height,
stem_width, stem_height, void_width, void_height)` in mm. `delta` is the
candidate minimum separability minus the nominal value.

| Candidate | Minimum separability | Delta | Parameters | Side / distal / minimum ligament (mm) | Limiting axis |
| ---: | ---: | ---: | --- | --- | --- |
| 49 | 0.1273767967 | +0.0522701753 | (3.9372, 7.3098, 7.2899, 5.1023, 0.6932, 1.2691) | 10.6619 / 4.5632 / 4.5632 | location |
| 25 | 0.1247773117 | +0.0496706902 | (3.8060, 7.0420, 6.8995, 6.4409, 1.3757, 0.2131) | 10.1746 / 3.8196 / 3.8196 | location |
| 5 | 0.1247318529 | +0.0496252314 | (4.4753, 8.6790, 6.5019, 6.6271, 0.7432, 1.4256) | 11.0059 / 4.7883 / 4.7883 | location |
| 19 | 0.1221557551 | +0.0470491334 | (5.3382, 7.3544, 8.7502, 6.6447, 1.8076, 1.3645) | 8.8173 / 4.0296 / 4.0296 | location |
| 1 | 0.1210024159 | +0.0458957944 | (4.1790, 7.5294, 8.6845, 5.4179, 1.0348, 1.1898) | 9.6229 / 4.6002 / 4.6002 | location |
| 53 | 0.1204143915 | +0.0453077709 | (4.4320, 8.3950, 7.9002, 6.3104, 0.9816, 1.3174) | 10.0683 / 4.7325 / 4.7325 | location |
| 41 | 0.1198348423 | +0.0447282208 | (3.5777, 7.7595, 8.1379, 6.7502, 1.8990, 0.3268) | 9.0320 / 3.6196 / 3.6196 | location |
| 35 | 0.1192997393 | +0.0441931178 | (5.0045, 7.5721, 7.5276, 6.3364, 1.4644, 1.4735) | 9.7718 / 4.2919 / 4.2919 | location |
| 60 | 0.1138341216 | +0.0387275001 | (4.5262, 9.1949, 7.4397, 6.8839, 1.8258, 1.2062) | 9.4543 / 4.9795 / 4.9795 | location |
| 9 | 0.1137400106 | +0.0386333891 | (3.6097, 8.3214, 7.5699, 5.7109, 1.6793, 1.0058) | 9.5357 / 4.6426 / 4.6426 | location |

The best observed sampled point was candidate 49. This single Sobol sweep is
not sufficient to establish a new nominal morphology or revised optimization
bounds. The only failed candidate does not establish a general failure region;
it is retained as a FEM follow-up observation.

## Observed trends and follow-up regions

The top ten contain several lower flat-pad-height and semielliptical-pad-height
points, but this descriptive pattern is not a causal sensitivity result. Their
limiting comparison is overwhelmingly the location axis. The sampled points
also remained comfortably above the 2.0 mm conservative ligament threshold;
there were no geometry rejections in this particular Sobol draw.

Candidate 49 is the promising sampled point for follow-up because it had the
highest successful minimum separability. Candidate 50 is the failure-prone
follow-up point observed in this run: it was admissible, but FEM did not
converge at the `(-3.0 mm, 0.5 mm, 4.0 mm)` scenario. One failure is not enough
to define a general failure region or to narrow the proposed BO bounds.

## Decision

Recommended nominal candidate: **pending scientific review**. Candidate 49 is
the best observed successful point and should be reviewed first, but the
current `FingertipParameters()` nominal defaults remain unchanged.

Proposed Bayesian-optimization ranges: **pending scientific review**. No BO
campaign or new bounds were created from this exploratory result.

The main reasons for deferring the decision are the single FEM non-convergence,
the limited 64-point sample, and the need to review optical, mechanical, and
manufacturing implications together with the conservative ligament rule.

Raw artifacts are under
`output/validation/optimization/pre_bo_nominal_sweep/`, including
`checkpoint.json`, `summary.json`, per-candidate inputs/results, and child
logs. The runner does not write this document automatically.

This sweep was exploratory and is not part of the Bayesian optimization
campaign.
