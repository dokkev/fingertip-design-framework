# Production optimized parameters

The production morphology vector has exactly four active variables:

```text
flat_pad_height
stem_width
stem_height
void_width
```

The fixed envelope is `flat_pad_width=30 mm`; the derived height is
`semielliptical_pad_height=14-flat_pad_height`. `void_height=0 mm`,
`basal_interface="bonded"`, and `internal_contact="sides_separate"` are fixed
production conditions. Material, bond, link, and LED parameters remain fixed.

The evaluator uses one monotonic 0-to-2 mm FEM trajectory with 48 steps for
each of the 12 diameter/location pairs, capturing depths 0.5, 1.0, 1.5, and
2.0 mm. It computes one unloaded reference and 48 loaded PLANAR_2D optical
states. The optimization objective is the minimum across the 12 depth-AUC
values of lateral `J_contact`; minimization callers use its negative.

See [optimized-morphology-parameters.md](optimized-morphology-parameters.md)
for the full protocol and [evaluation.md](evaluation.md) for the metric.
