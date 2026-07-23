# Commands

Use the Kratos interpreter for solver-backed commands:

```bash
LIT_PYTHON=/home/dk/miniconda3/envs/lit/bin/python
```

## Tests

```bash
$LIT_PYTHON -m pytest tests/unit -q
$LIT_PYTHON -m pytest tests/smoke -q -m "smoke and not kratos"
$LIT_PYTHON -m pytest tests/smoke -q -m kratos
```

## Validation

```bash
$LIT_PYTHON -m validation.benchmarks.volumetric_locking run \
  --output output/validation/benchmarks/volumetric_locking.json
$LIT_PYTHON -m validation.benchmarks.mixed_volumetric run \
  --output output/validation/benchmarks/mixed_volumetric.json
$LIT_PYTHON -m validation.fingertip.mesh --levels medium fine \
  --output-directory output/validation/fingertip/mesh
$LIT_PYTHON -m validation.fingertip.indentation.no_void
$LIT_PYTHON -m validation.fingertip.transfer_map \
  --output-dir output/validation/fingertip/transfer_map \
  --reference-dir output/validation/fingertip/indentation/no_void
```

## Figures

```bash
$LIT_PYTHON -m visualization examples/transfer_map_comparison.yaml \
  --output-dir output/figures/transfer_map_comparison
$LIT_PYTHON -m visualization examples/displacement_vector_atlas.yaml \
  --output-dir output/figures/displacement_vector_atlas
```

All generated validation artifacts and figures are written below `output/`.
