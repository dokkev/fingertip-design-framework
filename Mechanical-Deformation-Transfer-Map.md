# Mechanical Deformation Transfer Map

## Scope

Phase 4K constructs the nonlinear contact-to-observation deformation transfer
map (CODTM)

```text
(xi_cmd, delta_n) -> [b_left(eta), b_right(eta), F_n,
                      contact_length, xi_centroid]
```

from the adopted Phase 4J no-void external-contact model. It does not alter the
mixed T3 solid, material, mesh sizing, ALM contact, convergence settings, or
global loading direction. It does not run internal contact, optical simulation,
void design, or geometry optimization.

A compliance gradient describes local compliance inside a structure. CODTM is
the complete nonlinear map from a contact input to camera-side mechanical
deformation. Its local transfer Jacobian,

```text
J_T = [partial sidewall signature / partial xi,
       partial sidewall signature / partial delta_n],
```

is only a local linearization of CODTM; it is not a compliance gradient.

## Reference coordinates

- `xi=0`: right bonded endpoint of the undeformed semantic `PadOuterArc`.
- `xi=0.5`: crown.
- `xi=1`: left bonded endpoint.
- `delta_n`: displacement of the rigid indenter in the unchanged Phase 4J
  global loading direction.
- Right observation side: full-arc `xi=0 -> 0.25`.
- Left observation side: full-arc `xi=1 -> 0.75`.
- On both observation sides, `eta=0` is the bonded endpoint and `eta=1` is
  crownward.

Each observation side uses 41 fixed reference-arc samples. The samples are
linear Line2 shape-function interpolants, not selected mesh nodes. The primary
channel is

```text
b(eta) = u(X0(eta)) dot n0(eta),
```

where `n0` is selected by a reference-material probe so outward bulging is
positive on both sides. Global `ux`, `uy`, tangential displacement, reference
coordinates, and deformed coordinates are retained.

## Kratos contact-variable contract

The installed runtime is Kratos
`10.3.0-14ee273e-Release-x86_64`. Source and runtime inspection established:

- `LAGRANGE_MULTIPLIER_CONTACT_PRESSURE`: historical scalar slave-node ALM
  unknown.
- `AUGMENTED_NORMAL_CONTACT_PRESSURE`: non-historical nodal effective pressure;
  negative is compression.
- `NODAL_AREA`: non-historical nodal mortar weight.
- `NORMAL`: historical unit slave normal.
- `CONTACT_FORCE`, `CONTACT_PRESSURE`, `NORMAL_CONTACT_STRESS`, and
  `CONTACT_NORMAL`: registered but not stored on the actual external slave
  nodes in this formulation.

The diagnostic global-normal resultant follows the installed ALM process'
`NODAL_AREA * AUGMENTED_NORMAL_CONTACT_PRESSURE` convention, with the contact
normal projected onto the unchanged global load direction. `F_n` remains the
canonical prescribed-indenter reaction. Centroid and 2D active length are
exposed as verified only when

```text
abs(F_contact - F_n) / max(abs(F_n), force_floor) <= 0.02.
```

The pressure floor is explicit:
`numerical_force_tolerance / (reference PadOuterArc length * 1 mm thickness)`.
Failed closure keeps raw nodal diagnostics but publishes no trusted centroid or
length. The length is a plane-strain contact length, never a 3D contact area.

## Runs and results

All cases used `1.5 mm / 48` steps, the Phase 4J solver configuration, and a
fresh `Kratos.Model`.

| Case | Solve | Existing case gate | Final reaction (N) | Verified centroid | Verified length (mm) | Final closure | Min det(F) |
|---|---|---|---:|---:|---:|---:|---:|
| medium `xi=.20` | PASS | FAIL | 0.132235 | 0.173066 | 2.34749 | 0.133% | 0.927521 |
| medium `xi=.35` | PASS | PASS | 0.432058 | 0.328307 | 2.93384 | 0.0129% | 0.873200 |
| medium `xi=.50` | PASS | PASS | 0.861926 | 0.500159 | 3.32329 | 0.508% | 0.762817 |
| medium `xi=.65` | PASS | FAIL | 0.422103 | unavailable | unavailable | 4.327% | 0.855006 |
| medium `xi=.80` | PASS | FAIL | 0.130621 | unavailable | unavailable | 2.286% | 0.927267 |
| fine `xi=.20` | PASS | PASS | 0.133677 | 0.172961 | 2.13546 | 0.0317% | 0.909885 |
| fine `xi=.50` | PASS | PASS | 0.864680 | 0.500039 | 3.42512 | 0.0675% | 0.698391 |
| fine `xi=.80` | PASS | FAIL | 0.129790 | unavailable | unavailable | 4.702% | 0.909893 |

All eight logical cases reached 48 converged steps with finite displacement
fields and positive `det(F)`. The legacy case gate failures are preserved:
`xi=.20` medium misses the 2% support/indenter equilibrium gate only at step 1;
right-side cases also have unverified pressure-resultant closure. The queue
continued after every such terminal case result.

The K2 center cases reproduce the immutable Phase 4J J1/J2 final reaction and
minimum `det(F)` exactly. The measured target coordinate is
`xi=0.5000000000000002`; it was not assumed.

## Mechanical signature results

At `delta_n=1.5 mm`, the medium indentation-normalized signature gains for
`xi=[.20,.35,.50,.65,.80]` are

```text
[0.27158, 0.33281, 0.21803, 0.33238, 0.27149].
```

Fixed-indentation off-diagonal signature distances grow from
`0.0351–0.1244 mm` at `0.25 mm` to `0.2654–0.9821 mm` at `1.5 mm`.
Amplitude-normalized shape distances span `0.5526–1.9685` at `1.5 mm`.
These are descriptive separability results, not an observability PASS:
there is no optical/noise model.

Fixed-force comparison uses five levels over the common
`0.009019–0.130621 N` range. It performs piecewise-linear interpolation at the
first crossing on each unmodified loading path; every location has exactly one
crossing at every selected force. Off-diagonal distance spans
`0.00890–0.05031 mm` at the lowest force and `0.13495–0.75230 mm` at the
highest.

The mean-centered five-location signature singular values at `1.5 mm` are

```text
[5.5853, 2.4213, 0.6252, 0.3407, ~0] mm.
```

They describe four nonzero sampled location modes; they are not a noise-based
rank or an optical observability criterion.

## Mesh dependence

At `xi=.20/.50/.80`, medium/fine final reaction differences are
`1.078%`, `0.319%`, and `0.641%`. At `1.5 mm`, normal-profile relative L2
differences are `0.635%`, `0.255%`, and `0.659%`; maximum absolute differences
are `0.00549`, `0.00344`, and `0.00592 mm`. Normalized-shape correlations all
exceed `0.99996`, while transfer-gain differences are `0.527%`, `0.0546%`, and
`0.555%`.

These numerical comparisons are favorable, but CODTM mesh convergence remains
`PROVISIONAL` because no CODTM-specific profile threshold was declared before
the runs. Contact centroid/length convergence at `xi=.80` is unavailable
because force closure failed on both meshes.

## Verdict and artifacts

- CODTM extraction pipeline: **PASS**.
- Center baseline reconstruction: **PASS**.
- Medium location map: **PARTIAL** (right-side contact descriptors unverified).
- Fine spot checks: **PARTIAL** (right-side contact descriptors unverified).
- CODTM mesh convergence: **PROVISIONAL**.
- Contact-distribution force closure: **PARTIAL** (`256/384` load-bearing
  records verified).
- Mechanical separability: **DESCRIPTIVE ONLY**.

Artifacts are under `output/phase4_mechanical_transfer_map/`. The primary
machine-readable files are `codtm_long.csv`, `codtm_arrays.npz`,
`map_metadata.json`, `source_trace.json`, `validation.json`, and
`summary.json`. Before geometry optimization, define an optical forward/noise
model and a predeclared CODTM-specific mesh criterion.
