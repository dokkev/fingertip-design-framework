# Current iteration: OptiX sensor-facing design verification

## Task execution controls

Reviewer subagent: YES
Complete checklist: YES
Fix reviewer findings: YES
Expensive reruns: INVALIDATED_ONLY

---

# Goal

Implement and benchmark a bounded first-pass **sensor-facing 3D OptiX design-verification pipeline** for the LIT/LUMO fingertip.

The purpose of this iteration is not to redesign the 2D optimizer and not to turn OptiX into a general-purpose tactile renderer.

The decision question is:

> Does the morphology selected by the existing 2D design process preserve its optical-transport advantage under 3D propagation and produce more distinguishable camera-space contact-state observations than the nominal morphology?

For this first iteration, compare the existing **nominal** morphology against **candidate49**.

A negative or ambiguous result is a valid scientific outcome.

Do not modify the implementation merely to force candidate49 to outperform nominal.

---

# Scientific framing

Keep the following quantities conceptually separate.

## Reduced-model bridge

`P3_xy` / `J3D-path` are model-reduction verification quantities.

Their role is only to test whether morphology-dependent internal optical-transport trends identified in the 2D reduced model persist under 3D propagation.

They are not sensor observability metrics.

Do not change the existing 2D optimization objective in this task.

## Intrinsic 3D transport

The outgoing boundary transport should conceptually be represented by a phase-space quantity such as

`Q(surface_position, outgoing_direction[, source_id])`.

Any direction-marginalized surface quantity such as `M(u,z)` is an intrinsic optical-transport descriptor, not camera observability.

Preserve total outgoing energy separately from normalized spatial redistribution metrics.

Do not call surface exit flux itself "sensor observability."

## Sensor-facing response

Sensor-facing quantities must be computed only after applying a camera observation model/operator to the available outgoing transport data.

Conceptually:

`outgoing transport -> camera observation operator -> expected linear sensor response mu`

Fisher information and contact-state distinguishability in this task must operate on this camera-space response, not directly on `M`, `P3`, or normalized surface-flux fields.

---

# Scope

Implement the smallest repository-consistent pipeline required to:

1. obtain or construct a camera-space expected linear response for each requested contact state;
2. compute pairwise contact-state distinguishability;
3. compute local finite-difference image Jacobians with respect to contact state;
4. compute Fisher-information-derived metrics;
5. optionally marginalize a global photometric gain nuisance parameter;
6. benchmark nominal versus candidate49;
7. preserve machine-readable results and concise visual summaries under `output/`;
8. have an independent reviewer subagent inspect the implementation, benchmark evidence, and scientific claims.

Inspect the current repository before editing.

Reuse existing OptiX outputs, escape-event data, camera geometry, state definitions, benchmark infrastructure, and validation utilities when they already provide the required behavior.

Do not recreate functionality that already exists.

---

# Explicit non-goals

Do not modify or re-optimize the 2D morphology optimizer.

Do not change the existing 2D TV objective.

Do not perform a new multi-morphology optimization or broad morphology sweep.

Do not implement a fully coupled 3D mechanics model.

Do not describe the existing extruded 2D mechanical deformation as fully 3D optomechanics.

Do not add a general renderer, scene graph, material framework, plugin registry, generalized metrics framework, or compatibility layer unless the current production pipeline already requires one.

Do not implement full microfacet roughness, realistic LED package geometry, calibrated scattering, finite five-LED full-finger validation, or broad optical-parameter sensitivity studies in this iteration.

Those remain follow-up verification tasks unless existing infrastructure makes a limited comparison essentially free.

Do not rewrite unrelated production architecture.

Do not clean up unrelated code.

---

# Camera-space response

## Preferred path

If the repository already contains sufficient outgoing ray/escape-event information, construct the camera-space response downstream from those events rather than rerunning optical transport separately for every camera-space metric.

Reuse stored information such as, where available:

* exit position;
* outgoing direction;
* ray weight or power;
* source ID;
* wavelength/channel information;
* geometry state identifier.

Apply the smallest physically meaningful camera observation operator supported by the repository.

At minimum, camera acceptance should account for available geometry such as:

* camera pose;
* field of view;
* outgoing direction;
* sensor/lens acceptance or aperture when available;
* visibility/occlusion when already represented.

Do not silently integrate all outgoing flux and label the result a camera image.

## Missing physical camera parameters

If required physical camera parameters are not currently available in the repository, do not invent a calibrated real-camera model.

Implement the smallest explicit, configurable idealized observation model needed to exercise the analysis pipeline.

Clearly label results from such a model as an idealized camera-space or camera-projected response rather than experimentally validated sensor observability.

Report unavailable physical calibration as a limitation or UNCLEAR item where appropriate.

Do not block the entire implementation solely because calibration data are absent.

---

# Sensor response domain

Use an expected **linear sensor-domain response** for the analysis.

Prefer linear RAW-like intensity or the closest existing linear response representation in the repository.

Do not introduce nonlinear display processing, gamma correction, tone mapping, arbitrary contrast normalization, or image-by-image normalization before computing Fisher information.

Any global normalization that removes physically meaningful total-energy differences must be explicit and must not silently replace the unnormalized analysis.

---

# Contact state

For the first implementation, use the smallest mechanically justified local contact state supported by the existing extruded plane-strain model.

Preferred parameterization:

`theta = [x_c, delta]`

where:

* `x_c` is contact position in the modeled cross-sectional sensing direction;
* `delta` is indentation/deformation state already supported by the FEM/OptiX pipeline.

Use existing repository state coordinates and FEM states where possible.

Do not introduce unsupported 3D contact DOFs merely because Fisher information can mathematically accept them.

If the repository uses an equivalent physical state variable instead of `delta`, reuse that state variable and document the mapping.

---

# Pairwise camera-space distinguishability

Implement a direct pairwise response-separation metric alongside Fisher information.

For two expected sensor responses `mu_a` and `mu_b`, support a noise-normalized squared separation of the form

`d2 = (mu_a - mu_b)^T Sigma^-1 (mu_a - mu_b)`

or its diagonal-noise equivalent.

At minimum evaluate, where the available state grid permits:

* contact versus no-contact separation;
* neighboring indentation-state separation at fixed contact position;
* neighboring contact-position separation at fixed indentation.

Preserve these results separately.

Do not collapse all state comparisons into one scalar before the underlying distributions have been inspected.

---

# Noise model

Implement a simple explicit first-pass sensor noise model.

Prefer a diagonal covariance model compatible with a linear sensor response, for example:

`variance_p = alpha * mu_p + sigma_read^2`

when supported by the current response representation.

Keep noise-model parameters explicit and recorded in benchmark metadata.

If real camera noise calibration is unavailable, use clearly labeled nominal parameters for computational comparison only.

Do not claim calibrated absolute detectability from uncalibrated noise parameters.

Where useful, also preserve a noise-free response-difference result so that optical response differences can be separated from assumptions about camera noise.

---

# Finite-difference Jacobian

For local contact state `theta = [x_c, delta]`, compute the image Jacobian

`J = d mu / d theta`

using centered finite differences wherever neighboring states are available.

Conceptually:

`dmu/dx ~= [mu(x + dx, delta) - mu(x - dx, delta)] / (2 dx)`

`dmu/ddelta ~= [mu(x, delta + ddelta) - mu(x, delta - ddelta)] / (2 ddelta)`

Use one-sided differences only at unavoidable state-grid boundaries and label them accordingly.

Prefer derivatives from already-computed physical states rather than synthesizing interpolated states unless interpolation already exists and is scientifically justified.

Do not silently mix incompatible meshes, camera grids, exposure scales, or state definitions when forming derivatives.

---

# Fisher information

Compute the local Fisher information matrix from the camera-space response:

`F = J^T Sigma^-1 J`

For the two-state case this should be a 2 x 2 matrix.

The implementation must preserve the raw matrix or sufficient values to reconstruct it.

For every valid evaluated state, report at least:

* minimum eigenvalue;
* maximum eigenvalue;
* condition number;
* log determinant where numerically defined;
* CRLB covariance or equivalent inverse/pseudoinverse result;
* contact-position CRLB standard deviation in physical units;
* indentation/state CRLB standard deviation in physical units.

Do not silently regularize a singular or nearly singular Fisher matrix simply to obtain attractive finite numbers.

If regularization or a pseudoinverse is required for reporting, make that choice explicit and preserve a validity/rank indicator.

A singular or poorly conditioned state is scientifically meaningful.

---

# State scaling

Eigenvalue-, determinant-, and conditioning-based Fisher summaries are sensitive to parameter units and scaling.

Therefore maintain both:

1. physical-state coordinates for interpretable CRLB values; and
2. an explicitly documented dimensionless state scaling for cross-parameter conditioning/eigenvalue summaries.

For example, normalize position and indentation by fixed physical reference scales shared across nominal and candidate49.

Do not choose separate normalization factors per morphology.

Do not choose scaling after observing which choice makes candidate49 look better.

Record all reference scales in output metadata.

---

# Global photometric-gain nuisance

Implement an optional global photometric gain nuisance parameter if it can be supported without broad architecture changes.

Use a model equivalent to:

`mu(theta, g) = g * mu0(theta)`

or the closest appropriate linear form.

Construct the joint Fisher matrix for contact state and gain.

Partition it into contact and nuisance blocks and compute the effective contact information using the Schur complement:

`F_contact_eff = F_tt - F_tg * inv(F_gg) * F_gt`

or the numerically appropriate scalar/block equivalent.

Compare:

* contact Fisher information without the gain nuisance;
* effective contact Fisher information after marginalizing gain.

This is intended to distinguish genuine spatial/structural contact information from improvements caused primarily by global brightness changes.

Verify numerically that nuisance marginalization does not spuriously increase contact information beyond numerical tolerance.

If the nuisance block is singular or invalid, report the affected state rather than hiding it.

---

# Numerical sanity checks

Focused checks are explicitly authorized for this iteration.

Do not create a broad new test suite.

At minimum verify:

1. Fisher matrices are symmetric within numerical tolerance.
2. Fisher eigenvalues are nonnegative within expected floating-point tolerance.
3. noise variances are finite and positive where required.
4. camera responses being differenced use identical sensor grids and scaling.
5. nominal and candidate49 use identical benchmark settings.
6. dimensionless state scaling is identical across morphologies.
7. Schur-complement nuisance marginalization does not increase contact information except for negligible numerical error.
8. singular/near-singular states are surfaced explicitly.
9. no per-state image normalization accidentally removes total-energy information.
10. finite-difference derivatives are not dominated by an obvious implementation or state-ordering error.

Perform a bounded finite-difference step sanity check at representative interior states if the available state sampling allows it.

The purpose is to detect gross derivative instability, not to launch an open-ended convergence study.

---

# Benchmark

Run a focused design-verification benchmark for:

* nominal morphology;
* candidate49.

Use identical:

* FEM/contact states;
* OptiX settings;
* ray counts where new tracing is required;
* camera geometry;
* observation model;
* noise assumptions;
* state scaling;
* metric implementation.

Prefer reusing already-valid expensive OptiX/FEM artifacts.

Do not rerun expensive simulations merely to obtain prettier logs or reviewer evidence.

Rerun an expensive stage only if an implementation defect invalidates the corresponding scientific result, consistent with `Expensive reruns: INVALIDATED_ONLY`.

Use deterministic seeds/settings where stochastic sampling is involved and where the current infrastructure supports them.

---

# Benchmark outputs

Place generated files under an appropriate untracked directory such as:

`output/optix_design_verification/`

Preserve enough machine-readable information to inspect results without rerunning expensive simulation.

Prefer simple explicit formats already used by the repository.

The result set should contain, as applicable:

* benchmark configuration/metadata;
* morphology identifier;
* contact-state coordinates;
* camera/noise parameters;
* dimensionless state scales;
* pairwise detectability values;
* raw Fisher matrix entries;
* Fisher eigenvalues;
* minimum eigenvalue;
* condition number;
* log determinant;
* rank/validity status;
* physical-unit CRLB quantities;
* gain-marginalized Fisher quantities;
* total camera-space signal/power where useful;
* paths or identifiers of source simulation artifacts.

Generate concise plots only when useful for scientific interpretation.

Useful first-pass visualizations include:

* contact-position x indentation heatmap of minimum Fisher eigenvalue;
* contact-position CRLB heatmap;
* indentation CRLB heatmap;
* gain-marginalized equivalents;
* candidate49 / nominal information ratios where mathematically meaningful;
* pairwise state-separation heatmaps or distributions.

Visualization code belongs in the validation/analysis layer and must respect repository dependency rules.

---

# Primary benchmark comparisons

Do not judge the outcome from one scalar alone.

Compare nominal and candidate49 using at least the following perspectives:

## Camera-space state separation

Does candidate49 increase separation between physically neighboring contact states?

Inspect both typical and weak regions.

## Local information

Compare the local Fisher matrices across the supported contact-state domain.

Pay particular attention to minimum eigenvalue and CRLB rather than only trace or determinant.

## Conditioning

Determine whether either morphology creates contact-state directions that are poorly observable relative to others.

## Spatial/state coverage

Inspect the distribution over the contact-state domain.

Where enough states exist, report summary statistics such as median and a lower percentile in addition to the mean.

Do not claim broad uniform improvement from a small number of favorable states.

## Gain robustness

Determine whether any candidate49 advantage survives marginalization of global photometric gain.

A reduction or disappearance of the advantage is a valid and important result.

---

# Interpretation rules

The benchmark is intended to answer a design-verification question, not to prove calibrated real-sensor accuracy.

Allowed conclusions depend on the evidence.

If candidate49 preserves the expected transport trend and improves camera-space contact information under the implemented observation/noise assumptions, report that result.

If the advantage exists only in intrinsic transport but disappears after camera projection, report that result.

If the advantage is mostly explained by global brightness and disappears after gain marginalization, report that result.

If Fisher information is highly state-dependent or singular in parts of the sensing domain, report that result.

If candidate49 performs worse than nominal under the sensor-facing metrics, report the negative result and stop rather than modifying the benchmark to make candidate49 win.

Do not use the terms "observability," "detectability," "information," or "CRLB" beyond what the implemented observation and noise model actually supports.

Do not claim experimental sensor fidelity from an uncalibrated camera/noise model.

---

# Implementation checklist

Complete and report every item as PASS, FAIL, or UNCLEAR.

* [ ] Existing implementation and architecture inspected before editing.
* [ ] Existing 2D optimization path remains unchanged.
* [ ] Existing `J3D-path` role remains reduced-model/3D-transport verification only.
* [ ] Camera-space linear response is defined explicitly.
* [ ] Camera-space response is not silently replaced by total outgoing surface flux.
* [ ] Camera/noise assumptions are explicit and preserved in output metadata.
* [ ] Contact state uses mechanically supported variables.
* [ ] Pairwise camera-space contact-state separation is implemented.
* [ ] Finite-difference image Jacobian is implemented.
* [ ] Physical and dimensionless state parameterizations are distinguished.
* [ ] Fisher matrix is computed from camera-space response.
* [ ] Fisher symmetry/PSD sanity checks are performed.
* [ ] Minimum eigenvalue is reported.
* [ ] Condition number is reported.
* [ ] Log determinant is reported where valid.
* [ ] Physical-unit CRLB quantities are reported.
* [ ] Singular/near-singular states are explicitly identified.
* [ ] Global gain nuisance marginalization is implemented or a concrete blocker is reported.
* [ ] Gain-marginalized information passes basic consistency checks.
* [ ] Nominal and candidate49 are benchmarked using identical settings.
* [ ] Benchmark artifacts are saved under `output/` and remain untracked.
* [ ] Results include enough machine-readable data to inspect without repeating expensive simulations.
* [ ] Result interpretation distinguishes transport, camera-space response, and sensor information.
* [ ] Negative or ambiguous findings are preserved rather than optimized away.
* [ ] No unrelated architecture/refactor/compatibility work was introduced.
* [ ] Reviewer subagent independently inspected the final implementation and evidence.

---

# Reviewer subagent instructions

After implementation and the pre-review checklist are complete, invoke one reviewer subagent.

The reviewer is read-only.

The reviewer must inspect:

* `AGENTS.md`;
* this `INSTRUCTION.md`;
* relevant architecture documentation;
* the actual current implementation;
* the implementation diff;
* generated benchmark artifacts and metadata;
* focused test/check results;
* benchmark interpretation.

The reviewer must not rely solely on the implementation agent's summary.

The reviewer should behave as a skeptical scientific/software reviewer, with special attention to whether the implementation could produce a numerically plausible but scientifically invalid positive result.

The reviewer must explicitly evaluate the following.

## Scientific validity

* Is Fisher information computed from a legitimate camera-space response rather than intrinsic exit flux?
* Does the observation model actually use available outgoing-direction/camera geometry information?
* Are camera calibration limitations stated correctly?
* Are total-energy changes preserved rather than accidentally normalized away?
* Are contact-state variables compatible with the current extruded plane-strain mechanics?
* Are CRLB claims restricted to local model-based bounds rather than estimator performance claims?

## Fisher implementation

* Are finite differences formed between physically comparable states?
* Are state units and dimensionless scaling handled correctly?
* Is the Fisher matrix mathematically consistent with the implemented noise model?
* Are singular and poorly conditioned matrices handled honestly?
* Are eigenvalue, determinant, condition-number, and CRLB calculations numerically defensible?
* Is any pseudoinverse or regularization explicit rather than hidden?

## Nuisance treatment

* Is global gain represented consistently?
* Is the Schur complement implemented correctly?
* Does nuisance marginalization avoid artificially increasing available contact information?
* Is the result interpreted as robustness to gain uncertainty rather than proof of general nuisance robustness?

## Benchmark fairness

* Are nominal and candidate49 evaluated using identical settings?
* Are stochastic settings controlled or at least recorded?
* Are pre-existing expensive artifacts reused safely?
* Are invalidated artifacts clearly distinguished from valid ones?
* Were thresholds, scaling factors, state subsets, or noise assumptions chosen after seeing the results in a way that favors candidate49?

## Scientific conclusion

The reviewer must determine whether the evidence supports any of the following, without forcing one to be true:

1. candidate49 improves sensor-facing contact information;
2. candidate49 improves only intrinsic transport, not camera-space information;
3. candidate49's apparent improvement is primarily global photometric gain;
4. performance is mixed or strongly state-dependent;
5. nominal is equal or superior;
6. current evidence is insufficient.

A negative conclusion is fully acceptable.

## Scope review

Also identify:

* unintended scope expansion;
* unnecessary abstractions;
* duplicated metrics or observation implementations;
* repository dependency violations;
* changes to the 2D optimizer;
* hidden compatibility layers;
* unrelated cleanup;
* benchmark modifications whose purpose appears to be making the candidate pass.

Report all required checklist items as PASS, FAIL, or UNCLEAR with concrete evidence.

---

# Reviewer repair loop

Because `Fix reviewer findings: YES`, address reviewer FAIL findings only when they are:

* within the original scope;
* genuine implementation or benchmark correctness defects;
* resolvable without broadening the task.

Do not treat UNCLEAR findings as authorization for speculative new features or experiments.

If a reviewer finding invalidates an expensive result, rerun only the affected stage.

Reuse the same reviewer subagent for follow-up verification when possible.

Do not create a chain of new reviewer agents.

Stop when in-scope FAIL findings are resolved or a concrete blocker remains.

---

# Completion criteria

This iteration is complete when enough trustworthy evidence exists to answer the stated decision question for nominal versus candidate49 under the implemented first-pass sensor observation model.

Completion does not require candidate49 to win.

The final report must clearly separate:

1. implementation changes;
2. checks performed;
3. benchmark configuration;
4. measured results;
5. nominal versus candidate49 comparison;
6. gain-nuisance result;
7. implementation-agent checklist status;
8. independent reviewer findings;
9. reviewer findings that were fixed;
10. remaining FAIL or UNCLEAR items;
11. scientific limitations;
12. intentionally deferred follow-up work.

Do not continue into roughness studies, LED-profile studies, finite five-LED geometry validation, full 3D mechanics, prototype calibration, multi-morphology sweeps, or optimization changes unless the user explicitly starts a later iteration for those tasks.
