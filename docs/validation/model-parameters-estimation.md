## Fixed Mechanics and Optical Parameters

The fingertip geometry is optimized while the Newton numerical mechanics
contract and optical inputs are held fixed. The compliant pad is fabricated
using Smooth-On Solaris, Shore A15, but a constitutive material-identification
experiment is not represented by the current production code.

### Fingertip material parameters

Production evaluation uses the values serialized by
`FingertipParameters.viscoelastic`. In particular, the current Newton path
receives `density_kg_m3`, `k_mu_pa`, `k_lambda_pa`, and `k_damp` directly from
that fingertip-owned material group. These values are frozen numerical inputs
for reproducible search; this document does not reinterpret them as an
experimentally calibrated Young's modulus or Poisson ratio.

`lumo.mechanics_contract.DEFAULT_MECHANICS_CONTRACT` remains responsible for
solver iteration, contact penalty, timestep, and checkpoint-acceptance
settings. It does not define the fingertip material.

The bulk optical values are likewise stored in `FingertipParameters.optical`.
The LED source/package remains a separate fixed `LED` input because its package
fit and emission model are distinct from the silicone bulk material.

Any future `E, nu` inputs must first define a reviewed constitutive mapping to
the Newton backend and validation evidence. Until then, they are deliberately
absent as standalone fields from `FingertipParameters` and from the
optimization design space.

---

### Optical parameters

| Parameter | Value | Basis |
| --- | ---: | --- |
| Air refractive index | `1.00` | Standard assumption |
| Solaris refractive index | `1.41` | Manufacturer data |
| Bulk absorption coefficient | `0.02 mm^-1` | Nominal estimate consistent with reported Solaris transmission |

#### Refractive index

The Solaris manufacturer data reports a refractive index of approximately

\[
n_\mathrm{silicone} = 1.41.
\]

The surrounding air is modeled as

\[
n_\mathrm{air} = 1.00.
\]

These values are used directly for Snell refraction, Fresnel reflection, and
total internal reflection in the optical transport model.

#### Optical attenuation

The manufacturer does not provide a wavelength-dependent bulk absorption
coefficient for Solaris.

Available transmission measurements for approximately 2 mm thick Solaris
samples indicate visible-light transmission on the order of 90--95%. Part of
this loss is caused by Fresnel reflection at the two air--silicone interfaces.

For a refractive index of \(n=1.41\), the normal-incidence reflection
coefficient of a single air--silicone interface is approximately

\[
R =
\left(
\frac{n_\mathrm{silicone}-n_\mathrm{air}}
     {n_\mathrm{silicone}+n_\mathrm{air}}
\right)^2
\approx 0.029.
\]

Therefore, even an ideally non-absorbing silicone sample would transmit only
approximately

\[
(1-R)^2 \approx 0.943
\]

through two air--silicone interfaces.

Assuming a total transmission near the lower end of the reported range
(\(\sim 90\%\)) for a 2 mm sample gives an approximate bulk transmission of

\[
T_\mathrm{bulk}
\approx
\frac{0.90}{0.943}
\approx 0.954.
\]

Using a Beer--Lambert attenuation model,

\[
T_\mathrm{bulk}(L) = \exp(-\mu_a L),
\]

gives

\[
\mu_a
=
-\frac{\ln(T_\mathrm{bulk})}{L}
\approx
0.02\ \text{mm}^{-1}.
\]

The optical model therefore uses

\[
\boxed{\mu_a = 0.02\ \text{mm}^{-1}}
\]

as a **nominal bulk attenuation parameter**.

This value should not be interpreted as a directly measured absorption
coefficient of Solaris. It is a physically plausible modeling value chosen to
be consistent with reported optical transmission.

A small sensitivity range can be evaluated using

\[
\mu_a \in
\{0,\ 0.02,\ 0.05\}\ \text{mm}^{-1}.
\]

Again, the purpose is to determine whether the ranking of optimized fingertip
geometries is robust to uncertainty in optical attenuation.

---

### Optimization treatment

Mechanics and optical inputs are **not optimization variables**.

The optimization changes only the fingertip morphology while keeping the
nominal material parameters fixed:

```python
refractive_index_air = 1.00
refractive_index_silicone = 1.41
absorption_per_mm = 0.02
```

The complete mechanics values are taken from `DEFAULT_MECHANICS_CONTRACT` and
are included in the evaluation contract fingerprint.
