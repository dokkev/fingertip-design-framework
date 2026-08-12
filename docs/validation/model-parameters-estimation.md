## Fixed Material and Optical Parameters

The fingertip geometry is optimized while the silicone material properties are
held fixed. The compliant pad is fabricated using Smooth-On Solaris, Shore A15.
Because a full material-identification experiment was outside the scope and
time constraints of this study, representative material parameters were chosen
from manufacturer data and simple physically motivated estimates.

These values should therefore be interpreted as **model parameters used for
design optimization**, rather than experimentally calibrated material
constants.

### Mechanical parameters

| Parameter | Value | Basis |
| --- | ---: | --- |
| Young's modulus, `E` | `0.55 MPa` | Representative estimate for a Shore A15 elastomer |
| Poisson ratio, `ν` | `0.49` | Nearly incompressible silicone assumption |

#### Young's modulus

The Solaris datasheet specifies a Shore A hardness of approximately 15 but does
not directly provide a small-strain Young's modulus.

A nominal value of

\[
E = 0.55\ \text{MPa}
\]

is used based on empirical hardness-to-modulus estimates for soft elastomers.

This value is not treated as an experimentally identified property of Solaris.
Instead, it provides a representative stiffness for geometry optimization.

Because silicone stiffness can vary with strain state, curing conditions, and
the constitutive model used in FEM, sensitivity to the assumed modulus should
be checked using a small range such as

\[
E \in \{0.30,\ 0.55,\ 0.80\}\ \text{MPa}.
\]

The purpose of this sensitivity study is not to estimate the true modulus of
Solaris, but to verify that the relative performance of candidate fingertip
geometries is not strongly dependent on the nominal stiffness assumption.

#### Poisson ratio

Silicone elastomers are approximately incompressible. The FEM therefore uses

\[
\nu = 0.49.
\]

This value avoids the exact incompressible limit while representing the
near-incompressible mechanical response of the silicone pad.

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

Mechanical and optical material properties are **not optimization variables**.

The optimization changes only the fingertip morphology while keeping the
nominal material parameters fixed:

```python
young_modulus_mpa = 0.55
poisson_ratio = 0.49

refractive_index_air = 1.00
refractive_index_silicone = 1.41
absorption_per_mm = 0.02