# Fixed Mechanics and Optical Parameters

The current default mechanics material is Smooth-On Dragon Skin 10 NV. Newton
receives `density_kg_m3`, `k_mu_pa`, `k_lambda_pa`, and `k_damp` directly from
`FingertipParameters.viscoelastic`; these values are numerical inputs, not a
complete material-identification result. `LumoSimulation` separately owns the
mesh and solver settings.

`FingertipParameters.optical` stores one `SiliconeOptics` value with only the
monochromatic properties used by current transport. The low/nominal/high
presets below support sensitivity analysis. They are not optimization variables
and are not calibrated measurements of the actual LUMO casting.

## Green Sequin LED

The current source is the [Adafruit Green LED Sequin, Product
1756](https://www.adafruit.com/product/1756). Adafruit specifies a `4 x 9 mm`
board that is `2 mm` thick and identifies its LED as a 1206 green device. The
2-D fingertip geometry therefore uses the board's `4 x 2 mm` cross-section.

The linked [LuckyLight S150PGC-G5-1B technical
datasheet](https://cdn-shop.adafruit.com/datasheets/S150PGC-G5-1B.pdf) gives:

| `LEDParameters` field | Default | Source status |
| --- | ---: | --- |
| `dominant_wavelength_nm` | `525` | LuckyLight manufacturer typical value |
| `peak_wavelength_nm` | `520` | LuckyLight manufacturer typical value |
| `spectral_half_width_nm` | `35` | LuckyLight manufacturer typical value |
| `viewing_half_angle_deg` | `60` | Half of the manufacturer `120 deg` full half-intensity angle |
| `normalized_power` | `1.0` | Modeling normalization; not optical watts |

Transport remains monochromatic at `525 nm` and uses an ideal Lambertian
emitter. The datasheet's half-intensity angle is consistent with this
first-order model because `cos(60 deg) = 0.5`; the implementation does not claim
to reproduce the complete measured radiation diagram.

## Silicone refractive index

The [Smooth-On Solaris technical
bulletin](https://www.smooth-on.com/tb/files/Solaris_TB.pdf) describes Solaris
as clear, ultra-transparent, and intended for maximum light transmission. It
reports a refractive index of `1.41` at `20 deg C` using ASTM D-1218. That
manufacturer value is used by all Solaris presets.

The [Smooth-On Dragon Skin NV technical
bulletin](https://www.smooth-on.com/tb/files/DRAGON_SKIN_NV_SERIES_TB.pdf)
describes Dragon Skin 10 NV as translucent but reports neither refractive index
nor optical attenuation. The Dragon Skin presets therefore use `1.4348`, a
near-green literature prior measured for generic Sylgard 184 PDMS at `532 nm`,
as reported in [Polydimethylsiloxane as a more biocompatible alternative to
glass in optogenetics](https://pmc.ncbi.nlm.nih.gov/articles/PMC10522705/).
This is not a Dragon Skin 10 NV measurement.

The surrounding air remains the explicit modeling assumption `n_air = 1.0`.

## Effective bulk extinction

Neither Smooth-On bulletin provides a wavelength-dependent extinction
coefficient. Current transport therefore uses literature propagation losses as
sensitivity priors. Power propagation loss in `dB/cm` is converted to the
Beer-Lambert coefficient in `1/m` by

\[
\mu_\mathrm{ext}
= \alpha_\mathrm{dB/cm}\,100\,\frac{\ln(10)}{10}.
\]

The source values are:

| Visible PDMS literature value | Converted `mu_ext` | Source |
| ---: | ---: | --- |
| `0.36 dB/cm` at `635 nm` | `8.289306 1/m` | Clear PDMS waveguide measurement in [Ersen and Sahin, 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5997005/) |
| `0.63 dB/cm` at `441.6 nm` | `14.506286 1/m` | PDMS-fiber length measurement in [Ding et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6780825/) |
| `1.8 dB/cm` at `532 nm` | `41.446532 1/m` | Liquid-PDMS-core/PDMS waveguide summarized in [Microfabrication and Applications of Opto-Microfluidic Sensors](https://www.mdpi.com/1424-8220/11/5/5360) |
| `3.1 dB/cm` at `532 nm` | `71.380138 1/m` | Cast PDMS/air waveguide summarized in the same review |
| `4.8 dB/cm` at `473 nm` | `110.524084 1/m` | Flexible PDMS waveguide measurement in [Fabrication and Characterization of PDMS Waveguides for Flexible Optrodes](https://pmc.ncbi.nlm.nih.gov/articles/PMC11469164/) |

These studies use different PDMS formulations, wavelengths, geometries, and
fabrication processes. Their propagation losses can include absorption,
scattering, surface roughness, and other unresolved effects. LUMO consequently
names the parameter `extinction_coefficient_m_inv`, not an intrinsic absorption
coefficient.

The concrete sensitivity presets are:

| Material | Assumption | `n` | Literature prior | `mu_ext [1/m]` |
| --- | --- | ---: | ---: | ---: |
| Solaris | low | `1.41` | `0.36 dB/cm` | `8.289306` |
| Solaris | nominal | `1.41` | `0.63 dB/cm` | `14.506286` |
| Solaris | high | `1.41` | `1.8 dB/cm` | `41.446532` |
| Dragon Skin 10 NV | low | `1.4348` | `1.8 dB/cm` | `41.446532` |
| Dragon Skin 10 NV | nominal | `1.4348` | `3.1 dB/cm` | `71.380138` |
| Dragon Skin 10 NV | high | `1.4348` | `4.8 dB/cm` | `110.524084` |

Assigning the lower literature band to manufacturer-described clear Solaris
and the upper band to manufacturer-described translucent Dragon Skin is a
modeling choice for sensitivity analysis. It is not evidence that either
product has those exact coefficients. In particular, Dragon Skin scattering is
represented only as effective loss from the tracked ballistic path; the current
model does not redirect scattered power or implement volumetric transport.

## Current use

`FingertipParameters.optical` defaults to the nominal Dragon Skin 10 NV preset,
matching the default mechanics material. The LED sensor-response validation
uses the same mechanics deformation and common optical samples for all six
presets, so the reported differences isolate only the stated optical
assumptions.
