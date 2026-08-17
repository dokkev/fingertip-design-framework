# Optical Evaluation

## 1. Evaluation Goal

The optimization objective is to make the optical response to contact as
distinct as possible from the no-contact state.

The design problem is therefore formulated as a state-separation problem:

\[
\text{NO CONTACT}
\quad\longleftrightarrow\quad
\text{CONTACT}.
\]

The morphology is not optimized to maximize brightness alone, nor to maximize
relative percentage change.

Instead, it is optimized to maximize the absolute optical separation between
the two physical states at the lateral optical output.


## 2. Evaluation States

For each morphology, evaluate two states under the same illumination
configuration.

### No-contact state

\[
\mathbf{s}_0
\]

is the lateral optical signal before contact.

### Contact state

\[
\mathbf{s}_c
\]

is the lateral optical signal after contact, including:

- mechanically deformed fingertip geometry,
- the mechanically identified contact patch,
- the corresponding silicone-object optical interface.

The contact state therefore differs from the no-contact state through both
mechanical deformation and contact-induced optical boundary changes.


## 3. Primary Optimization Objective

Let

\[
\phi_L(s), \qquad \phi_R(s)
\]

denote the outgoing optical weight density along the left and right lateral
boundaries.

The primary morphology objective is

\[
\boxed{
J_{\mathrm{contact}}
=
\frac{
\|\phi_L^{c}-\phi_L^{0}\|_1
+
\|\phi_R^{c}-\phi_R^{0}\|_1
}{
W_{\mathrm{launch}}
}
}
\]

where \(W_{\mathrm{launch}}\) is the total launched optical weight.

Equivalently,

\[
J_{\mathrm{contact}}
=
\frac{
\|\mathbf{s}_c-\mathbf{s}_0\|_1
}{
W_{\mathrm{launch}}
}.
\]

The optimizer maximizes \(J_{\mathrm{contact}}\).

For a minimization-based optimizer,

\[
C = -J_{\mathrm{contact}}.
\]


## 4. Why Absolute Change Is Used

The objective measures absolute state separation rather than relative
percentage change.

For example,

\[
1 \rightarrow 2
\]

has a relative change of \(100\%\), but an absolute change of only \(1\).

Meanwhile,

\[
100 \rightarrow 120
\]

has a relative change of only \(20\%\), but an absolute change of \(20\).

For contact discrimination, the second response represents a larger optical
separation between the two states.

Therefore, the production objective does not use

\[
\frac{S_c-S_0}{S_0}.
\]

Instead, all state differences are normalized only by the common launched
optical weight:

\[
\frac{S_c-S_0}{W_{\mathrm{launch}}}.
\]

This avoids artificially favoring morphologies with very small baseline
signals.


## 5. No Weighted Multi-Term Objective

The production objective intentionally avoids a weighted combination such as

\[
J =
\alpha J_{\mathrm{contact}}
+
\beta J_{\mathrm{brightness}}
+
\gamma J_{\mathrm{redistribution}}.
\]

No empirical weighting coefficients are introduced.

The optimization objective contains one quantity:

\[
\boxed{
\text{contact/no-contact lateral state separation}
}
\]

Other optical quantities are retained as diagnostics rather than mixed into
the cost function.


## 6. Lateral Throughput

Absolute lateral throughput is retained separately:

\[
S_{\mathrm{side}}^{0}
=
\frac{E_{\mathrm{side}}^{0}}
{W_{\mathrm{launch}}},
\]

\[
S_{\mathrm{side}}^{c}
=
\frac{E_{\mathrm{side}}^{c}}
{W_{\mathrm{launch}}}.
\]

These values answer a different question:

> How much optical signal is available at the lateral boundary?

They are reported for every morphology but are not currently included as
weighted objective terms.

If a future physical sensor establishes a minimum usable signal level, that
requirement should be introduced as a physically motivated constraint rather
than as a tunable objective weight.


## 7. Source-Centered Quadrant Descriptor

Internal transport redistribution is summarized using four regions defined
relative to the light-source center.

Let

\[
\xi=x-x_{\mathrm{LED}},
\qquad
\eta=y-y_{\mathrm{LED}}.
\]

Define:

\[
Q_{UL}: \xi<0,\eta>0,
\]

\[
Q_{UR}: \xi>0,\eta>0,
\]

\[
Q_{LL}: \xi<0,\eta<0,
\]

\[
Q_{LR}: \xi>0,\eta<0.
\]

For each region,

\[
E_i
=
\int_{Q_i}P(x,y)\,dx\,dy.
\]

The contact-induced change is

\[
\Delta E_i
=
\frac{
E_i^c-E_i^0
}{
W_{\mathrm{launch}}
}.
\]

The resulting vector

\[
\boxed{
\Delta\mathbf{E}_Q
=
[
\Delta E_{UL},
\Delta E_{UR},
\Delta E_{LL},
\Delta E_{LR}
]
}
\]

is retained as an interpretable contact-transport signature.


## 8. Role of the Quadrant Descriptor

The quadrant descriptor is not the production optimization objective.

Its purpose is to explain how contact redistributes optical transport.

Thus:

\[
\Delta\mathbf{E}_Q
\]

answers

> Where did the optical transport change?

while

\[
J_{\mathrm{contact}}
\]

answers

> How strongly can the lateral optical output distinguish contact from
> no contact?

This separation prevents an internal transport descriptor from being mistaken
for the actual design objective.


## 9. Left and Right Signals

Left and right lateral signals are retained separately.

This prevents cancellation. For example,

\[
\Delta S_L=+20,
\qquad
\Delta S_R=-20
\]

represents a strong optical change even though

\[
\Delta S_L+\Delta S_R=0.
\]

The objective therefore measures the magnitude of the change before combining
the two lateral boundaries.


## 10. Historical Metrics

Earlier transport metrics such as \(J_2\) may be retained for historical
reproduction and comparison.

They are not required as production optimization objectives once the design
goal is explicitly defined as contact/no-contact lateral state separation.

Likewise, reduced internal-field descriptors should not be interpreted as
sensor observability metrics.


## 11. Evaluation Hierarchy

The evaluation hierarchy is

\[
\boxed{
\text{raw transport}
\rightarrow
\text{quadrant contact signature}
\rightarrow
\text{lateral contact/no-contact separation}
\rightarrow
\text{optimization objective}
}
\]

Specifically:

- raw transport: full mechanistic information,
- \(\Delta\mathbf{E}_Q\): interpretable redistribution descriptor,
- lateral profiles and throughput: output-level response,
- \(J_{\mathrm{contact}}\): scalar production optimization objective.


## 12. Future Sensor-Aware Evaluation

The current objective is deliberately independent of a camera or detector noise
model.

Once a physical sensing configuration and noise statistics are available,
contact/no-contact separation can be evaluated in the measurement domain using
a noise-aware distance such as

\[
(\mathbf{s}_c-\mathbf{s}_0)^T
\Sigma^{-1}
(\mathbf{s}_c-\mathbf{s}_0).
\]

Such a metric belongs to sensor-facing observability evaluation and is not part
of the current morphology optimization objective.