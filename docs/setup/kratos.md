# Kratos environment

Kratos is intentionally not a Python project dependency. Solver-backed
validation uses the externally managed interpreter:

```text
/home/dk/miniconda3/envs/lit/bin/python
Kratos package/application: 10.3.x
Kratos kernel: 10.3.0
```

The repository imports StructuralMechanicsApplication,
ConstitutiveLawsApplication, and ContactStructuralMechanicsApplication only
inside the FEM backend or solver-backed validation.
