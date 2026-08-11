from __future__ import annotations

from types import SimpleNamespace

from fem.kratos_adapter import _configure_properties
from model import FingertipParameters


class _FakeProperties(dict):
    pass


class _FakeModelPart:
    def __init__(self) -> None:
        self.Properties: dict[int, _FakeProperties] = {}

    def HasProperties(self, identifier: int) -> bool:
        return identifier in self.Properties

    def CreateNewProperties(self, identifier: int) -> _FakeProperties:
        properties = _FakeProperties()
        self.Properties[identifier] = properties
        return properties


class _FakeKratos:
    YOUNG_MODULUS = "YOUNG_MODULUS"
    POISSON_RATIO = "POISSON_RATIO"
    THICKNESS = "THICKNESS"
    DENSITY = "DENSITY"
    VOLUME_ACCELERATION = "VOLUME_ACCELERATION"
    CONSTITUTIVE_LAW = "CONSTITUTIVE_LAW"


class _FakeConstitutiveLaws:
    @staticmethod
    def HyperElasticPlaneStrain2DLaw() -> str:
        return "HyperElasticPlaneStrain2DLaw"


def test_pad_properties_use_fingertip_parameters() -> None:
    parameters = FingertipParameters(
        young_modulus_mpa=2.75,
        poisson_ratio=0.32,
    )
    mesh = SimpleNamespace(parameters=parameters)

    pad_properties, carrier_properties = _configure_properties(
        _FakeModelPart(),
        _FakeKratos,
        _FakeConstitutiveLaws,
        mesh,
    )

    assert pad_properties[_FakeKratos.YOUNG_MODULUS] == 2.75
    assert pad_properties[_FakeKratos.POISSON_RATIO] == 0.32
    assert carrier_properties[_FakeKratos.YOUNG_MODULUS] == 1.0
    assert carrier_properties[_FakeKratos.POISSON_RATIO] == 0.49
