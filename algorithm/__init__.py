"""Learning-free online optical contact localization for LUMO."""

from .canonical import (
    CanonicalFingerConfig,
    CanonicalFingerMap,
    build_canonical_map,
    landmark_longitudinal_coordinates,
    transform_canonical_map,
    warp_to_canonical,
)
from .contact_localization import (
    ContactLocalizationConfig,
    ContactLocalizationResult,
    OnlineContactLocalizer,
    UnloadedOpticalReference,
    build_unloaded_reference,
    canonical_position_to_mm,
    causal_median_response,
    compute_longitudinal_response,
    localize_response_profile,
)
from .fingertip_segmentation import FingertipRegion, segment_fingertip
from .led_localization import (
    LED_POSITIONS_MM,
    LedGeometry,
    detect_leds,
    estimate_led_similarity_transform,
    reanchor_leds,
    track_leds,
)

__all__ = [
    "LED_POSITIONS_MM",
    "CanonicalFingerConfig",
    "CanonicalFingerMap",
    "ContactLocalizationConfig",
    "ContactLocalizationResult",
    "FingertipRegion",
    "LedGeometry",
    "OnlineContactLocalizer",
    "UnloadedOpticalReference",
    "build_canonical_map",
    "build_unloaded_reference",
    "canonical_position_to_mm",
    "causal_median_response",
    "compute_longitudinal_response",
    "detect_leds",
    "estimate_led_similarity_transform",
    "landmark_longitudinal_coordinates",
    "localize_response_profile",
    "reanchor_leds",
    "segment_fingertip",
    "track_leds",
    "transform_canonical_map",
    "warp_to_canonical",
]
