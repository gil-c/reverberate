"""Moving scenes as recipes: the format's reader, its rules, and a generator.

``docs/formats/scene-recipe.md`` is the contract and ADR 0016 the decision.
This package writes and validates recipes; the application draws them and the
trace reads them, both through the functions below.

- :mod:`.recipe`: the typed recipe, strict parsing, the canonical form and
  ``recipe_sha256``.
- :mod:`.kinematics`: where a source and the listener are at time ``t``. The
  single definition; nothing else may compute a position from a recipe.
- :mod:`.validate`: the numbered rules.
- :mod:`.layout`: a storey's stations and rails.
- :mod:`.generate`: a recipe from ranges and a seed, deterministic.
- :mod:`.describe`: a recipe in words.
- :mod:`.clips`: the library of dry clips the generator names: its manifest,
  and the files built from the bucket, digests checked
  (``docs/formats/clip-library.md``).

The package is pure: no network, and no file is touched outside
``load_recipe``, ``save_recipe``, ``load_clip_library`` and the HSSD loaders of
:mod:`.layout`. :mod:`.clips` is the exception, and only when it is handed a
store: ``python -m reverberate.scenes clips fetch`` reads the bucket and
writes the clips.
"""

from reverberate.scenes.describe import describe, low_band_positions
from reverberate.scenes.generate import (
    GENERATOR_NAME,
    GENERATOR_VERSION,
    ClipEntry,
    ClipLibrary,
    GenerationError,
    Parameters,
    generate,
    load_clip_library,
    placeholder_assets,
    placeholder_clips,
)
from reverberate.scenes.kinematics import (
    AUDIBLE_TAIL_S,
    YAW_STEP_S,
    ListenerState,
    LowBandPositions,
    SourceState,
    audible_steps,
    listener_state,
    low_band_source_positions,
    rail_arc_lengths,
    rail_length,
    rail_samples,
    sample_times,
    seat_rail_heights,
    source_state,
)
from reverberate.scenes.layout import (
    Fixture,
    Floor,
    Layout,
    LayoutSettings,
    Room,
    SeatObject,
    build_layout,
    load_hssd_floor,
    load_hssd_layout,
)
from reverberate.scenes.recipe import (
    Recipe,
    RecipeError,
    canonical_bytes,
    load_recipe,
    parse_recipe,
    quantise,
    recipe_sha256,
    save_recipe,
)
from reverberate.scenes.social import (
    SOCIAL_GENERATOR_VERSION,
    SocialParameters,
    generate_social,
    placeholder_social_clips,
)
from reverberate.scenes.validate import Violation, check, validate, validate_text

__all__ = [
    "AUDIBLE_TAIL_S",
    "GENERATOR_NAME",
    "GENERATOR_VERSION",
    "SOCIAL_GENERATOR_VERSION",
    "YAW_STEP_S",
    "ClipEntry",
    "ClipLibrary",
    "Fixture",
    "Floor",
    "GenerationError",
    "Layout",
    "LayoutSettings",
    "ListenerState",
    "Parameters",
    "Recipe",
    "RecipeError",
    "Room",
    "SeatObject",
    "LowBandPositions",
    "SocialParameters",
    "SourceState",
    "Violation",
    "audible_steps",
    "build_layout",
    "canonical_bytes",
    "check",
    "describe",
    "generate",
    "generate_social",
    "listener_state",
    "load_clip_library",
    "load_hssd_floor",
    "load_hssd_layout",
    "load_recipe",
    "low_band_positions",
    "low_band_source_positions",
    "parse_recipe",
    "placeholder_assets",
    "placeholder_clips",
    "placeholder_social_clips",
    "quantise",
    "rail_arc_lengths",
    "rail_length",
    "rail_samples",
    "recipe_sha256",
    "sample_times",
    "save_recipe",
    "seat_rail_heights",
    "source_state",
    "validate",
    "validate_text",
]
