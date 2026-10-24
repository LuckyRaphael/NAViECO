"""Advanced parameter configuration for NAViECO Layer0."""

from numbers import Integral

_DEFAULTS = {
    "latent_dim_coarse": 100,
    "n_epochs_coarse": 15,
    "latent_dim_fine": 300,
    "n_epochs_fine": 15,
    "seed": 42,
}


def resolve_advanced(level, advanced=None):
    """Validate and return only parameters active for the selected level."""

    if level not in ("coarse", "fine"):
        raise ValueError("level must be 'coarse' or 'fine'.")

    if advanced is None:
        advanced = {}
    if not isinstance(advanced, dict):
        raise TypeError("advanced must be a dictionary or None.")

    unknown = set(advanced) - set(_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown advanced settings: {sorted(unknown)}")

    # Fine parameters are completely inactive when level='coarse'.
    active = ("latent_dim_coarse", "n_epochs_coarse", "seed")

    if level == "fine":
        active += ("latent_dim_fine", "n_epochs_fine")

    config = {key: advanced.get(key, _DEFAULTS[key]) for key in active}

    for key, value in config.items():
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{key} must be an integer.")

        if key.startswith("latent_dim") and not (8 <= value <= 512):
            raise ValueError(f"{key} must be between 8 and 512.")

        if key.startswith("n_epochs") and value < 1:
            raise ValueError(f"{key} must be >= 1.")

        if key == "seed" and value < 0:
            raise ValueError("seed must be >= 0.")

    return {key: int(value) for key, value in config.items()}
