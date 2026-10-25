"""Layer 0 CONCORD settings, including backwards-compatible aliases."""

from numbers import Integral

_DEFAULTS = {
    "latent_dim_level1": 100, "n_epochs_level1": 15,
    "latent_dim_level2": 100, "n_epochs_level2": 15,
    "latent_dim_level3": 300, "n_epochs_level3": 15,
    "seed": 42,
}
_LEGACY = {
    "latent_dim_coarse": ("latent_dim_level1", "latent_dim_level2"),
    "n_epochs_coarse": ("n_epochs_level1", "n_epochs_level2"),
    "latent_dim_fine": ("latent_dim_level3",),
    "n_epochs_fine": ("n_epochs_level3",),
}


def resolve_advanced(level, advanced=None):
    """Resolve per-level parameters. Old coarse/fine names remain accepted."""
    if level not in {"coarse", "fine"}:
        raise ValueError("level must be 'coarse' or 'fine'.")
    if advanced is None:
        advanced = {}
    if not isinstance(advanced, dict):
        raise TypeError("advanced must be a dictionary or None.")
    unknown = set(advanced) - set(_DEFAULTS) - set(_LEGACY)
    if unknown:
        raise ValueError(f"Unknown advanced settings: {sorted(unknown)}")

    active = ["latent_dim_level1", "n_epochs_level1",
              "latent_dim_level2", "n_epochs_level2", "seed"]
    if level == "fine":
        active += ["latent_dim_level3", "n_epochs_level3"]
    effective = {name: _DEFAULTS[name] for name in active}
    for old, targets in _LEGACY.items():
        if old in advanced:
            for target in targets:
                if target in effective:
                    if target in advanced and advanced[target] != advanced[old]:
                        raise ValueError(f"Conflicting settings: {old} and {target}.")
                    effective[target] = advanced[old]
    for key in active:
        if key in advanced:
            effective[key] = advanced[key]
    for name, value in effective.items():
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer.")
        if name.startswith("latent_dim") and not 8 <= value <= 512:
            raise ValueError(f"{name} must be between 8 and 512.")
        if name.startswith("n_epochs") and value < 1:
            raise ValueError(f"{name} must be >= 1.")
        if name == "seed" and value < 0:
            raise ValueError("seed must be >= 0.")
    return {name: int(value) for name, value in effective.items()}
