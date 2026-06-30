"""Generic ordered registry mapping names to strategy objects.

New strategies (fusion methods, metrics, grouping) are added by subclassing
and calling ``register(name, instance)`` once at import — no ``if``-chains
to update across pipeline stages.

``Registry`` is an ordered ``Mapping`` (name → object), so stages iterate it
with ``.items()`` / ``.keys()`` / ``registry[name]``. Registration collisions
are always a loud error. ``require_registered`` / ``require_all_registered``
validate config values early, failing fast with the valid set listed.

Library code only — never imports ``config``.
"""

from collections.abc import Mapping
from typing import Iterator


class RegistryError(Exception):
    """Raised on a registration collision or a lookup of an unknown name."""


class Registry(Mapping):
    """Ordered ``name -> object`` store with collision-checked registration.

    Implements ``Mapping`` (iteration in registration order). Add entries with
    :meth:`register`; built-in registries are populated once at import.
    """

    def __init__(self, name: str):
        self._name = name
        self._items: dict[str, object] = {}

    # -- Mapping protocol --------------------------------------------------- #
    def __getitem__(self, key: str):
        try:
            return self._items[key]
        except KeyError:
            raise RegistryError(
                f"{self._name!r} has no entry {key!r}; registered: {self.names()}"
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key) -> bool:
        # Mapping default catches KeyError, but __getitem__ raises RegistryError,
        # so test the backing dict directly.
        return key in self._items

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Registry({self._name!r}, {self.names()})"

    # -- Registration ------------------------------------------------------- #
    def register(self, name: str, obj=None, *, override: bool = False):
        """Register ``obj`` under ``name``; works as a direct call or a decorator.

        Direct: ``registry.register("Average", AverageFusion())``
        Decorator: ``@registry.register("Average")`` — registers and returns the class.

        Raises :class:`RegistryError` on a name collision unless ``override=True``.
        """
        if obj is None:
            def decorator(decorated):
                self._insert(name, decorated, override=override)
                return decorated

            return decorator
        self._insert(name, obj, override=override)
        return obj

    def _insert(self, name: str, obj, *, override: bool) -> None:
        if not isinstance(name, str):
            raise RegistryError(f"{self._name!r} registry key must be a str, got {name!r}")
        if name in self._items and not override:
            raise RegistryError(
                f"{self._name!r} already has an entry {name!r}; "
                f"pass override=True to replace it"
            )
        self._items[name] = obj

    def names(self) -> list[str]:
        """Registered names, in registration order."""
        return list(self._items)


# --------------------------------------------------------------------------- #
# Config validation driven from Registry.names().
# --------------------------------------------------------------------------- #
def require_registered(value, registry: Registry, *, field: str) -> None:
    """Raise ``ValueError`` if ``value`` is not in ``registry``.

    Validates one config field at stage start; ``None`` is accepted (unset field).
    """
    if value is None:
        return
    if value not in registry:
        raise ValueError(
            f"{field}={value!r} is not a registered choice; "
            f"valid options: {registry.names()}"
        )


def require_all_registered(values, registry: Registry, *, field: str) -> None:
    """Raise ``ValueError`` if any element of ``values`` is not in ``registry``.

    For collection-valued config fields. ``None`` is accepted; empty passes.
    """
    if values is None:
        return
    unknown = [value for value in values if value not in registry]
    if unknown:
        raise ValueError(
            f"{field} contains unregistered {unknown}; "
            f"valid options: {registry.names()}"
        )
