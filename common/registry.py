"""Generic, in-tree registry with additive out-of-tree extension (§3.7).

The pipeline used to read its fusion methods / metrics / grouping names from
literal lists and ``if``-chains. This module replaces that with a single
:class:`Registry` type so a new strategy is *registered*, not forked in:

* in-tree strategies are registered in a fixed, pinned order (the order the
  golden ``registry`` snapshot guards);
* a third-party package contributes a plugin by exposing a Python *entry point*
  in one of the ``moe.fusion`` / ``moe.grouping`` / ``moe.metric`` groups and
  calling :meth:`Registry.load_entry_points` -- no fork of ``common`` required;
* registration collisions are an explicit error (``override=False`` by default),
  never a silent shadow.

A :class:`Registry` is an ordered :class:`~collections.abc.Mapping`
(name -> object), so it is a drop-in for the plain ``name -> instance`` dicts the
stages already iterate (``.items()`` / ``.keys()`` / ``registry[name]`` /
``name in registry``). Iteration order is registration order.

Config validation (§1.6) is driven from :meth:`Registry.names` via
:func:`require_registered` / :func:`require_all_registered`, so a bad config
choice fails fast at the top of a stage with the valid set spelled out, rather
than deep inside with a ``KeyError`` or a silently skipped branch.

Library code only -- this module never imports an experiment ``config``.
"""

from collections.abc import Mapping
from typing import Callable, Iterator, Optional


class RegistryError(Exception):
    """Raised on a registration collision or a lookup of an unknown name."""


class Registry(Mapping):
    """Ordered ``name -> object`` registry with collision-checked registration.

    Implements the :class:`~collections.abc.Mapping` protocol (iteration in
    registration order), so it can stand in for the plain dicts the pipeline
    stages consume. New entries are added with :meth:`register`; the canonical
    in-tree registries are populated once at import in a pinned order.
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
        # Override the Mapping mixin: its default probes ``__getitem__`` and only
        # swallows ``KeyError``, but our ``__getitem__`` raises ``RegistryError``
        # on a miss -- so ``name in registry`` must test the backing dict directly
        # (used by the config-validation helpers).
        return key in self._items

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Registry({self._name!r}, {self.names()})"

    # -- Registration ------------------------------------------------------- #
    def register(self, name: str, obj=None, *, override: bool = False):
        """Register ``obj`` under ``name``; usable as a function or a decorator.

        Function form::

            registry.register("Average", AverageFusion())

        Decorator form (``obj`` omitted; the decorated object is registered and
        returned unchanged)::

            @registry.register("Average")
            class AverageFusion(...):
                ...

        A name already present raises :class:`RegistryError` unless
        ``override=True`` -- collisions are loud, never a silent shadow.
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

    # -- Out-of-tree discovery (entry points) ------------------------------- #
    def load_entry_points(self, group: str) -> list[str]:
        """Discover and register plugins advertised under entry-point ``group``.

        A third-party package registers a plugin by declaring an entry point in
        ``group`` (e.g. ``moe.fusion``); calling this pulls each one in by the
        fact of its installation, appended *after* the pinned in-tree entries so
        the in-tree order/prefix stays deterministic. Collisions with existing
        names raise (``override=False``). Returns the names that were loaded.

        Kept explicit (not auto-run at import) on purpose: the canonical
        registries are pinned in-tree, and the golden ``registry`` snapshot is an
        exact match, so auto-loading an installed plugin would (correctly) be a
        loud snapshot drift. Integrators opt in by calling this.
        """
        loaded: list[str] = []
        for entry in _iter_entry_points(group):
            self.register(entry.name, entry.load())
            loaded.append(entry.name)
        return loaded


def _iter_entry_points(group: str):
    """Yield entry points in ``group`` across importlib.metadata API versions."""
    from importlib import metadata

    entry_points = metadata.entry_points()
    # Python >=3.10 selectable API; fall back to the legacy dict-grouped API.
    select = getattr(entry_points, "select", None)
    if select is not None:
        return list(select(group=group))
    return list(entry_points.get(group, []))  # pragma: no cover - legacy API


# --------------------------------------------------------------------------- #
# Config validation driven from Registry.names() (§1.6).
# --------------------------------------------------------------------------- #
def require_registered(value, registry: Registry, *, field: str) -> None:
    """Raise a clear error if ``value`` is not a registered name.

    Validates one config choice (e.g. ``PAIRING_METHOD``) against the registry
    at the top of a stage, so a typo fails fast with the valid set listed.
    ``None`` is accepted (an unset/optional field).
    """
    if value is None:
        return
    if value not in registry:
        raise ValueError(
            f"{field}={value!r} is not a registered choice; "
            f"valid options: {registry.names()}"
        )


def require_all_registered(values, registry: Registry, *, field: str) -> None:
    """Raise if any element of ``values`` is not registered.

    For collection-valued config fields (``FILTER_FUSIONS`` / ``FILTER_METRICS``).
    ``None`` is accepted (an unset filter); an empty collection passes trivially.
    """
    if values is None:
        return
    unknown = [value for value in values if value not in registry]
    if unknown:
        raise ValueError(
            f"{field} contains unregistered {unknown}; "
            f"valid options: {registry.names()}"
        )
