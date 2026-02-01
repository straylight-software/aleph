# prelude/utils/set.bzl
#
# Set data structure for Starlark.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRELUDE ARCHAEOLOGY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Starlark's builtin `dedupe` compares by identity, not equality:
#
#     things = ["huh", "huh"]
#     len(dedupe(things)) == 2  # Still 2!
#
# A set compares by equality, so will properly deduplicate.
#
# Usage:
#     s = set()
#     s.add("foo")
#     s.add("bar")
#     s.add("foo")  # No-op, already present
#     s.list()  # ["foo", "bar"]
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set_record = record(
    _entries = field(dict),
    list = field(typing.Callable),
    add = field(typing.Callable),
    remove = field(typing.Callable),
    update = field(typing.Callable),
    contains = field(typing.Callable),
    size = field(typing.Callable),
)

set_type = set_record

def set(initial_entries: list = []) -> set_type:
    """Create a new set, optionally with initial entries."""
    
    def set_list():
        """Return set entries as a list."""
        return self._entries.keys()

    def set_add(v) -> bool:
        """Add value to set, return True if new."""
        if self.contains(v):
            return False
        self._entries[v] = None
        return True

    def set_contains(v) -> bool:
        """Check if value is in set."""
        return v in self._entries

    def set_remove(v) -> bool:
        """Remove value from set, return True if was present."""
        if self.contains(v):
            self._entries.pop(v)
            return True
        return False

    def set_update(values: list) -> list:
        """Add multiple values, return list of newly added."""
        return [v for v in values if self.add(v)]

    def set_size() -> int:
        """Return number of entries in set."""
        return len(self._entries)

    self = set_record(
        _entries = {},
        list = set_list,
        add = set_add,
        remove = set_remove,
        update = set_update,
        contains = set_contains,
        size = set_size,
    )

    self.update(initial_entries)
    return self
