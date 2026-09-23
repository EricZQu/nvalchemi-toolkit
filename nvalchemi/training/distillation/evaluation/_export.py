# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Base record every evaluation measurement exports from and rebuilds into.

Every measurement exports with ``to_dict`` and rebuilds with ``from_dict``, so a
sweep can persist each student's results and aggregate them later. The two
live here once: a record is a frozen Pydantic model whose export is its field
dump and whose rebuild is validation, with a key the record does not declare
or a required one the export lacks refused where the export is read.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, ValidationError

__all__ = ["MeasurementRecord"]


class MeasurementRecord(BaseModel):
    """Frozen measurement that exports as a plain dictionary and rebuilds from one.

    Subclasses declare their fields as annotations; ``to_dict`` returns the
    field dump, tuples and ``None`` kept, and ``from_dict`` validates a mapping
    back into the record, so a list read out of JSON returns as the tuple its
    field declares and a non-finite float spelled as a string by a strict
    writer returns as the number.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    def to_dict(self) -> dict[str, Any]:
        """Return every field as a plain dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Rebuild the record from a :meth:`to_dict` export.

        Raises
        ------
        ValueError
            If *data* carries a key the record does not declare, omits one of
            its fields that has no default, or holds a value a field refuses.
        """
        try:
            return cls.model_validate(dict(data))
        except ValidationError as exc:
            raise ValueError(_rebuild_failure(cls, exc)) from exc


def _rebuild_failure(cls: type[MeasurementRecord], exc: ValidationError) -> str:
    """Return the rebuild error for *exc*, naming the keys or the fields at fault."""
    unknown = sorted(
        str(error["loc"][0])
        for error in exc.errors()
        if error["type"] == "extra_forbidden"
    )
    if unknown:
        return (
            f"{cls.__name__} cannot be rebuilt from a mapping carrying "
            f"{unknown!r}; expected keys from {sorted(cls.model_fields)!r}."
        )
    missing = sorted(
        str(error["loc"][0]) for error in exc.errors() if error["type"] == "missing"
    )
    if missing:
        return (
            f"{cls.__name__} cannot be rebuilt from a mapping missing the "
            f"required {missing!r}."
        )
    faults = "; ".join(
        f"{cls.__name__}.{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
        for error in exc.errors()
    )
    return f"{cls.__name__} cannot be rebuilt from a mapping: {faults}."
