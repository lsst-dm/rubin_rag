#
# This file is part of rubin_rag.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Base class and registry for chunking strategies."""

import re
from abc import ABC, abstractmethod
from typing import ClassVar


class BaseChunkingStrategy(ABC):
    """Abstract base class for chunking strategies.

    Subclasses implement ``split`` to define how a document's text is
    divided into chunks. All source-specific concerns (record format,
    metadata) are handled by the caller; the strategy only deals with
    plain text.

    Subclasses are auto-registered in ``registry`` on definition. The
    registry key is derived from the class name: converted to snake_case
    with the ``_strategy`` suffix removed. For example,
    ``RecursiveCharacterStrategy`` registers as ``recursive_character``.

    Parameters
    ----------
    params : dict
        Strategy-specific parameters (e.g. chunk size, overlap). Each
        subclass is responsible for validating its own params.
    """

    registry: ClassVar[dict[str, type["BaseChunkingStrategy"]]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        key = re.sub(r"(?<!^)(?=[A-Z])", "_", cls.__name__).lower()
        key = key.removesuffix("_strategy")
        BaseChunkingStrategy.registry[key] = cls

    def __init__(self, params: dict) -> None:
        self._params = params

    @abstractmethod
    def split(self, text: str, metadata: dict | None = None) -> list[str]:
        """Split a document's text into chunks.

        Parameters
        ----------
        text : str
            Full text of a single document.
        metadata : dict | None
            Optional document metadata. Most strategies ignore this;
            it is provided for strategies that need document context
            to make splitting decisions (e.g. choosing separators
            based on document type).

        Returns
        -------
        list[str]
            Ordered list of text chunks.
        """
