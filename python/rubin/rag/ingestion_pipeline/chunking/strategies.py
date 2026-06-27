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

"""Concrete chunking strategy implementations.

Each class subclasses BaseChunkingStrategy and is auto-registered in
BaseChunkingStrategy._registry under a key derived from its class name
(snake_case, ``_strategy`` suffix stripped). For example,
``RecursiveCharacterStrategy`` registers as ``recursive_character``.

To add a new strategy: define a subclass here. Registration is automatic.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base import BaseChunkingStrategy


class RecursiveCharacterStrategy(BaseChunkingStrategy):
    """Chunking strategy using LangChain's RecursiveCharacterTextSplitter.

    Splits text by trying a list of separators in order (paragraphs, lines,
    words) until chunks are small enough. Character count is used as the
    length measure.

    Registered as ``recursive_character``.

    Parameters
    ----------
    params : dict
        Accepted keys:

        ``chunk_size`` : int, default 1000
            Maximum number of characters per chunk.
        ``chunk_overlap`` : int, default 50
            Number of characters of overlap between consecutive chunks.
        ``separators`` : list[str], optional
            Ordered list of separators to try. Defaults to LangChain's
            built-in list (paragraph, line, word, character).
    """

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        kwargs: dict = {
            "chunk_size": params.get("chunk_size", 1000),
            "chunk_overlap": params.get("chunk_overlap", 50),
        }
        if "separators" in params:
            kwargs["separators"] = params["separators"]
        self._splitter = RecursiveCharacterTextSplitter(**kwargs)

    def split(self, text: str, metadata: dict | None = None) -> list[str]:
        return self._splitter.split_text(text)
