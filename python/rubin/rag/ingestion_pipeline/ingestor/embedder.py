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

"""Embedding protocol, provider implementations, and per-model limits.

The ingestor works against ``EmbeddingFn`` — an object that embeds a batch
of texts (``__call__``) and exposes the provider/model's batching limits
(``limits``).  Concrete implementations hide all provider-specific details
(auth, extra parameters, retry logic).  ``EmbeddingFn`` is a
``typing.Protocol`` satisfied structurally, so a class needs no base class.

Batching limits are **facts about each provider/model's API**, not user
preferences, so they live in the ``_LIMITS`` table here (keyed by
``(provider, model)``) rather than in user config.  See
``memo/investigate_weaviate_20260701.md`` (2026-08-10) for the decision.

This module does **not** count tokens: exact per-provider tokenizers add a
first-use network dependency (HF Hub for Cohere/Voyage), which a batching
guard does not justify. The ingestor instead sizes batches with a single
model-agnostic ``tiktoken`` (cl100k_base) estimate for all providers.

Supported providers (selected via ``config["embedding"]["provider"]``):

- ``"openai"`` — reads ``OPENAI_API_KEY`` from the environment.
- ``"cohere"`` — reads ``COHERE_API_KEY`` from the environment.
- ``"voyageai"`` — reads ``VOYAGE_API_KEY`` from the environment.

To add a provider: implement the protocol (``__call__`` + ``limits``), add a
``_LIMITS`` row per model, and add a branch to ``build_embedder``.
"""

import logging
from dataclasses import dataclass
from typing import Protocol, cast

import cohere
import voyageai
from dotenv import load_dotenv
from openai import OpenAI

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingLimits:
    """Per-call batching limits imposed by an embedding provider/model.

    Attributes
    ----------
    max_tokens_per_item : int
        Per-input context limit. A single chunk over this is rejected
        before embedding (it cannot be split by the batcher).
    max_items_per_batch : int
        Maximum number of inputs per embedding API call.
    max_tokens_per_batch : int or None
        Maximum total tokens per API call. ``None`` means the provider
        imposes no separate total-token cap, so only the item count is
        enforced.
    """

    max_tokens_per_item: int
    max_items_per_batch: int
    max_tokens_per_batch: int | None


# Facts about each (provider, model) API. Values seeded from the example
# configs; TODO: confirm against provider docs and cite the source per row
# (see memo/investigate_weaviate_20260701.md, 2026-08-10).
_LIMITS: dict[tuple[str, str], EmbeddingLimits] = {
    ("openai", "text-embedding-3-small"): EmbeddingLimits(8192, 500, 250000),
    ("openai", "text-embedding-3-large"): EmbeddingLimits(8192, 500, 250000),
    ("cohere", "embed-english-v3.0"): EmbeddingLimits(512, 96, 49152),
    ("voyageai", "voyage-3"): EmbeddingLimits(8192, 128, 320000),
}

# Conservative default for a model not yet in the table: small batch, and
# enforce only the item count (no assumed total-token cap).
_FALLBACK_LIMITS = EmbeddingLimits(
    max_tokens_per_item=512,
    max_items_per_batch=96,
    max_tokens_per_batch=None,
)


def _lookup_limits(provider: str, model: str) -> EmbeddingLimits:
    """Return the limits for ``(provider, model)`` or a safe default."""
    limits = _LIMITS.get((provider, model))
    if limits is None:
        _log.warning(
            "No limits table entry for provider=%r model=%r; using "
            "conservative fallback %r.",
            provider,
            model,
            _FALLBACK_LIMITS,
        )
        return _FALLBACK_LIMITS
    return limits


class EmbeddingFn(Protocol):
    """Embeds a batch of texts and reports its API batching limits.

    ``__call__`` input and output are parallel — ``result[i]`` is the
    vector for ``texts[i]``.  Implementations must be synchronous.
    """

    limits: EmbeddingLimits

    def __call__(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts and return their vectors."""
        ...


class OpenAIEmbedder:
    """``EmbeddingFn`` backed by the OpenAI embeddings API.

    Reads ``OPENAI_API_KEY`` from the environment. ``max_retries`` is
    passed to the underlying ``OpenAI`` client.
    """

    provider = "openai"

    def __init__(
        self,
        model: str,
        dimensions: int,
        max_retries: int = 3,
    ) -> None:
        self._client = OpenAI(max_retries=max_retries)
        self._model = model
        self._dimensions = dimensions
        self.limits = _lookup_limits(self.provider, model)

    def __call__(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=self._model,
            dimensions=self._dimensions,
            input=texts,
        )
        return [r.embedding for r in response.data]


class CohereEmbedder:
    """``EmbeddingFn`` backed by the Cohere embeddings API.

    Reads ``COHERE_API_KEY`` from the environment.

    ``input_type`` distinguishes ingestion (``"search_document"``, default)
    from query-time use (``"search_query"``).
    """

    provider = "cohere"

    def __init__(
        self,
        model: str,
        input_type: str = "search_document",
    ) -> None:
        self._client = cohere.ClientV2()
        self._model = model
        self._input_type = input_type
        self.limits = _lookup_limits(self.provider, model)

    def __call__(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embed(
            texts=texts,
            model=self._model,
            input_type=self._input_type,
            embedding_types=["float"],
        )
        embeddings = response.embeddings.float_
        if embeddings is None:
            raise RuntimeError(
                "Cohere returned no float embeddings; "
                "ensure embedding_types includes 'float'."
            )
        return embeddings


class VoyageAIEmbedder:
    """``EmbeddingFn`` backed by the Voyage AI embeddings API.

    Reads ``VOYAGE_API_KEY`` from the environment.

    ``input_type`` distinguishes ingestion (``"document"``, default) from
    query-time use (``"query"``).
    """

    provider = "voyageai"

    def __init__(
        self,
        model: str,
        input_type: str = "document",
    ) -> None:
        self._client = voyageai.Client()  # type: ignore[attr-defined]
        self._model = model
        self._input_type = input_type
        self.limits = _lookup_limits(self.provider, model)

    def __call__(self, texts: list[str]) -> list[list[float]]:
        result = self._client.embed(
            texts,
            model=self._model,
            input_type=self._input_type,
        )
        # We request float embeddings; the SDK types this as float|int lists.
        return cast("list[list[float]]", result.embeddings)


def build_embedder(config: dict) -> EmbeddingFn:
    """Return an ``EmbeddingFn`` built from the top-level config dict.

    Parameters
    ----------
    config : dict
        Top-level config dict.  Reads ``config["embedding"]["provider"]``,
        ``["model"]``, and ``["dimensions"]`` (OpenAI only).  Batching
        limits are *not* read from config — they come from ``_LIMITS``.

    Raises
    ------
    ValueError
        If ``provider`` is not a known value.
    """
    load_dotenv(override=True)
    cfg = config["embedding"]
    provider = cfg["provider"]
    if provider == "openai":
        return OpenAIEmbedder(
            model=cfg["model"],
            dimensions=int(cfg["dimensions"]),
        )
    if provider == "cohere":
        return CohereEmbedder(model=cfg["model"])
    if provider == "voyageai":
        return VoyageAIEmbedder(model=cfg["model"])
    raise ValueError(f"Unknown embedding provider: {provider!r}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_dotenv(override=True)

    _TEST_TEXT = (
        "Rubin Observatory will survey the southern sky every few nights."
    )

    _CASES: list[tuple[str, EmbeddingFn]] = [
        (
            "OpenAI",
            OpenAIEmbedder(model="text-embedding-3-small", dimensions=1536),
        ),
        (
            "Cohere",
            CohereEmbedder(model="embed-english-v3.0"),
        ),
        (
            "Voyage AI",
            VoyageAIEmbedder(model="voyage-3"),
        ),
    ]

    for _provider, _embedder in _CASES:
        _log.info("--- %s ---", _provider)
        _log.info("limits=%s", _embedder.limits)
        try:
            _result = _embedder([_TEST_TEXT])
            _vec = _result[0]
            _log.info(
                "OK  dimensions=%d  first_5=%s",
                len(_vec),
                [round(v, 6) for v in _vec[:5]],
            )
        except Exception as exc:
            _log.error("FAILED  %s", exc)
