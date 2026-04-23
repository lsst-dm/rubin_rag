"""Retrieval quality evaluation for Weaviate using a pre-generated RAGAS testset.

Workflow
--------
1. Load a pre-generated testset from a JSON file (produced by ragas_generate_testset.py).
2. For each question, retrieve contexts from Weaviate.
3. Evaluate retrieval quality with RAGAS metrics:
     - ContextPrecision  : of retrieved chunks, how many are useful? (precision)
     - LLMContextRecall  : does retrieved context cover the reference answer? (recall)

No LLM answer generation is needed for these two retrieval-only metrics.

Usage
-----
    # Discover available Weaviate collections first:
    python ragas_eval.py --list-collections

    # Run evaluation:
    python ragas_eval.py \
        --testset-path testset.json \
        --collection LangChain_xxxx \
        --output ragas_results.csv

Required environment variables
-------------------------------
    OPENAI_API_KEY      — used for RAGAS evaluation metrics
    WEAVIATE_API_KEY    — Weaviate auth key

Weaviate connection (pick one):
    Option A — Weaviate Cloud Service (WCS):
        WEAVIATE_CLOUD_URL   e.g. https://your-cluster.weaviate.network

    Option B — custom in-cluster host:
        HTTP_HOST            e.g. weaviate-headless.rubin-rag.svc.cluster.local
        GRPC_HOST            e.g. weaviate-grpc.rubin-rag.svc.cluster.local
"""

import argparse
import json
import logging
import os
from pathlib import Path

import weaviate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics import ContextPrecision, LLMContextRecall
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
)
_log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Weaviate connection
# ---------------------------------------------------------------------------


def connect_weaviate() -> weaviate.WeaviateClient:
    """Connect to Weaviate using environment variables.

    Tries WEAVIATE_CLOUD_URL first (WCS), then falls back to
    HTTP_HOST / GRPC_HOST for an in-cluster custom connection.
    """
    api_key = os.environ["WEAVIATE_API_KEY"]
    openai_key = os.environ["OPENAI_API_KEY"]
    cloud_url = os.getenv("WEAVIATE_CLOUD_URL")

    if cloud_url:
        _log.info("Connecting to Weaviate Cloud at %s", cloud_url)
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=cloud_url,
            auth_credentials=Auth.api_key(api_key),
            headers={"X-OpenAI-Api-Key": openai_key},
        )

    http_host = os.environ["HTTP_HOST"]
    grpc_host = os.environ["GRPC_HOST"]
    _log.info("Connecting to custom Weaviate at %s", http_host)
    return weaviate.connect_to_custom(
        http_host=http_host,
        http_port=8080,
        http_secure=False,
        grpc_host=grpc_host,
        grpc_port=50051,
        grpc_secure=False,
        auth_credentials=Auth.api_key(api_key),
        headers={"X-OpenAI-Api-Key": openai_key},
        skip_init_checks=True,
    )


def list_collections() -> None:
    """Print all collection names in the connected Weaviate instance."""
    client = connect_weaviate()
    try:
        collections = client.collections.list_all()
        if not collections:
            print("No collections found.")
        else:
            print("Available collections:")
            for name in collections:
                print(f"  {name}")
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Testset loading
# ---------------------------------------------------------------------------


def load_testset(testset_path: Path) -> list[dict]:
    """Load a pre-generated testset from a JSON file."""
    samples = json.loads(testset_path.read_text())
    _log.info("Loaded %d samples from %s", len(samples), testset_path.name)
    return samples


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def retrieve_contexts(
    question: str,
    client: weaviate.WeaviateClient,
    collection_name: str,
    k: int = 6,
    source_key: str | None = "jira",
) -> list[str]:
    """Hybrid search on Weaviate, returning the top-k page_content strings."""
    from weaviate.classes.query import Filter

    collection = client.collections.get(collection_name)

    filters = None
    if source_key:
        filters = Filter.by_property("source_key").equal(source_key)

    response = collection.query.hybrid(
        query=question,
        limit=k,
        filters=filters,
        alpha=1,  # pure vector; set 0.5 for BM25+vector hybrid
        return_metadata=MetadataQuery(score=True),
    )
    return [obj.properties.get("page_content", "") for obj in response.objects]


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------


def run_evaluation(
    testset_path: Path,
    collection_name: str,
    output_path: Path,
    source_key: str | None = "jira",
    retrieval_k: int = 6,
) -> None:
    """Load pre-generated testset, retrieve from Weaviate, and evaluate retrieval."""
    openai_api_key = os.environ["OPENAI_API_KEY"]

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", dimensions=1536, api_key=openai_api_key
    )

    # --- Step 1: load testset -----------------------------------------------
    testset = load_testset(testset_path)

    # --- Step 2: retrieve contexts from Weaviate ----------------------------
    client = connect_weaviate()
    samples: list[SingleTurnSample] = []

    try:
        for i, entry in enumerate(testset):
            question = entry.get("user_input")
            reference = entry.get("reference")

            if not question:
                _log.warning("Sample %d has no user_input, skipping.", i + 1)
                continue
            if not reference:
                _log.warning(
                    "Sample %d has no reference answer, skipping.", i + 1
                )
                continue

            _log.info(
                "[%d/%d] Retrieving for: %s",
                i + 1,
                len(testset),
                question[:80],
            )
            contexts = retrieve_contexts(
                question,
                client,
                collection_name,
                k=retrieval_k,
                source_key=source_key,
            )

            if not contexts:
                _log.warning(
                    "No contexts retrieved for sample %d, skipping.", i + 1
                )
                continue

            samples.append(
                SingleTurnSample(
                    user_input=question,
                    retrieved_contexts=contexts,
                    reference=reference,
                )
            )
    finally:
        client.close()

    if not samples:
        _log.error("No samples with retrieved contexts — cannot evaluate.")
        return

    _log.info("Running RAGAS evaluation on %d samples…", len(samples))

    # --- Step 4: evaluate retrieval -----------------------------------------
    eval_dataset = EvaluationDataset(samples=samples)
    result = evaluate(
        dataset=eval_dataset,
        metrics=[LLMContextRecall(), ContextPrecision()],
        llm=llm,
        embeddings=embeddings,
    )

    # --- Step 5: save and print results -------------------------------------
    df = result.to_pandas()
    df.to_csv(output_path, index=False)
    _log.info("Results saved to %s", output_path)

    print("\n=== RAGAS Retrieval Evaluation ===")
    for col in ["llm_context_recall", "context_precision"]:
        if col in df.columns:
            print(
                f"  {col:<25} {df[col].mean():.4f}  (mean over {len(df)} samples)"
            )
    print(f"\nFull results: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RAGAS retrieval quality evaluation for Rubin RAG"
    )
    p.add_argument(
        "--list-collections",
        action="store_true",
        help="Print available Weaviate collections and exit",
    )
    p.add_argument(
        "--testset-path",
        default="testset.json",
        help="Path to the testset JSON file produced by ragas_generate_testset.py",
    )
    p.add_argument(
        "--collection",
        default=None,
        help="Weaviate collection name (required unless --list-collections)",
    )
    p.add_argument(
        "--output",
        default="ragas_results.csv",
        help="Output CSV path (default: ragas_results.csv)",
    )
    p.add_argument(
        "--source-key",
        default="jira",
        help="Filter Weaviate retrieval to this source_key (default: jira, pass '' to skip)",
    )
    p.add_argument(
        "--retrieval-k",
        type=int,
        default=6,
        help="Number of chunks to retrieve per question (default: 6)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.list_collections:
        list_collections()
    else:
        if not args.collection:
            raise SystemExit(
                "error: --collection is required (run --list-collections to discover it)"
            )
        run_evaluation(
            testset_path=Path(args.testset_path),
            collection_name=args.collection,
            output_path=Path(args.output),
            source_key=args.source_key or None,
            retrieval_k=args.retrieval_k,
        )
