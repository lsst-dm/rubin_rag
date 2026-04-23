"""Generate a synthetic RAGAS testset from scraped documents.

Workflow
--------
1. Load documents from a pickle file (list of LangChain Documents).
2. Generate synthetic QA pairs (questions + reference answers) with RAGAS.
3. Save the testset to a JSON file for use with ragas_eval.py.

Usage
-----
    python ragas_generate_testset.py \
        --pkl-path rubin_rag_20260423_005800/jira/SP_01.pkl \
        --testset-size 20 \
        --output testset.json

Required environment variables
-------------------------------
    OPENAI_API_KEY  — used for testset generation
"""

import argparse
import json
import logging
import os
import pickle
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents.base import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.testset import TestsetGenerator

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
)
_log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


def load_docs(pkl_path: Path) -> list[Document]:
    """Load LangChain Documents from a single pickle file."""
    with pkl_path.open("rb") as fh:
        docs: list[Document] = pickle.load(fh)  # noqa: S301
    non_empty = [d for d in docs if d.page_content and d.page_content.strip()]
    _log.info(
        "Loaded %d docs (%d non-empty) from %s",
        len(docs),
        len(non_empty),
        pkl_path.name,
    )
    return non_empty


def generate_testset(
    pkl_path: Path, testset_size: int, output_path: Path
) -> None:
    openai_api_key = os.environ["OPENAI_API_KEY"]

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", dimensions=1536, api_key=openai_api_key
    )

    docs = load_docs(pkl_path)

    _log.info(
        "Generating synthetic test set (%d samples) from %d docs…",
        testset_size,
        len(docs),
    )
    generator = TestsetGenerator.from_langchain(
        llm=llm, embedding_model=embeddings
    )
    testset = generator.generate_with_langchain_docs(
        docs,
        testset_size=testset_size,
        raise_exceptions=False,
    )
    _log.info("Generated %d test samples", len(testset.samples))

    samples = []
    for i, test_sample in enumerate(testset.samples):
        eval_s = test_sample.eval_sample
        question = getattr(eval_s, "user_input", None)
        reference = getattr(eval_s, "reference", None)

        if not question:
            _log.warning("Sample %d has no user_input, skipping.", i + 1)
            continue
        if not reference:
            _log.warning("Sample %d has no reference answer, skipping.", i + 1)
            continue

        samples.append({"user_input": question, "reference": reference})

    output_path.write_text(json.dumps(samples, indent=2))
    _log.info("Saved %d samples to %s", len(samples), output_path)
    print(f"\nTestset saved: {output_path} ({len(samples)} samples)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate RAGAS synthetic testset from scraped documents"
    )
    p.add_argument(
        "--pkl-path",
        default="rubin_rag_20260423_005800/jira/SP_01.pkl",
        help="Path to the pickle file of scraped documents",
    )
    p.add_argument(
        "--testset-size",
        type=int,
        default=20,
        help="Number of synthetic QA pairs to generate (default: 20)",
    )
    p.add_argument(
        "--output",
        default="testset.json",
        help="Output JSON path (default: testset.json)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_testset(
        pkl_path=Path(args.pkl_path),
        testset_size=args.testset_size,
        output_path=Path(args.output),
    )
