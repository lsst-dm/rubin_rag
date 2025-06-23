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

"""Run scraping for each source. This script is resumeable, and automatically
writes in batches of pickle files under the directory of the source they were
scraped from.
"""

import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.documents import Document
from scrapers.scrape_confluence import scrape_confluence
from scrapers.scrape_discourse import scrape_discourse
from scrapers.scrape_github import scrape_github
from scrapers.scrape_jira import scrape_jira
from scrapers.scrape_lsst_bib import scrape_lsst_bib
from scrapers.scrape_refs_ads import scrape_refs_ads
from scrapers.scrape_slack import scrape_slack
from scrapers.scrape_webpage import scrape_webpage

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)


def scrape_source(
    yaml_path: Path,
    source: str,
    output_dir: Path,
) -> list[Document]:
    """Run scraper function corresponding to source, helper function for main.

    Parameters
    ----------
    yaml_path: Path
        Path to yaml file of data config (e.g. data/github_sources.yaml).
    source: str
        source to scrape (e.g. confluence, jira, etc.)
    output_dir: Path
        Path to output directory for scraped pickles.

    Returns
    -------
        scraped LangChain documents.
    """
    scraper_scripts = {
        "confluence": scrape_confluence,
        "discourse": scrape_discourse,
        "jira": scrape_jira,
        "lsst_bib": scrape_lsst_bib,
        "github": scrape_github,
        "webpage": scrape_webpage,
        "refs_ads": scrape_refs_ads,
        "slack": scrape_slack,
    }
    return scraper_scripts[source](yaml_path, output_dir)


def find_latest_rubin_rag_dir(base_path: Path) -> Path | None:
    """
    Scan `base_path` for folders matching 'rubin_rag_YYYYMMDD_HHMMSS' pattern,
    return the Path with the latest timestamp or None if none found.
    """
    pattern = re.compile(r"rubin_rag_(\d{8}_\d{6})")
    candidates = []

    for p in base_path.iterdir():
        if p.is_dir():
            m = pattern.fullmatch(p.name)
            if m:
                ts_str = m.group(1)
                try:
                    ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S").replace(
                        tzinfo=UTC
                    )
                    candidates.append((ts, p))
                except ValueError:
                    pass  # ignore folders with bad timestamp format

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def main(*, resume: bool = False) -> None:
    """Run main scraping logic. Set resume=False only if you want an entirely
    new scraping run.
    """
    sources = [
        "github",
        "jira",
        "webpage",
        "lsst_bib",
        "discourse",
        "refs_ads",
    ]
    base_dir = Path()  # or wherever you want to look for these folders

    if resume:
        latest_dir = find_latest_rubin_rag_dir(base_dir)
        if latest_dir:
            base_output_dir = latest_dir
            _log.info(f"Resuming with existing directory {base_output_dir}")
        else:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            base_output_dir = base_dir / f"rubin_rag_{timestamp}"
            base_output_dir.mkdir(parents=True, exist_ok=True)
            _log.info(
                f"No existing directory found, creating new {base_output_dir}"
            )
    else:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        base_output_dir = base_dir / f"rubin_rag_{timestamp}"
        base_output_dir.mkdir(parents=True, exist_ok=True)
        _log.info(f"Starting fresh with directory {base_output_dir}")

    for source in sources:
        start = time.time()
        yaml_path = Path(f"data/{source}_sources.yaml")
        _log.info(f"Scraping documents from {source}")

        source_output_dir = base_output_dir / source
        source_output_dir.mkdir(parents=True, exist_ok=True)

        scrape_source(yaml_path, source, source_output_dir)
        end = time.time()
        _log.info(f"Scraped {source} in {(end - start) / 60:.2f} minutes.")


if __name__ == "__main__":
    main(resume=True)
