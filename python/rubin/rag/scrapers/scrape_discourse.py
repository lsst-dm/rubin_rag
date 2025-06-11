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
# along with this program.  If not, see <https://www.gnu.org/licenses/>.```

"""Utilities for scraping Discourse forum contents into LangChain documents."""

import json
import logging
import os
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from langchain_core.documents.base import Document
from scrapers.utils import (
    batch_by_tokens,
    chunk_docs,
    load_progress,
    save_progress,
    write_batches_to_pickle,
)

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

forum_username = "leanne"
forum_key = os.getenv("COMMUNITY_API_KEY")

SLEEP_TIME = 1.0
DISCOURSE_URL = "https://community.lsst.org/"
headers = {
    "Accept": "application/json",
    "Api-Key": forum_key,
    "Api-Username": forum_username,
}


def count_all_pages(discourse_url: str = DISCOURSE_URL) -> int:
    """
    Count the total number of pages on the forum.

    Parameters
    ----------
    discourse_url : str
        URL of the discourse forum.

    Returns
    -------
    page_count : int
        the number of pages of Discourse API content.
    """
    headers = {
        "Accept": "application/json",
    }
    page_count = 1
    next_url = f"{discourse_url}/latest.json"

    while True:
        response = requests.get(next_url, headers=headers, timeout=10)
        if response.status_code != 200:
            _log.error(
                f"Error fetching page {page_count}: {response.status_code}"
            )
            continue

        data = response.json()
        more_url = data.get("topic_list", {}).get("more_topics_url")

        if not more_url:
            break

        next_url = urljoin(discourse_url, more_url)
        page_count += 1

    _log.info(f"\nTotal pages found: {page_count}")
    return page_count


def get_latest_topics(page: int) -> list:
    """
    Get the latest topics on a page.

    Parameters
    ----------
    page : int
        the page number within the Discourse API results.

    Returns
    -------
    list
        the list of Discourse topics found within the page.
    """
    url = f"{DISCOURSE_URL}/latest.json?page={page}"
    response = requests.get(url, headers=headers, timeout=10)
    if response.status_code != 200:
        _log.warning(f"Failed to fetch page {page}: {response.status_code}")
        return []
    data = response.json()
    return data.get("topic_list", {}).get("topics", [])


class StatusCodeError(Exception):
    """Custom exception class for bad requests response status code."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


def clean_post(post: dict) -> dict:
    """
    Format the data in a Discourse post.

    Parameters
    ----------
    post : dict
        The post as retrieved from the Discourse API.

    Returns
    -------
    dict
        A well-formatted dictionary representing the post.
    """
    return {
        "post_id": post.get("id"),
        "username": post.get("username", "unknown"),
        "created_at": post.get("created_at"),
        "cooked": post.get("cooked", ""),
        "raw": post.get("raw", ""),
    }


def get_posts_for_topic(topic_id: int) -> dict:
    """
    Get all posts for a Discourse topic.

    Parameters
    ----------
    topic_id : int
        The discourse topic identifier, which is an integer.

    Returns
    -------
    dict
        A dictionary containing the posts for the relevant
        Discourse topic.
    """
    url = f"{DISCOURSE_URL}/t/{topic_id}.json"
    response = requests.get(url, headers=headers, timeout=10)
    if response.status_code != 200:
        raise StatusCodeError(f"HTTP {response.status_code}")
    data = response.json()
    topic_title = data.get("title", f"topic_{topic_id}")
    posts = data.get("post_stream", {}).get("posts", [])
    cleaned_posts = []

    for post in posts:
        cleaned = clean_post(post)
        cleaned_posts.append(cleaned)

    return {
        "topic_id": topic_id,
        "topic_title": topic_title,
        "posts": cleaned_posts,
    }


def topics_to_docs(topics: list) -> list:
    """Convert Discourse topics to a list of LangChain docs.

    Parameters
    ----------
    topics : list
        list of Discourse topics, where each topic is a dictionary.

    Returns
    -------
    docs : list
       list of LangChain documents (not chunked).
    """
    docs = []
    for topic in topics:
        for post in topic["posts"]:
            if post["cooked"] == "":
                continue
            metadata = post.copy()
            del metadata["cooked"]
            metadata["source_key"] = "discourse"
            metadata["topic_id"] = topic["topic_id"]
            metadata["topic_title"] = topic["topic_title"]
            # https://community.lsst.org/t/10138
            metadata["source"] = f"{DISCOURSE_URL}/t/{topic['topic_id']}"
            doc = Document(page_content=post["cooked"], metadata=metadata)
            docs.append(doc)

    return docs


def scrape_discourse(output_dir: str, *, max_pages: int | None = None) -> None:
    """Scrape many Discourse pages/topics.

    Parameters
    ----------
    output_dir : str
        Name of output directory for pickle files and log file.
    max_pages : int
        Maximum page number of Discourse API content to scrape,
        starting from page number 0. If not specified, max_pages
        will be set to the total number of pages available.
    """
    n_pages_total = count_all_pages(DISCOURSE_URL)
    if max_pages is None:
        max_pages = n_pages_total
    max_pages = min(max_pages, n_pages_total)
    seen_topic_ids = set()

    base_dir = Path(output_dir)
    log_path = base_dir / "progress.log"
    completed_keys = load_progress(log_path)

    for page in range(max_pages):
        space_key = f"page{page}"
        if space_key in completed_keys:
            continue
        _log.info(f"WORKING ON PAGE {page + 1} OF {max_pages}")
        all_posts = []
        topics = get_latest_topics(page)
        if not topics:
            _log.info("No more topics found.")
            break

        for topic in topics:
            topic_id = topic["id"]
            if topic_id in seen_topic_ids:
                continue
            seen_topic_ids.add(topic_id)

            try:
                posts = get_posts_for_topic(topic_id)
                all_posts.extend([posts])
                time.sleep(SLEEP_TIME)
            except Exception as e:
                _log.error(f"Error fetching topic {topic_id}: {e}")
                continue

        all_docs = topics_to_docs(all_posts)

        chunked = chunk_docs(all_docs)
        batched = batch_by_tokens(chunked)
        write_batches_to_pickle(batched, space_key, base_dir)
        completed_keys.add(space_key)
        save_progress(log_path, completed_keys)

        _log.info(f"Saved {len(all_docs)} posts to {output_dir}")

    if max_pages == n_pages_total:
        completed_keys.add("done")
    with Path.open(log_path, "w", encoding="utf-8") as f:
        json.dump(list(completed_keys), f, indent=2)
