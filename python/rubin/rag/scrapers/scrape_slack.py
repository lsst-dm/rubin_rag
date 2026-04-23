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

"""Scrape Slack messages from a zip file and convert them into Langchain
documents.
"""

import gc
import json
import logging
import os
import pickle
import re
import time
from collections import defaultdict
from pathlib import Path

import requests
import yaml
from langchain_community.document_loaders import SlackDirectoryLoader
from langchain_core.documents.base import Document
from scrapers.utils import (
    batch_by_tokens,
    chunk_docs,
    load_progress,
    save_progress,
    write_batches_to_pickle,
    write_raw_to_pickle,
)
from slack_sdk import WebClient

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

slack_token = os.getenv("SLACK_API_TOKEN")

headers = {"Authorization": f"Bearer {slack_token}"}

missing_channels: set[str] = set()


def is_excluded_channel(channel_name: str) -> bool:
    """Specify if a channel is to be excluded, returns True if the channel
    should be excluded.
    """
    exclude = [
        r".*test.*",
        r".*bot.*",
        r".*tmp.*",
        r"^FC.*",
        r"^3_1415926535",
        r"^atlassian-admins",
        r"^random",
        r"^transition-to-staff-slack",
    ]
    return any(re.match(pat, channel_name) for pat in exclude)


def channel_lookup() -> dict[str, str]:
    """Create a dictionary mapping channel names to channel IDs."""
    try:
        token = os.getenv("SLACK_API_TOKEN")
        if not token:
            raise ValueError("SLACK_API_TOKEN environment variable not set.")
        client = WebClient(token=token)
    except Exception as e:
        _log.error(
            f"Error connecting to Slack API. Make sure the SLACK_API_TOKEN "
            f"env variable is set: {e}"
        )
        return {}

    lookup = {}
    cursor = None

    while True:
        try:
            result = client.conversations_list(limit=100, cursor=cursor)
            page_channels = result["channels"]
            for channel in page_channels:
                lookup[channel["name"]] = channel["id"]
            cursor = result["response_metadata"].get("next_cursor")
            if not cursor:
                break
        except Exception as e:
            _log.error(f"Error fetching channels from Slack: {e}")
            break

    return lookup


def get_channel_id(channel_name: str, lookup: dict[str, str]) -> str | None:
    """Return the channel ID for a given channel name from the lookup table.

    Parameters
    ----------
    channel_name: str
        The channel name to find the channel ID for.
    lookup: dict
        Dictionary mapping channel names to channel IDs, created by
        channel_lookup().

    Returns
    -------
    Optional(string)
        The channel ID if it is in the lookup table, otherwise None.
    """
    channel_id = lookup.get(channel_name)
    if not channel_id:
        _log.debug(f"Channel '{channel_name}' not found.")
        return None
    return channel_id


def source_2_link(timestamp: str, channel_id: str) -> str:
    """Write source information to a Slack link.

    Parameters
    ----------
    timestamp: str
        timestamp from the document metadata.
    channel_id: str
        channel ID obtained through get_channel_id.

    Returns
    -------
    str
        A direct link to the Slack message.
    """
    ts = timestamp.replace(".", "")
    return f"https://rubin-obs.slack.com/archives/{channel_id}/p{ts}"


def anonymize_mentions(text: str) -> str:
    """Anonymize user mentions in the text by replacing them with a
    placeholder.

    Parameters
    ----------
    text: str
        The page content of a Langchain document object produced by the
        Slack loader.

    Returns
    -------
    str
        The page content with all @ mentions replace with USER_ANON.
    """
    if "@" in text:
        text = re.sub(r"<@[^>]+>", "USER_ANON", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    else:
        return text


def sanitize_metadata(
    docs: list[Document], lookup: dict[str, str], *, anonymize: bool = False
) -> list[Document]:
    """Sanitize the metadata of the documents by removing user information
    and adding source and source_key parameters.

    Parameters
    ----------
    docs: list[Document]
        A list of Langchain document objects from the Slack loader.
    lookup: dict[str, str]
        Mapping of channel names to Slack channel IDs.
    anonymize: bool
        If True, remove user mentions from metadata and page content.

    Returns
    -------
    list[Document]
        A sanitized list of Langchain documents.
    """
    documents = []

    for doc in docs:
        metadata = doc.metadata
        timestamp = metadata.get("timestamp", None)
        channel_name = metadata.get("channel", None)
        if not channel_name:
            _log.info(f"Channel name {channel_name} does not exist, skipping")
            continue
        if is_excluded_channel(str(channel_name)):
            continue

        if "source" in metadata and "channel" in metadata:
            channel_id = get_channel_id(str(channel_name), lookup)
            if channel_id and timestamp:
                metadata["source"] = source_2_link(
                    timestamp=timestamp, channel_id=channel_id
                )
            elif timestamp:
                missing_channels.add(channel_name)
                metadata["source"] = f"Slack:{channel_name} - {timestamp}"
            elif channel_name:
                metadata["source"] = f"Slack:{channel_name} - unknown time"
            else:
                metadata["source"] = "Slack: unknown channel - unknown time"

        if anonymize and doc.page_content:
            doc.page_content = anonymize_mentions(doc.page_content)
            metadata.pop("user", None)

        metadata["source_key"] = "slack"

        documents.append(doc)

    return documents


def process_channel(
    channel_name: str,
    channel_id: str,
    completed_keys: set[str],
    log_path: Path,
    output_dir: Path,
    *,
    anon: bool = True,
) -> None:
    """Process a given channel using the Slack API.

    Parameters
    ----------
    channel_name: str
        Name of Slack channel to scrape.
    channel_id: str
        ID of Slack channel, found using the channel lookup table and needed
        to create URL to message.
    completed_keys: set[str]
        set of already scraped Slack channels from progress log file.
    log_path: Path
        path to log file.
    output_dir: Path
        path to output directory for pickle files.
    anon: bool
        If True, will hide all user mentions and usernames.
    """
    messages = []
    url = "https://slack.com/api/conversations.history"
    cursor = None

    while True:
        params: dict[str, str | int] = {
            "channel": channel_id,
            "limit": 200,
        }
        if cursor:
            params["cursor"] = cursor

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=10,
        )
        data = response.json()

        if not data.get("ok"):
            _log.error(
                f"Error fetching messages for {channel_name}, "
                f"channel may not exist or bot does not have access: {data}"
            )
            return

        messages.extend(data["messages"])

        cursor = data.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(1)  # avoid rate-limiting

    documents = []

    for msg in messages:
        text = msg.get("text", "")
        timestamp = msg.get("ts", "")
        if timestamp:
            source = source_2_link(timestamp, channel_id)
        else:
            source = f"Slack:{channel_id} - unknown time"

        metadata = {
            "timestamp": timestamp,
            "source_key": "slack",
            "source": source,
        }

        if anon:
            text = anonymize_mentions(text)
        else:
            metadata["user"] = msg.get("user", "")

        doc = Document(page_content=text, metadata=metadata)
        documents.append(doc)

    write_raw_to_pickle(documents, channel_name, output_dir)
    chunked = chunk_docs(documents)
    batched = batch_by_tokens(chunked)
    write_batches_to_pickle(batched, channel_name, output_dir)

    completed_keys.add(channel_name)
    save_progress(log_path, completed_keys)

    del documents, chunked, batched
    gc.collect()


def scrape_slack_bot(
    channels: list[str], output_dir: Path, *, anon: bool = True
) -> None:
    """Scrape all specified channels using the Slack API.

    Parameters
    ----------
    channels: list[str]
        list of channel names to scrape.
    output_dir: Path
        path to output directory for pickle files.
    anon: bool
        If True, hides all user mentions and usernames.
    """
    log_path = output_dir / "progress.log"

    lookup = channel_lookup()
    completed_keys = load_progress(log_path)

    for channel in channels:
        _log.info(f"Scraping contents from {channel}...")
        channel_id = get_channel_id(channel, lookup)
        if not channel_id:
            _log.error(
                f"Could not find channel ID for {channel}, skipping ingest."
            )
            continue
        process_channel(
            channel,
            channel_id,
            completed_keys,
            log_path,
            output_dir,
            anon=anon,
        )

    completed_keys.add("done")
    with Path(log_path).open("w", encoding="utf-8") as f:
        json.dump(list(completed_keys), f, indent=2)


def scrape_slack_zip(
    zipfile: Path, output_dir: Path, *, anon: bool = True
) -> None:
    """Scrape Slack using a zip file from an export stored locally.

    Parameters
    ----------
    zipfile: Path
        path to Slack export zip file.
    output_dir: Path
        path to output directory for pickle files.
    anon: bool = True
        If True, hides all user mentions and usernames.
    """
    _log.info(f"Scraping {zipfile}...")
    loader = SlackDirectoryLoader(zipfile)
    docs = loader.load()
    lookup_table = channel_lookup()
    _log.info(f"Loaded {len(docs)} documents from {zipfile}")

    documents = sanitize_metadata(docs, lookup_table, anonymize=anon)

    # Group raw docs by channel and write one raw pickle per channel
    raw_channel_docs: dict[str, list] = defaultdict(list)
    for doc in documents:
        channel = doc.metadata.get("channel", "unknown")
        raw_channel_docs[channel].append(doc)
    for channel, channel_raw_list in raw_channel_docs.items():
        write_raw_to_pickle(channel_raw_list, channel, output_dir)

    chunked_docs = chunk_docs(documents)

    channel_docs = defaultdict(list)

    # Group docs by channel
    for doc in chunked_docs:
        channel = doc.metadata.get("channel", "unknown")
        channel_docs[channel].append(doc)

    # Save each channel's docs as a pickle file
    for channel, channel_docs_list in channel_docs.items():
        output_path = output_dir / f"{channel}.pkl"
        with output_path.open("wb") as f:
            pickle.dump(channel_docs_list, f)
        _log.info(f"Saved {len(channel_docs_list)} docs to {output_path}")

    del documents, chunked_docs, channel_docs
    gc.collect()

    for channel in sorted(missing_channels):
        _log.warning(
            f"{channel} not found in lookup table. Setting source key to "
            "channel name"
        )


def scrape_slack(yaml_path: str, output_dir: str) -> None:
    """Load the Slack data file and scrape Slack.

    Parameters
    ----------
    yaml_path: str
        String of path to slack_sources.yaml
    output_dir: str
        String of path to output directory, typically a timestamped directory
        specified in run_scraping.
    """
    if slack_token is None:
        raise ValueError("SLACK_API_TOKEN environment variable is not set.")
    _log.info("using new scrape slack function")
    base_dir = Path(output_dir)

    path = Path(yaml_path)
    with path.open(mode="r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    anon = config["anonymize"]

    if config["use_zipfile"]:
        _log.info("Scraping Slack using local zip file.")
        zipfile = Path(config["zipfile"])
        scrape_slack_zip(zipfile, base_dir, anon=anon)

    else:
        _log.info("Scraping Slack using API.")
        channels = config.get("channels", [])
        scrape_slack_bot(channels, base_dir, anon=anon)
