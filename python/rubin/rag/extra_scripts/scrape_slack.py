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

import argparse
import logging
import os
import re

from langchain_community.document_loaders import SlackDirectoryLoader
from langchain_core.documents.base import Document
from slack_sdk import WebClient

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)


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
    """Return the channel ID for a given channel name from the lookup table."""
    channel_id = lookup.get(channel_name)
    if not channel_id:
        _log.warning(f"Channel '{channel_name}' not found.")
    return channel_id


def source_2_url(timestamp: str, channel_id: str) -> str:
    """Take timestamp and channel ID and return URL to chat."""
    ts = timestamp.replace(".", "")
    return f"https://rubin-obs.slack.com/archives/{channel_id}/p{ts}"


def anonymize_mentions(text: str) -> str:
    """Anonymize user mentions in the text by replacing them with a
    placeholder.

    Paramters
    ---------
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
    anonymize: bool, optional
        If True, remove user mentions from metadata and page content.

    Returns
    -------
    list[Document]
        A sanitized list of Langchain documents.
    """
    documents = []
    for doc in docs:
        metadata = doc.metadata
        timestamp = metadata.get("timestamp")
        if "source" in metadata and "channel" in metadata:
            channel_id = get_channel_id(metadata["channel"], lookup)
            if channel_id and timestamp:
                metadata["source"] = source_2_url(
                    timestamp=timestamp, channel_id=channel_id
                )

        if anonymize and doc.page_content:
            doc.page_content = anonymize_mentions(doc.page_content)
            metadata.pop("user", None)

        metadata["source_key"] = "slack"

        documents.append(doc)

    return documents


def main(zipfile: str, *, anonymize: bool = False) -> list[Document]:
    """Load and sanitize Slack documents.

    Parameters
    ----------
    zipfile: str
        A raw zip file containing Slack messages.
    anonymize: bool, optional
        Remove user mentions from metadata and page content if True.

    Returns
    -------
    list[Document]
        A list of Langchain document objects with clean metadata.
    """
    loader = SlackDirectoryLoader(zipfile)
    docs = loader.load()
    lookup_table = channel_lookup()
    _log.info(f"Loaded {len(docs)} documents from {zipfile}")

    return sanitize_metadata(docs, lookup_table, anonymize=anonymize)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Slack zipfile and sanitize content."
    )
    parser.add_argument("zipfile", help="Path to Slack zipfile")
    parser.add_argument(
        "-anon", action="store_true", help="Anonymize user mentions"
    )

    args = parser.parse_args()
    lookup_table = channel_lookup()

    docs = main(args.zipfile, anonymize=args.anon)
