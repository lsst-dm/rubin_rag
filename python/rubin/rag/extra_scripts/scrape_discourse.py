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
import os
from pathlib import Path
import time
import requests
import logging
from urllib.parse import urljoin
from langchain_core.documents.base import Document

forum_username = 'leanne'
forum_key = os.getenv("COMMUNITY_API_KEY")

SLEEP_TIME = 1.0
DISCOURSE_URL = "https://community.lsst.org/" 
HEADERS = {
    "Accept": "application/json",
    # Uncomment and set your API key/username if the forum requires it:
    "Api-Key": forum_key,
    "Api-Username": forum_username
}


def count_all_pages(discourse_url):
    """
    Counts the total number of pages on the fourm.

    Args:
        discourse_url (type): URL of the discourse forum

    Returns:
        int: The number of pages
    """
    HEADERS = {
        "Accept": "application/json",
    }    
    page_count = 1
    next_url = f"{DISCOURSE_URL}/latest.json"

    while True:
        response = requests.get(next_url, headers=HEADERS)
        if response.status_code != 200:
            # raise Exception(f"Error fetching page {page_count}: {response.status_code}")
            print(f"Error fetching page {page_count}: {response.status_code}")
            continue
        
        data = response.json()
        more_url = data.get("topic_list", {}).get("more_topics_url")
 
        if not more_url:
            break
        
        next_url = urljoin(DISCOURSE_URL, more_url)
        page_count += 1

    print(f"\nTotal pages found: {page_count}")
    return page_count


def get_latest_topics(page):
    """
    Get the latest topics on a page

    Args:
        page (str): The page

    Returns:
        int: The list of topics
    """
    url = f"{DISCOURSE_URL}/latest.json?page={page}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        print(f"Failed to fetch page {page}: {response.status_code}")
        return []
    data = response.json()
    return data.get("topic_list", {}).get("topics", [])


def get_posts_for_topic(topic_id):
    """
    Get all posts for a topic topics

    Args:
        discourse_url (type): URL of the discourse forum
        topic_id (int): The topic ID

    Returns:
        dict: The posts
    """
    url = f"{DISCOURSE_URL}/t/{topic_id}.json"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        raise Exception(f"HTTP {response.status_code}")
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
        "posts": cleaned_posts
    }


def clean_post(post):
    """
    Format the data in a post

    Args:
        post (str): The post

    Returns:
        str: The well-formatted post
    """
    return {
        "post_id": post.get("id"),
        "username": post.get("username", "unknown"),
        "created_at": post.get("created_at"),
        "cooked": post.get("cooked", ""),
        "raw": post.get("raw", ""),
    }


def clean_filename(name):
    """
    Format a filename

    Args:
        name (str): The filename

    Returns:
        str: The well-formatted filename
    """
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def scrape_all_topics(MAX_PAGES, OUTPUT_DIR):
    seen_topic_ids = set()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for page in range(0, MAX_PAGES):
        topics = get_latest_topics(page)
        if not topics:
            print("No more topics found. Done.")
            break

        for topic in topics:
            topic_id = topic["id"]
            if topic_id in seen_topic_ids:
                continue
            seen_topic_ids.add(topic_id)

            try:
                topic_data = get_posts_for_topic(topic_id)
                filename = f"{topic_id}_{clean_filename(topic_data['topic_title'][:50])}.json"
                path = os.path.join(OUTPUT_DIR, filename)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(topic_data, f, indent=2, ensure_ascii=False)
                time.sleep(SLEEP_TIME)
            except Exception as e:
                print(f"Error fetching topic {topic_id}: {e}")
                continue


def topics_to_docs(topics):
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

def scrape_and_aggregate(MAX_PAGES, OUTPUT_FILE):
    all_posts = []
    seen_topic_ids = set()

    for page in range(0, MAX_PAGES):
        print('WORKING ON PAGE ', page + 1, ' OF ', MAX_PAGES)
        topics = get_latest_topics(page)
        if not topics:
            print("No more topics found.")
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
                print(f"Error fetching topic {topic_id}: {e}")
                continue
                
    # Write everything to a single file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_posts, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(all_posts)} posts to {OUTPUT_FILE}")

    return all_posts
