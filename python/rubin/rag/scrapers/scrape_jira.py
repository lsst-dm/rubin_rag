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

"""Fetch and process JIRA ticket data, extract relevant details, and
save results in a structured JSON format.
"""

import gc
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import requests
import yaml
from langchain_core.documents.base import Document
from requests.auth import AuthBase
from scrapers.utils import (
    batch_by_tokens,
    chunk_docs,
    load_progress,
    sanitize_dates,
    save_progress,
    write_batches_to_pickle,
)

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

username = str(os.getenv("CONFLUENCE_USERNAME"))
api_token = str(os.getenv("CONFLUENCE_API_TOKEN"))
if username == "None":
    raise ValueError("Missing CONFLUENCE_USERNAME")
if api_token == "None":
    raise ValueError("Missing CONFLUENCE_API_TOKEN")


def get_with_retries(
    url: str,
    auth: AuthBase,
    headers: dict[str, str],
    timeout: int = 10,
    retries: int = 3,
    delay: int = 5,
) -> requests.Response | None:
    """Request from URL with retries on failure.

    Parameters
    ----------
    url : str
        Link to website.
    auth : AuthBase
        Authentication info, e.g., HTTPBasicAuth(email, token).
    headers : dict
        HTTP headers to include in the request.
    timeout : int
        Request timeout in seconds.
    retries : int
        Number of retry attempts.
    delay : int
        Delay in seconds between retries.

    Returns
    -------
    requests.Response
        The HTTP response.
    """
    for attempt in range(retries):
        try:
            return requests.get(
                url, auth=auth, headers=headers, timeout=timeout
            )
        except requests.exceptions.ReadTimeout as e:
            if attempt < retries - 1:
                time.sleep(delay)
                continue
            raise requests.exceptions.RequestException(
                f"GET request to {url} failed after {retries} retries."
            ) from e
    return None


def get_jira_issue(
    issue_name: str,
    email: str = username,
    api_token: str = api_token,
) -> tuple:
    """Get the JIRA issue data from the JIRA API.

    Parameters
    ----------
    issue_name : str
        name of the Jira issue including the prefix and dash e.g., DM-40000
    api_token : str
        Jira API token
    email : str
        email address of Jira account associated with the API token

    Returns
    -------
    tuple
        The second of two elements is a status code; None if successful,
        otherwise a string including the error's status code. If successful,
        the first element is a JSON dict. If unsuccessful, the first element
        is either an empty list or None depending on the status code.
    """
    url = f"https://rubinobs.atlassian.net/rest/api/latest/issue/{issue_name}"
    auth = requests.auth.HTTPBasicAuth(email, api_token)
    headers = {"Content-Type": "application/json"}
    response = get_with_retries(url, auth, headers, timeout=5, retries=2)

    if response and response.status_code == 200:
        return response.json(), None
    if response and response.status_code == 429:
        return [], f"{issue_name}: {response.status_code}"
    if not response:
        return None, f"{issue_name}: request to URL failed."
    else:
        return None, f"{issue_name}: {response.status_code}"


def extract_reviewer_from_customfield(jira_data: dict) -> list:
    """Extract the reviewer(s) from the customfield_10048.

    Parameters
    ----------
    jira_data : dict
        Dictionary of Jira issue data as returned by the Jira API.

    Returns
    -------
    list
        List of Jira reviewer names (first and last) for this issue.
        Returns a single-element list if there is one reviewer, and
        also returns a single element list if no reviewers assigned.
    """
    # Extract reviewer information from customfield_10048 if available
    reviewers = jira_data["fields"].get("customfield_10048", [])
    if reviewers:
        # Extract the display name(s) of the reviewer(s) into a list
        return [
            reviewer.get("displayName", "Reviewer not found")
            for reviewer in reviewers
        ]
    return ["No reviewer assigned"]


def extract_related_issues(jira_data: dict) -> list:
    """Extract the related issues from the JIRA data.

    Parameters
    ----------
    jira_data : dict
        Dictionary of Jira issue data as returned by the Jira API.

    Returns
    -------
    related_issues : list
        List of related issues, each of which is a summirized with
        a dict. Empty list is returned if there are no related
        issues.
    """
    related_issues = []
    issue_links = jira_data["fields"].get("issuelinks", [])

    for link in issue_links:
        issue_relation = {}
        if "inwardIssue" in link:
            issue_relation = {
                "key": link["inwardIssue"].get("key", ""),
                "summary": link["inwardIssue"]["fields"].get(
                    "summary", "No summary"
                ),
                "status": link["inwardIssue"]["fields"]["status"].get(
                    "name", "Unknown status"
                ),
                "relationship": link["type"].get(
                    "outward", "Unknown relation"
                ),
            }
        elif "outwardIssue" in link:
            issue_relation = {
                "key": link["outwardIssue"].get("key", ""),
                "summary": link["outwardIssue"]["fields"].get(
                    "summary", "No summary"
                ),
                "status": link["outwardIssue"]["fields"]["status"].get(
                    "name", "Unknown status"
                ),
                "relationship": link["type"].get(
                    "outward", "Unknown relation"
                ),
            }
        related_issues.append(issue_relation)

    return related_issues


def extract_components(jira_data: dict) -> list:
    """Extract the components from the JIRA data.

    Parameters
    ----------
    jira_data : dict
        Dictionary of Jira issue data as returned by the Jira API.

    Returns
    -------
    list
        List of components, each of which is a string. Empty list returned
        if Jira issue has no components specified.
    """
    components = jira_data["fields"].get("components", [])
    return [component.get("name", "No component") for component in components]


def extract_comments(jira_data: dict) -> list:
    """Extract the comments from the JIRA data.

    Parameters
    ----------
    jira_data : dict
        Dictionary of Jira issue data as returned by the Jira API.

    Returns
    -------
    list
        List of comments, each of which is represented with a dictionary.
        Each comment dictionary contains the comment author name and
        comment text. Empty list returned if Jira issue has no comments.
    """
    comments = jira_data["fields"].get("comment", {}).get("comments", [])
    return [
        {
            "author": comment["author"].get("displayName", "Unknown author"),
            "body": comment.get("body", "No comment body"),
        }
        for comment in comments
    ]


def extract_parent_issue(jira_data: dict) -> dict:
    """Extract the parent issue from the JIRA data.

    Parameters
    ----------
    jira_data : dict
        Dictionary of Jira issue data as returned by the Jira API.

    Returns
    -------
    dict
        Parent issue represented as a dictionary. Empty dictionary
        returned if no parent issue found.
    """
    parent_issue = jira_data["fields"].get("parent", None)
    if parent_issue:
        return {
            "key": parent_issue.get("key", ""),
            "summary": parent_issue["fields"].get("summary", "No summary"),
            "status": parent_issue["fields"]["status"].get(
                "name", "Unknown status"
            ),
        }
    return {}


def safe_get(
    dictionary: dict[str, Any], keys: list, default: Any | None = None
) -> Any:
    """Safely extract information from nested dictionaries.

    Parameters
    ----------
    dictionary : dict
        Dictionary of with string keys and values of any type.
    keys : list
        List of string keys for which to attempt to extract
        values from the input (nested) dictionary.
    default : Any
        Default value to return in the event that retrieving the
        value for a key fails. Defaults to None.

    Returns
    -------
    Any
        Value found corresponding to (nested) key(s). Returns default
        if key(s) not found.
    """
    d = dictionary
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return default
    return d


def reformat_jira_data(jira_data: dict, ticket: str) -> dict:
    """Reformat the JIRA data into a simplified dictionary.

    Parameters
    ----------
    jira_data : dict
        Dictionary of Jira issue data as returned by the Jira API.
    ticket : str
        Name of Jira issue.

    Returns
    -------
    dict
        reformatted/simplified dictionary representing the Jira ticket
        data/metadata, if input Jira data is not an empty dictionary.
        Otherwise a placeholder dictionary with the same structure but
        no meaningful content in the values.
    """
    if not jira_data:
        # If jira_data is None or empty, return a default
        # dictionary with the error message
        return {
            "key": ticket,
            "summary": "No data available",
            "description": "Unauthorized or no data available",
            "status": "Unknown",
            "assignee": "Unassigned",
            "reviewers": ["No reviewer assigned"],
            "reporter": "Unknown",
            "created": "N/A",
            "updated": "N/A",
            "resolution": "Unresolved",
            "labels": [],
            "attachments": [],
            "comments": [{"author": "Unknown", "body": "No comments"}],
            "parent_issue": None,
            "related_issues": [],
            "components": ["No components"],
            "team": "No team",
            "project": "No project",
        }

    return {
        "key": jira_data.get("key", ""),
        "summary": jira_data["fields"].get("summary", ""),
        "description": jira_data["fields"].get("description", ""),
        "status": jira_data["fields"]["status"].get("name", "Unknown"),
        "assignee": safe_get(
            jira_data["fields"], ["assignee", "displayName"], "Unassigned"
        ),
        "reviewers": extract_reviewer_from_customfield(jira_data),
        "reporter": safe_get(
            jira_data["fields"], ["reporter", "displayName"], "Unknown"
        ),
        "created": jira_data["fields"].get("created", ""),
        "updated": jira_data["fields"].get("updated", ""),
        "resolution": safe_get(
            jira_data["fields"], ["resolution", "name"], "Unresolved"
        ),
        "labels": jira_data["fields"].get("labels", []),
        "attachments": [
            {
                "filename": attachment.get("filename", ""),
                "url": attachment.get("content", ""),
            }
            for attachment in jira_data["fields"].get("attachment", [])
        ],
        "comments": extract_comments(jira_data),
        "parent_issue": extract_parent_issue(
            jira_data
        ),  # Parent issue extraction
        "related_issues": extract_related_issues(
            jira_data
        ),  # Related issues extraction
        "components": extract_components(jira_data),  # Extract components
        "team": safe_get(
            jira_data["fields"], ["customfield_10056", "value"], "No team"
        ),
        "project": safe_get(
            jira_data["fields"], ["project", "name"], "No project"
        ),
    }


def write_to_file(
    results: dict,
    folder: str = ".",
) -> None:
    """Write the JIRA ticket data to a JSON file.

    Parameters
    ----------
        results : dict
            dictionary of Jira ticket data and metadata
        folder : str
            base output folder within which to write the JSON file. The
            directory into which write-out happens is a subdirectory of
            the specified folder, where the subdirectory name is the Jira
            ticket's prefix. The output directory will be created if it
            doesn't already exist. Using the default folder, the output
            JSON is written into ./DM for a DM- prefixed ticket name.
    """
    # Extract the prefix from the key (letters before '-')
    ticket_key = results["key"]  # Assuming 'key' is something like 'DM-12345'
    prefix = ticket_key.split("-")[0]  # Get the letters before the '-'

    # Construct the new folder path by appending the prefix
    folder_with_prefix = Path(folder) / Path(prefix)

    # Ensure the folder exists
    if not Path.exists(folder_with_prefix):
        Path.mkdir(folder_with_prefix)

    ticket_key = results["key"]  # Assuming 'key' is the Jira ticket key
    file_path = Path(
        folder_with_prefix / f"{ticket_key}.json"
    )  # Create the JSON file path

    # Write the individual result to a JSON file
    with Path.open(file_path, "w") as f:
        json.dump(
            results, f, indent=4
        )  # Writing with indentation for readability


def fetch_ticket(
    ticket: str,
    email: str = username,
    api_token: str = api_token,
) -> tuple:
    """Fetch and reformat the ticket data from JIRA.

    Parameters
    ----------
        ticket : str
            name of the Jira issue including the prefix and dash e.g., DM-40000
        email : str
            email address of Jira account associated with the API token
        api_token : str
            Jira API token

    Returns
    -------
        dict
            reformatted/simplified dictionary representing the Jira ticket
            data/metadata, if successful. Otherwise a placeholder dictionary
            with the same structure but no meaningful content in the values.
        error_message : str
            None if successful, otherwise a string with the ticket name and
            error message.
    """
    _log.debug(f"Fetching Jira issue {ticket}")
    jira_data, error_message = get_jira_issue(ticket, email, api_token)
    return reformat_jira_data(jira_data, ticket), error_message


def retry_fetch_ticket(
    ticket: str,
    email: str = username,
    api_token: str = api_token,
    max_retries: int = 5,
) -> tuple:
    """Fetch Jira ticket with retry logic.

    Parameters
    ----------
    ticket : str
        name of the Jira issue including the prefix and dash e.g., DM-40000
    email : str
        email address of Jira account associated with the API token
    api_token : str
        Jira API token
    max_retries : int
        maximum number of attempts at fetching the Jira ticket

    Returns
    -------
    tuple
        Two-element tuple. If successful, the first element is a dict
        with the ticket data/metadata and the second element is None. If
        unsuccessful, the first element is either a default dictionary
        or None (if exceptions encountered on all retries), and the
        second element is a string error message.
    """
    for attempt in range(max_retries):
        try:
            result, error_message = fetch_ticket(ticket, email, api_token)
        except Exception:
            _log.warning(f"Failure #{attempt + 1} retrieving {ticket}.")
            if attempt + 1 == max_retries:
                raise  # Raise the error if max retries reached
            time.sleep(2**attempt + 2)  # Exponential backoff
        else:
            if error_message is not None:
                _log.warning(f"Error fetching Jira issue {error_message}")
            return result, error_message

    _log.warning(
        f"Failed to fetch ticket {ticket} after max number of retries."
    )
    return None, "Failed to fetch ticket"


def jira_to_document(jira_data: dict) -> Document:
    """Convert Jira ticket dictionary data to a LangChain Document.

    Parameters
    ----------
    jira_data : dict
        reformatted/simplified dictionary representing the Jira ticket
        data/metadata. Needs to have 'description' key.

    Returns
    -------
    Document
        A Document (langchain_core.documents.base.Document) object formed
        by considering everything other than the 'description' key within
        the Jira data dictionary to be metadata.
    """
    metadata = jira_data.copy()
    del metadata["description"]

    if jira_data["description"] is None:
        jira_data["description"] = ""

    return Document(page_content=jira_data["description"], metadata=metadata)


def jira_tickets_from_list(
    ticket_list: list,
    email: str = username,
    api_token: str = api_token,
    folder: str = ".",
    max_retries: int = 5,
    *,
    write: bool = False,
) -> tuple:
    """Ingest a list of Jira tickets into LangChain documents.

    Parameters
    ----------
    ticket_list : list
        list of names of the Jira issues to ingest including the
        prefix and dash in each case e.g., DM-40000. Each list
        element is a string.
    api_token : str
        Jira API token
    email : str
        email address of Jira account associated with the API token
    folder : str
        base output folder within which to write the JSON file. The
        directory into which write-out happens is a subdirectory of
        the specified folder, where the subdirectory name is the Jira
        ticket's prefix. The output directory will be created if it
        doesn't already exist. Using the default folder, the output
        is written into ./DM for a DM- prefixed ticket name. Unused
        if write is set to False.
    max_retries : int
        maximum number of attempts at fetching each Jira ticket.
        Defaults to 5.
    write : bool
        keyword-only argument. Whether or not to write out each
        successfully downloaded Jira issue to a file.

    Returns
    -------
    tuple
        two-element tuple. The first element is the list of
        LangChain documents for successfully retrieved Jira tickets.
        The second element of the returned tuple is a list of Jira
        tickets that were not fetched successfully. Each element of
        this list is a two-element tuple, where the first tuple
        element is the string ticket name (including prefix and dash)
        and the second tuple element is the corresponding string
        error message.
    """
    docs: list = []
    failures: list = []

    for ticket_name in ticket_list:
        jira_data, status = retry_fetch_ticket(
            ticket_name, email, api_token, max_retries
        )
        # only output the results if fetching was successful
        if status is None:
            docs.append(jira_to_document(jira_data))
            if write:
                write_to_file(jira_data, folder)
        else:
            failures.append((ticket_name, status))

    return sanitize_metadata(docs), failures


def jira_tickets_in_range(
    ticket_prefix: str,
    min_ticket_num: int,
    max_ticket_num: int,
    email: str = username,
    api_token: str = api_token,
    folder: str = ".",
    max_retries: int = 5,
    *,
    write: bool = False,
) -> tuple:
    """Ingest a numerical range of Jira tickets into LangChain documents.

    Parameters
    ----------
    ticket_prefix : str
        valid ticket prefix such as 'DM' without a trailing dash
    min_ticket_num : int
        minimum ticket number (inclusive)
    max_ticket_num : int
        maximum ticket number (inclusive)
    api_token : str
        Jira API token
    email : str
        email address of Jira account associated with the API token
    folder : str
        base output folder within which to write the JSON file. The
        directory into which write-out happens is a subdirectory of
        the specified folder, where the subdirectory name is the Jira
        ticket's prefix. The output directory will be created if it
        doesn't already exist. Using the default folder, the output
        is written into ./DM for a DM- prefixed ticket name. Unused
        if write is set to False.
    max_retries : int
        maximum number of attempts at fetching each Jira ticket.
        Defaults to 5.
    write : bool
        keyword-only argument. Whether or not to write out each
        successfully downloaded Jira issue to a file.

    Returns
    -------
    tuple
        two-element tuple. The first element is the list of
        LangChain documents for successfully retrieved Jira tickets.
        The second element of the returned tuple is a list of Jira
        tickets that were not fetched successfully. Each element of
        this list is a two-element tuple, where the first tuple
        element is the string ticket name (including prefix and dash)
        and the second tuple element is the corresponding string
        error message.
    """
    ticket_list = [
        ticket_prefix + "-" + str(i)
        for i in range(min_ticket_num, max_ticket_num + 1)
    ]
    return jira_tickets_from_list(
        ticket_list, email, api_token, folder, max_retries, write=write
    )


def get_max_issue_number(
    project: str,
    email: str = username,
    api_token: str = api_token,
) -> int:
    """Determine maximum Jira issue number for a project.

    Parameters
    ----------
    project : str
        name of Jira project like "DM" or "SP" (without dash)
    email : str
        email address of Jira account associated with the API token
    api_token : str
        Jira API token

    Returns
    -------
    int
        the maximum extant Jira ticket number for the specified project
    """
    url = f"https://rubinobs.atlassian.net/rest/api/latest/search?jql=project={project}+order+by+key+desc&maxResults=1"
    auth = requests.auth.HTTPBasicAuth(email, api_token)
    headers = {"Content-Type": "application/json"}
    response = get_with_retries(url, auth, headers)

    if response:
        data = response.json()
        issue_name = data["issues"][0]["key"]
        return int(issue_name.split("-")[-1])
    else:
        _log.error(f"Unable to find max issue for {project}, skipping...")
        return 0


def sanitize_comments(doc: Document) -> Document:
    """Remove comments from the metadata and add them to page_content."""
    if "comments" not in doc.metadata:
        return doc

    comments = doc.metadata.pop("comments", {})
    new_content = f"{doc.page_content}\n\nComments:\n{comments}"
    return Document(page_content=new_content, metadata=doc.metadata)


def sanitize_illegal_keys(meta: dict) -> None:
    """Sanitize illegal keys."""
    for bad_key in ("id", "vector"):
        meta.pop(bad_key, None)


def sanitize_parent_issue(meta: dict) -> None:
    """Sanitize parent issues."""
    parent = meta.get("parent_issue")
    if parent == {}:
        meta.pop("parent_issue", None)
    elif isinstance(parent, dict) and "key" in parent:
        meta["parent_issue"] = [parent["key"]]


def sanitize_related_issues(meta: dict) -> None:
    """Sanitize related issues."""
    issues = meta.get("related_issues")
    if not isinstance(issues, list):
        meta.pop("related_issues", None)
        return

    sanitized = []
    for issue in issues:
        if isinstance(issue, dict):
            sanitized.append(
                str(issue.get("key")) if "key" in issue else json.dumps(issue)
            )
        else:
            sanitized.append(str(issue))

    meta["related_issues"] = sanitized


def sanitize_attachments(meta: dict) -> None:
    """Sanitize attatchments."""
    attachments = meta.get("attachments")
    if not isinstance(attachments, list):
        meta.pop("attachments", None)
        return

    sanitized = []
    for att in attachments:
        if isinstance(att, dict):
            sanitized.append(
                str(att.get("filename"))
                if "filename" in att
                else json.dumps(att)
            )
        else:
            sanitized.append(str(att))

    meta["attachments"] = sanitized


def sanitize_metadata(docs: list[Document]) -> list[Document]:
    """Sanitize related_issues, attachments, and parent_issue keys from
    metadata and add source and source_key keys.

    Parameters
    ----------
    docs: list[Document]
        list of LangChain documents from Jira scrape to be cleaned.

    Returns
    -------
    list[Document]
        list of clean LangChain documents.
    """
    out = []

    for doc in docs:
        clean_doc = sanitize_comments(doc)

        raw_meta = getattr(clean_doc, "metadata", {}) or {}

        sanitize_illegal_keys(raw_meta)
        sanitize_parent_issue(raw_meta)
        sanitize_related_issues(raw_meta)
        sanitize_attachments(raw_meta)
        sanitize_dates(raw_meta)

        raw_meta["source_key"] = "jira"
        issue_key = raw_meta.get("key", "")
        raw_meta["source"] = (
            f"https://rubinobs.atlassian.net/browse/{issue_key}"
        )

        out.append(
            Document(page_content=clean_doc.page_content, metadata=raw_meta)
        )

    return out


def process_project(
    project: dict[str, Any],
    completed_keys: set[str],
    log_path: Path,
    output_dir: Path,
    exclude_status: list[str],
) -> None:
    """Scrape a Jira project and write to pickle files.

    Parameters
    ----------
    project: str
        name of Jira project to be scraped
    completed_keys: set[str]
        set of projects that have been scraped and written to pkl files.
    log_path: Path
        path to progress.log file the Jira scraping run.
    output_dir: Path
        path to output directory for the repo.
    exclude_status: list[str]
        list of status types to exclude.

    Returns
    -------
    documents : list
        list of LangChain documents, one per successfully retrieved
        Jira issue. Documents corresponding to Jira issues with
        certain statuses are dropped according to the YAML's
        specficiations.
    """
    project_name = project["name"]

    if project_name in completed_keys:
        _log.info(f"Skipping already processed space: {project_name}")
        return

    _log.info(f"Scraping from {project_name}...")
    start = project.get("start", 1)
    if "end" not in project:
        end = get_max_issue_number(project_name)
    else:
        end = project["end"]
    project_docs, failures = jira_tickets_in_range(project_name, start, end)
    project_docs = [
        d for d in project_docs if d.metadata["status"] not in exclude_status
    ]

    cleaned = sanitize_metadata(project_docs)
    chunked = chunk_docs(cleaned)
    batched = batch_by_tokens(chunked)

    write_batches_to_pickle(batched, project_name, output_dir)

    completed_keys.add(project_name)
    save_progress(log_path, completed_keys)

    del project_docs, cleaned, chunked, batched
    gc.collect()


def scrape_jira(yaml_path: str, output_dir: str) -> None:
    """Scrape Jira into LangChain documents and write docs to pickle files.

    Parameters
    ----------
    yaml_path: str
        String of path to jira_sources.yaml
    output_dir: str
        String of path to output directory, typically a timestamped directory
        specified in run_scraping.
    """
    base_dir = Path(output_dir)
    log_path = base_dir / "progress.log"

    path = Path(yaml_path)
    with path.open(mode="r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    completed_keys = load_progress(log_path)

    exclude_status = data.get("exclude_status", [])

    for project in data["projects"]:
        start = time.time()
        process_project(
            project, completed_keys, log_path, base_dir, exclude_status
        )
        end = time.time()
        _log.info(
            f"Scraped {project['name']} in {(end - start) / 60:.2f} minutes"
        )

    completed_keys.add("done")
    with Path.open(log_path, "w", encoding="utf-8") as f:
        json.dump(list(completed_keys), f, indent=2)
