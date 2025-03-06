"""Utilities for ingesting GitHub repo contents into langchain documents."""

# note that it's necessary to have set the env var
# GITHUB_PERSONAL_ACCESS_TOKEN to the relevant GitHub API access token

# see https://python.langchain.com/docs/integrations/document_loaders/github/

import logging

import requests
from langchain.schema.document import Document
from langchain_community.document_loaders import GithubFileLoader


def load_files_1repo(repo: str = "lsst/daf_butler") -> list:
    """Load all utf-8 files from a given GitHub repo.

    Parameters
    ----------
    repo : str
        name of the GitHub repository including the org and repo name

    Returns
    -------
    docs : list
        list of langchain documents
    """
    loader = GithubFileLoader(
        repo=repo,  # the repo name
        branch="main",  # the branch name
        github_api_url="https://api.github.com",
        file_filter=None,
    )

    docs = []

    for metadata in loader.get_file_paths():
        file_path = metadata["path"]

        try:
            string = loader.get_file_content_by_path(file_path)
            doc = Document(string, metadata=metadata)
            docs.append(doc)
        except Exception:
            logging.exception("Failed to load file.")

    return docs


def load_org(n_repo_max: int = 2, org_name: str = "lsst-dm") -> list:
    """Load all utf-8 files from GitHub repos in a GitHub org.

    Parameters
    ----------
    n_repo_max : int
        number of most recently updated repos for which to ingest
        contents; should be greater than 0 but less than or equal to 100
    org_name : str
        GitHub organization name

    Returns
    -------
    all_docs : list
        list of langchain documents
    """

    # this will give an output 'list' of repos sorted
    # from most to least recently updated
    api_url = (
        "https://api.github.com/orgs/"
        + org_name
        + "/repos?per_page=100&sort=updated"
    )

    result = requests.get(api_url, timeout=10)
    data = result.json()

    all_docs = []
    for i in range(n_repo_max):
        docs = load_files_1repo(repo=data[i]["full_name"])
        all_docs = all_docs + docs

    return all_docs
