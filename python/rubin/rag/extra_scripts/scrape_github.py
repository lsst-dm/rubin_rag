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

"""Utilities for ingesting GitHub repo contents into langchain documents."""

# see https://python.langchain.com/docs/integrations/document_loaders/github/

import logging
import os

import requests
from langchain.schema.document import Document
from langchain_community.document_loaders import GithubFileLoader

# note that it's necessary to have set the env var
# GITHUB_PERSONAL_ACCESS_TOKEN to the relevant GitHub API access token
access_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
if access_token is None:
    raise ValueError("Missing GITHUB_PERSONAL_ACCESS_TOKEN")


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
        repo=repo,
        branch="main",
        github_api_url="https://api.github.com",
        access_token=str(access_token),
        file_filter=None,
    )

    docs: list = []

    for metadata in loader.get_file_paths():
        file_path = metadata["path"]

        try:
            string = loader.get_file_content_by_path(file_path)
            doc = Document(string, metadata=metadata)
            docs.append(doc)
        except Exception:
            logging.exception("Failed to load file.")

    return docs


def repos_in_org(org_name: str = "lsst-dm") -> list:
    """Get list of repos with a GitHub organization.

    Parameters
    ----------
    org_name : str
        GitHub organization name

    Returns
    -------
    list
        list of strings, where each string is a repo in the format
        org_name/repo_name. Returns empty list in the case of a
        failed GitHub API response.
    """
    url = f"https://api.github.com/orgs/{org_name}/repos?simple=yes&per_page=100&page=1"

    try:
        res = requests.get(
            url, headers={"Authorization": access_token}, timeout=10
        )
    except Exception:
        logging.exception(
            f"Failed to retrieve list of repos in org {org_name}."
        )
        return []

    repos = res.json()

    while "next" in res.links:
        res = requests.get(
            res.links["next"]["url"],
            headers={"Authorization": access_token},
            timeout=10,
        )
        repos.extend(res.json())

    return [repo["full_name"] for repo in repos]


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
    all_docs: list = []

    repos = repos_in_org(org_name: str = "lsst-dm")

    for i, repo in enumerate(repos):
        if i >= n_repo_max:
            break
        docs = load_files_1repo(repo)
        all_docs = all_docs + docs

    return all_docs
