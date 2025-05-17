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

"""Utilities for scraping GitHub repo contents into LangChain documents."""

import logging
import os
import pickle
import subprocess
from pathlib import Path

import requests
import yaml
from langchain_community.document_loaders import TextLoader

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

# note that it's necessary to have set the env var
# GITHUB_PERSONAL_ACCESS_TOKEN to the relevant GitHub API access token
access_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
if access_token is None:
    raise ValueError("Missing GITHUB_PERSONAL_ACCESS_TOKEN")


def repos_in_org(org_name: str = "lsst-dm") -> list[str]:
    """Get list of repos within a GitHub organization.

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
        _log.exception(f"Failed to retrieve list of repos in org {org_name}.")
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


def clean_file_list(directory: str = "rubin_rag") -> list[str]:
    """Make a list of non-hidden files within a directory.

    Parameters
    ----------
    directory : str
        directory for which to make a list of non-hidden files.
        Note that files within all non-hidden subdirectories of
        directory are also returned. Note that we want to ignore
        hidden .git directories, for instance.

    Returns
    -------
    list
        list of strings, where each string is a relative file
        path. Returns empty list in the case of no non-hidden
        files found within the specified directory.
    """
    files = file_list(directory=directory)
    return [
        f for f in files if ((Path(f).name[0] != ".") and (f.find("/.") == -1))
    ]


def file_list(directory: str = "rubin_rag") -> list[str]:
    """Make a list of all files within a directory.

    Parameters
    ----------
    directory : str
        directory for which to make a list of all files, including
        hidden files and files within hidden subdirectories.
        Note that files within all subdirectories of directory
        are also returned.

    Returns
    -------
    list
        list of strings, where each string is a relative file
        path. Returns empty list in the case of no files found
        within the specified directory.
    """
    command = ["find", directory, "-type", "f"]
    try:
        process = subprocess.run(
            command, capture_output=True, text=True, check=True
        )
    except Exception:
        # this can happen if there's an empty GitHub repo e.g., https://github.com/lsst-it/ittn-041
        return []
    output = process.stdout.strip()
    if output:
        return output.split("\n")
    else:
        return []


def clone_repo(repo_name: str = "lsst/daf_butler") -> None:
    """Clone a GitHub repo to the current working directory.

    Parameters
    ----------
    repo_name : str
        repo name including the organization name, for instance
        lsst/daf_butler
    """
    repository_url = "https://github.com/" + repo_name + ".git"
    command = ["git", "clone", repository_url]
    subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )  # Capture output as text


def scrape_repo(repo_name: str = "lsst/daf_butler") -> list:
    """Scrape all non-hidden files in a locally cloned repo.

    Parameters
    ----------
    repo_name : str
        repo name including the organization name, for instance
        lsst/daf_butler. As part of scraping, the entire repo will
        be cloned from GitHub.

    Returns
    -------
    docs : list
        a list of scraped LangChain documents, one per non-hidden file
        in the cloned repo. Empty list if no files found/scraped.
    """
    clone_repo(repo_name=repo_name)
    repo_basename = Path(repo_name).name
    flist = clean_file_list(directory=repo_basename)

    # loop over LangChain docs with try/except
    # delete the cloned repo (BE CAREFUL WITH RM -R)
    # return the list of documents

    docs = []
    for i, f in enumerate(flist):
        _log.info(f"working on file {i}, {f}")
        loader = TextLoader(f)
        try:
            results = loader.load()
            doc = results[0]
            doc.metadata["source_key"] = "github"
            docs.append(doc)
        except Exception:
            _log.warning(f"possible non-text file : {f}")

    _log.info(f"REPO BASENAME: {repo_basename}")
    # delete the git clone !
    if Path(repo_basename).exists():
        command = ["rm", "-rf", repo_basename]
        subprocess.run(command, check=False)

    return docs


def scrape_org(org_name: str = "lsst-dmsst", *, write: bool = False) -> list:
    """Scrape all repos within a GitHub org.

    Parameters
    ----------
    org_name : str
        GitHub organization name including the organization name,
        for instance lsst-dm.

    Returns
    -------
    all_docs : list
        a list of scraped LangChain documents, one per non-hidden file
        spanning all repos within the org. Empty list if no files
        found/scraped.
    """
    repos = repos_in_org(org_name)

    all_docs: list = []
    for i, repo in enumerate(repos):
        _log.info(f"WORKING ON REPO : {repo} {i + 1} of {len(repos)}")
        docs = scrape_repo(repo_name=repo)
        all_docs = all_docs + docs

    if write:
        with Path(org_name + "_20250502.pickle").open("wb") as handle:
            pickle.dump(all_docs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return all_docs

def load_and_scrape(yaml_file : str) -> None:
    """Scrape all GitHub repos within multiple GitHub orgs."""

    path = Path(yaml_file)
    with path.open(mode="r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    orgs = data["organization"]
    for org in orgs:
        docs = scrape_org(org_name=org["name"], write=True)
        del docs
