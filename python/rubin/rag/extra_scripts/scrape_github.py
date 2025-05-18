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

import gc
import logging
import os
import pickle
import subprocess
import time
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

    return [
        repo["full_name"]
        for repo in repos
        if ("data" not in repo["name"].lower())
        and ("dustmaps" not in repo["name"].lower())
        and ("gen2" not in repo["name"].lower())
    ]


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
        f
        for f in files
        if ((Path(f).name[0] != ".") and (f.find("/.") == -1))
        and (f[-5:] != ".fits")
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


def scrape_repo(
    repo_name: str = "lsst/daf_butler", max_mb: int = 1024
) -> None:
    """
    Scrape all non-hidden files in a locally cloned repo and batch
    them into pickle files.

    Parameters
    ----------
    repo_name : str
        GitHub repository in the format 'org/repo'.
    max_mb : int
        Maximum size of each pickle file in megabytes.
    """
    # At start of scrape_repo

    repo_org, repo_basename = repo_name.split("/", 1)
    output_dir = Path(f"batched_github_output/{repo_org}/{repo_basename}")
    if any(output_dir.glob(f"{repo_basename}_*.pkl")):
        _log.info(f"Skipping {repo_name}, already has pickle files.")
        return

    # Extract repo organization and name
    if "/" not in repo_name:
        raise ValueError("Repository name should be in format 'org/repo'")

    # Clone the repository
    clone_repo(repo_name=repo_name)

    # Get list of files
    flist = clean_file_list(directory=repo_basename)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process files
    docs = []
    current_batch: list = []
    current_batch_size = 0
    batch_number = 1
    batch_size_limit = max_mb * 1024 * 1024  # Convert MB to bytes

    for i, f in enumerate(flist):
        _log.debug(f"working on file {i}, {f}")
        loader = TextLoader(f, encoding="utf-8")

        try:
            results = loader.load()
            doc = results[0]
            doc.metadata["source_key"] = "github"

            # Get approximate size of this document
            doc_size = len(pickle.dumps(doc))

            # If adding this document would exceed the batch size limit,
            # save the current batch
            if (
                current_batch_size + doc_size > batch_size_limit
                and current_batch
            ):
                batch_path = output_dir / f"{repo_basename}_{batch_number}.pkl"
                with Path(batch_path).open("wb") as f_out:
                    pickle.dump(current_batch, f_out)
                _log.info(f"Saved batch {batch_number} to {batch_path}")

                # Reset batch
                current_batch = []
                current_batch_size = 0
                batch_number += 1

            # Add document to current batch and overall docs list
            current_batch.append(doc)
            current_batch_size += doc_size
            docs.append(doc)

        except Exception as e:
            _log.debug(f"possible non-text file : {f} - Error: {e!s}")

    # Save any remaining documents in the last batch
    if current_batch:
        batch_path = output_dir / f"{repo_basename}_{batch_number}.pkl"
        with Path(batch_path).open("wb") as f_out:
            pickle.dump(current_batch, f_out)
        _log.info(f"Saved batch {batch_number} to {batch_path}")

    _log.debug(f"REPO BASENAME: {repo_basename}")

    # Delete the git clone
    clone_path = Path(repo_basename)
    if clone_path.exists() and (Path.cwd() in clone_path.resolve().parents):
        command = ["rm", "-rf", repo_basename]
        subprocess.run(command, check=False)

    # Log the completion message
    _log.info(f"Saved {repo_basename} to pickle in {batch_number} batches.")

    # Free up memory
    del docs
    gc.collect()


def scrape_org(org_name: str = "lsst-dmsst", max_mb: int = 1024) -> None:
    """Scrape all repos within a GitHub org.

    Parameters
    ----------
    org_name : str
        GitHub organization name including the organization name,
        for instance lsst-dm.
    max_mb : int
        Maximum size of each pickle file in megabytes.
    """
    start_org = time.time()

    repos = repos_in_org(org_name)

    for i, repo in enumerate(repos):
        _log.info(f"WORKING ON REPO : {repo} {i + 1} of {len(repos)}")
        scrape_repo(repo_name=repo, max_mb=max_mb)

    end_org = time.time()
    _log.info(
        f"Scraped {org_name} in {(end_org - start_org) / 60:.2f} minutes."
    )


def load_yaml_spec(yaml_file: str) -> dict:
    """Load YAML file specifying GitHub sources to scrape.

    Parameters
    ----------
    yaml_file : str
        file name of the YAML file specifying GitHub orgs to scrape
    """
    path = Path(yaml_file)
    with path.open(mode="r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_all_repos(yaml_file: str) -> list[str]:
    """Get list of repos across many orgs (exploratory utility).

    Parameters
    ----------
    yaml_file : str
        file name of the YAML file specifying GitHub orgs to scrape

    Returns
    -------
    all_repos : list[str]
        list of repositories found across all orgs in 'org/repo' format.
    """
    spec = load_yaml_spec(yaml_file)

    orgs = spec["organization"]
    all_repos: list = []
    for org in orgs:
        org_name = org["name"]
        _log.info(f"retrieving repo list for org {org_name}")
        repos = repos_in_org(org_name)
        all_repos += repos

    return all_repos


def load_and_scrape(yaml_file: str) -> None:
    """Scrape all GitHub repos within multiple GitHub orgs.

    Parameters
    ----------
    yaml_file : str
        file name of the YAML file specifying GitHub orgs to scrape
    """
    spec = load_yaml_spec(yaml_file)

    orgs = spec["organization"]
    for org in orgs:
        scrape_org(org_name=org["name"])
