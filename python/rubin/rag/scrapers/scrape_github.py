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

"""Scraping code for writing GitHub repo contents into LangChain documents."""

import gc
import json
import logging
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import requests
import yaml
from langchain_community.document_loaders import (
    BSHTMLLoader,
    NotebookLoader,
    PyMuPDFLoader,
    TextLoader,
)
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

# note that it's necessary to have set the env var
# GITHUB_PERSONAL_ACCESS_TOKEN to the relevant GitHub API access token
access_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
if access_token is None:
    raise ValueError("Missing GITHUB_PERSONAL_ACCESS_TOKEN")


def is_rfc3339(date_str: str) -> bool:
    """Parse date and return True if in RFC3339 format."""
    try:
        if date_str.endswith("Z"):
            date_str = date_str[:-1] + "+00:00"
        datetime.fromisoformat(date_str)
    except ValueError:
        return False
    else:
        return True


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
        and ("legacy" not in repo["name"].lower())
        and (int(repo["updated_at"][0:4]) > 2020)
        and (not repo["archived"])
    ]


def is_data_dump(doc: Document) -> bool:
    """Determine if a file has a high chance of being a data dump.

    Parameters
    ----------
    doc : langchain_core.documents.base.Document
        LangChain document. Must have a "source" key in its metadata.
    """
    size_mb = len(doc.page_content) / (1024.0**2)

    exten = Path(doc.metadata["source"]).suffix.lower()

    return (size_mb > 1) and (
        exten
        in [
            ".json",
            ".csv",
            ".txt",
            ".text",
            ".dat",
            ".log",
            ".sql",
            ".yaml",
            ".cfg",
            ".tbl",
        ]
    )


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
        and (not f.endswith("~"))
        and (f[-5:] != ".fits")
        and (f[-4:] != ".eps")
        and (f[-4:] != ".tar")
        and (f[-4:] != ".zip")
        and (f[-4:] != ".out")
        and (f[-4:] != ".pkl")
        and (f[-4:] != ".dax")
        and (f[-4:] != ".svg")
        and (f[-3:] != ".pd")
        and (f[-5:] != ".trim")
        and (f[-7:] != ".SIMLIB")
        and (f[-7:] != ".pickle")
        and (f[-7:] != ".lvproj")
        and (f[-7:] != ".lvbitx")
        and (f[-12:] != ".tsbuildinfo")
        and ("gen2" not in f.lower())
        and ("data" not in os.path.split(f)[0].lower())
        and ("images" not in f.split("/"))
        and ("figures" not in f.split("/"))
        and ("logs" not in f.split("/"))
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
    command = [
        "git",
        "clone",
        "--single-branch",
        "--depth",
        "1",
        repository_url,
    ]
    subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )  # Capture output as text


def delete_clone(repo_basename: str) -> None:
    """Delete a GitHub repo clone from local disk.

    Parameters
    ----------
    repo_basename: str
        repo base name e.g., daf_butler (does not include the org name)
    """
    clone_path = Path(repo_basename)
    try:
        shutil.rmtree(clone_path)
    except Exception as e:
        _log.warning(f"Failed to remove {repo_basename}: {e}")


def select_doc_loader(
    fname: str,
) -> BSHTMLLoader | NotebookLoader | TextLoader | PyMuPDFLoader:
    """Select which LangChain document loader to use for a file.

    Parameters
    ----------
    fname : str
        path of file name that will be loaded into a LangChain doc.
    """
    suffix = Path(fname).suffix

    if suffix == ".ipynb":
        return NotebookLoader(fname, remove_newline=True)
    elif suffix == ".html":
        return BSHTMLLoader(fname)
    elif suffix == ".pdf":
        return PyMuPDFLoader(fname)
    else:
        return TextLoader(fname, encoding="utf-8")


def prepare_path_for_link(path_str: str) -> str:
    """Inject /blob/main/ into the file path to allow source key to point to
    direct link.
    """
    parts = path_str.split(
        "/", 2
    )  # maxsplit=2 ensures we preserve the full path
    if len(parts) < 2:
        raise ValueError("Expected input like 'repo/path/to/file'")
    repo = parts[0]
    rest = parts[1] if len(parts) == 2 else parts[1] + "/" + parts[2]
    return f"{repo}/blob/main/{rest}"


def scrape_repo(
    repo_name: str,
    completed_keys: set[str],
    log_path: Path,
    output_dir: Path,
) -> None:
    """
    Scrape all non-hidden files in a locally cloned repo and batch
    them into pickle files.

    Parameters
    ----------
    repo_name : str
        GitHub repository in the format 'org/repo'.
    completed_keys: set[str]
        set of repos that have been scraped and written to pkl files.
    log_path: path
        path to progress.log file the Github scraping run.
    output_dir: path
        path to output directory for the repo.
    """
    if repo_name in completed_keys:
        _log.info(f"Skipping already processed space: {repo_name}")
        return

    repo_org, repo_basename = repo_name.split("/", 1)

    # Extract repo organization and name
    if "/" not in repo_name:
        raise ValueError("Repository name should be in format 'org/repo'")

    # Clone the repository
    clone_repo(repo_name=repo_name)

    # Get list of files
    flist = clean_file_list(directory=repo_basename)

    org_output_dir = output_dir / repo_org
    org_output_dir.mkdir(parents=True, exist_ok=True)

    # Process files
    documents = []

    for i, f in enumerate(flist):
        converted_path = prepare_path_for_link(f)
        _log.debug(
            f"working on file {i}: "
            f"https://github.com/{repo_org}/{converted_path}"
        )
        loader = select_doc_loader(f)

        try:
            results = loader.load()
            doc = results[0]
            doc.metadata["source_key"] = "github"
            doc.metadata["repo_basename"] = repo_basename
            doc.metadata["org_name"] = repo_org
            doc.metadata["repo"] = repo_name
            doc.metadata["source"] = (
                f"https://github.com/{repo_org}/{converted_path}"
            )

            creationdate = doc.metadata.get("creationdate")
            if creationdate is not None and is_rfc3339(creationdate):
                doc.metadata["creation_date"] = creationdate
            doc.metadata.pop("creationdate", None)

            moddate = doc.metadata.get("moddate")
            if moddate is not None and is_rfc3339(moddate):
                doc.metadata["mod_date"] = moddate
            doc.metadata.pop("moddate", None)

            if not is_data_dump(doc):
                documents.append(doc)

        except Exception as e:
            _log.debug(f"possible non-text file : {f} - Error: {e!s}")

    chunked = chunk_docs(documents)
    batched = batch_by_tokens(chunked)
    write_batches_to_pickle(batched, repo_basename, org_output_dir)

    completed_keys.add(repo_name)
    save_progress(log_path, completed_keys)

    del documents, chunked, batched
    gc.collect()

    # Delete the git clone
    delete_clone(repo_basename)


def scrape_org(
    org_name: str,
    completed_keys: set[str],
    log_path: Path,
    output_dir: Path,
    repos_ignore: list | None = None,
) -> None:
    """Scrape all repos within a GitHub org.

    Parameters
    ----------
    org_name : str
        GitHub organization
    completed_keys: set[str]
        set of repos that have been scraped and written to pkl files.
    log_path: Path
        path to progress.log file the Github scraping run.
    output_dir: Path
        path to output directory for the repo.
    repos_ignore: list
        List of repos to skip, specified in yaml file.
    """
    start_org = time.time()

    if repos_ignore is None:
        repos_ignore = []

    repos = [
        r
        for r in repos_in_org(org_name)
        if r.split("/")[1] not in repos_ignore
    ]

    for i, repo in enumerate(repos):
        _log.info(f"WORKING ON REPO : {repo} {i + 1} of {len(repos)}")
        scrape_repo(
            repo_name=repo,
            completed_keys=completed_keys,
            log_path=log_path,
            output_dir=output_dir,
        )

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


def scrape_github(yaml_path: str, output_dir: str) -> None:
    """Scrape Github based on settings in yaml file and write pickle files to
    output directory.

    Parameters
    ----------
    yaml_path: str
        String of path to github_sources.yaml
    output_dir: str
        String of path to output directory, typically a timestamped directory
        specified in run_scraping.
    """
    base_dir = Path(output_dir)
    log_path = base_dir / "progress.log"

    completed_keys = load_progress(log_path)
    spec = load_yaml_spec(yaml_path)

    orgs = spec["organization"]
    for org in orgs:
        repos_ignore = org.get("ignore_repos", [])
        scrape_org(
            org_name=org["name"],
            completed_keys=completed_keys,
            log_path=log_path,
            output_dir=base_dir,
            repos_ignore=repos_ignore,
        )

    completed_keys.add("done")
    with Path.open(log_path, "w", encoding="utf-8") as f:
        json.dump(list(completed_keys), f, indent=2)
