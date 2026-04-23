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
import random
import shutil
import subprocess
import time
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
    sanitize_dates,
    save_progress,
    write_batches_to_pickle,
    write_raw_to_pickle,
)

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

# note that it's necessary to have set the env var
# GITHUB_PERSONAL_ACCESS_TOKEN to the relevant GitHub API access token
access_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")


def repos_in_org(org_name: str = "lsst-dm") -> list[tuple[str, str]]:
    """Get list of repos within a GitHub organization.

    Parameters
    ----------
    org_name : str
        GitHub organization name

    Returns
    -------
    list
        list of tuples of strings, where the first is a repo in the format
        org_name/repo_name and the second is its default branch. Returns
        empty list in the case of a failed GitHub API response.
    """
    url = f"https://api.github.com/orgs/{org_name}/repos?simple=yes&per_page=100&page=1"
    try:
        res = requests.get(
            url, headers={"Authorization": access_token}, timeout=10
        )
        res.raise_for_status()
    except Exception:
        _log.exception(
            f"Request to Github failed for {org_name}. Check that "
            "you haven't exceeded API rate limits."
        )
        return []

    repos = res.json()

    if repos and isinstance(repos[0], str):
        _log.error(f"Unexpected response format: {repos[:2]}")
        return []

    while "next" in res.links:
        try:
            res = requests.get(
                res.links["next"]["url"],
                headers={"Authorization": access_token},
                timeout=10,
            )
            res.raise_for_status()
            page_repos = res.json()
            if page_repos and isinstance(page_repos[0], str):
                _log.error(
                    f"Unexpected response format in pagination: "
                    f"{page_repos[:2]}"
                )
                break
            repos.extend(page_repos)
        except Exception:
            _log.exception("Failed to retrieve paginated repos")
            break

    return [
        (str(repo.get("full_name")), str(repo.get("default_branch", "main")))
        for repo in repos
        if isinstance(repo, dict)  # Add safety check
        and ("data" not in repo["name"].lower())
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
    excluded_exts = {
        ".fits",
        ".eps",
        ".tar",
        ".zip",
        ".out",
        ".pkl",
        ".dax",
        ".svg",
        ".pd",
        ".trim",
        ".SIMLIB",
        ".pickle",
        ".lvproj",
        ".lvbitx",
        ".tsbuildinfo",
    }

    excluded_dirs = {"images", "figures", "logs"}

    return [
        f
        for f in files
        if (
            not Path(f).name.startswith(".")
            and "/." not in f
            and not any(
                f.lower().endswith(ext.lower()) for ext in excluded_exts
            )
            and "gen2" not in f.lower()
            and "data" not in os.path.split(f)[0].lower()
            and not any(part in excluded_dirs for part in f.split("/"))
        )
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


def prepare_path_for_link(path_str: str, branch: str) -> str:
    """Inject /blob/{branch}/ into the file path to allow source key
    to point to direct link.

    Parameters
    ----------
    path_str: str
        string of path to the file
    branch: str
        the default branch of the repo for link (i.e. master, main)

    Returns
    -------
    str
        the URL to the exact file in GitHub
    """
    parts = path_str.split(
        "/", 2
    )  # maxsplit=2 ensures we preserve the full path
    if len(parts) < 2:
        raise ValueError("Expected input like 'repo/path/to/file'")
    repo = parts[0]
    rest = parts[1] if len(parts) == 2 else parts[1] + "/" + parts[2]
    return f"{repo}/blob/{branch}/{rest}"


def scrape_repo(
    repo_name: str,
    default_branch: str,
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
    default_branch: str
        Used for creating link, typically main or master.
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
        converted_path = prepare_path_for_link(f, default_branch)
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

            sanitize_dates(doc.metadata)

            if not is_data_dump(doc) and len(doc.page_content) >= 50:
                documents.append(doc)

        except Exception as e:
            _log.debug(f"possible non-text file : {f} - Error: {e!s}")

    write_raw_to_pickle(documents, repo_basename, org_output_dir)
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
        (r, branch)
        for r, branch in repos_in_org(org_name)
        if r.split("/")[1] not in repos_ignore
    ]

    for i, (repo, branch) in enumerate(repos):
        _log.info(f"WORKING ON REPO : {repo} {i + 1} of {len(repos)}")
        scrape_repo(
            repo_name=repo,
            default_branch=branch,
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


def scrape_github(
    yaml_path: str, output_dir: str, n: int | None = None
) -> None:
    """Scrape Github based on settings in yaml file and write pickle files to
    output directory.

    Parameters
    ----------
    yaml_path: str
        String of path to github_sources.yaml
    output_dir: str
        String of path to output directory, typically a timestamped directory
        specified in run_scraping.
    n: int | None
        If provided, randomly sample n repos from the combined pool of all
        orgs instead of scraping all repos. Defaults to None (scrape all).
    """
    if access_token is None:
        raise ValueError("Missing GITHUB_PERSONAL_ACCESS_TOKEN")
    base_dir = Path(output_dir)
    log_path = base_dir / "progress.log"

    completed_keys = load_progress(log_path)
    spec = load_yaml_spec(yaml_path)

    # Collect all eligible repos from all orgs into a combined pool,
    # applying ignore_repos filtering per org after repos_in_org() filters.
    all_repos: list[tuple[str, str]] = []
    orgs = spec["organization"]
    for org in orgs:
        repos_ignore = org.get("ignore_repos", [])
        org_repos = [
            (r, branch)
            for r, branch in repos_in_org(org["name"])
            if r.split("/")[1] not in repos_ignore
        ]
        all_repos.extend(org_repos)

    if n is not None:
        all_repos = random.sample(all_repos, min(n, len(all_repos)))

    for i, (repo, branch) in enumerate(all_repos):
        _log.info(f"WORKING ON REPO : {repo} {i + 1} of {len(all_repos)}")
        scrape_repo(
            repo_name=repo,
            default_branch=branch,
            completed_keys=completed_keys,
            log_path=log_path,
            output_dir=base_dir,
        )

    completed_keys.add("done")
    with Path.open(log_path, "w", encoding="utf-8") as f:
        json.dump(list(completed_keys), f, indent=2)
