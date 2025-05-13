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

"""Utilities for ingesting GitHub repo contents into LangChain documents."""

import os
import pickle
import subprocess
import time

from langchain_community.document_loaders import TextLoader
from scrape_github import repos_in_org


def clean_file_list(directory="rubin_rag") -> list[str]:
    """Make a list of non-hidden files within a directory."""
    files = file_list(directory=directory)
    files = [
        f
        for f in files
        if ((os.path.basename(f)[0] != ".") and (f.find("/.") == -1))
    ]
    return files


def file_list(directory="rubin_rag"):
    """Make a list of all files within a directory."""
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


def clone_repo(repo_name="lsst/daf_butler"):
    """Clone a GitHub repo to the current working directory."""
    repository_url = "https://github.com/" + repo_name + ".git"
    command = ["git", "clone", repository_url]
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )  # Capture output as text


def ingest_1repo(repo_name="lsst/daf_butler"):
    """Ingest all non-hidden files in a locally cloned repo."""
    clone_repo(repo_name=repo_name)
    repo_basename = os.path.basename(repo_name)
    flist = clean_file_list(directory=repo_basename)

    # loop over LangChain docs with try/except
    # delete the cloned repo (BE CAREFUL WITH RM -R)
    # return the list of documents

    docs = []
    for i, f in enumerate(flist):
        print(i, f)
        loader = TextLoader(f)
        try:
            doc = loader.load()
            doc = doc[0]
            doc.metadata["source_key"] = "github"
            docs.append(doc)
        except Exception:
            print("possible non-text file : " + f)

    print("REPO BASENAME: " + repo_basename)
    # delete the git clone !
    if os.path.exists(repo_basename):
        command = ["rm", "-rf", repo_basename]
        process = subprocess.run(command, check=False)

    return docs


def scrape_1org(org_name="lsst-dmsst", write=False):
    """Ingest all repos within a GitHub org."""
    repos = repos_in_org(org_name)

    all_docs = []
    t0 = time.time()
    for i, repo in enumerate(repos):
        print("WORKING ON REPO : " + repo, i + 1, " of ", len(repos))
        docs = ingest_1repo(repo_name=repo)
        all_docs = all_docs + docs

    dt = time.time() - t0
    print(
        "took "
        + f"{dt:.1f}"
        + " seconds to scrape "
        + org_name
        + " ; "
        + str(len(all_docs))
        + " files successfully scraped"
    )

    if write:
        with open(org_name + "_20250502.pickle", "wb") as handle:
            pickle.dump(all_docs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return all_docs


def scrape_many_orgs():
    """Ingest all GitHub repos within multiple GitHub orgs."""
    orgs = [
        "lsst",
        "lsst-it",
        "lsst-dmsst",
        "lsst-pst",
        "lsst-sqre",
        "lsst-sqre-testing",
        "lsst-sitcom",
        "lsst-ts",
        "lsst-camera-dh",
        "rubin-observatory",
        "rubin-dp0",
        "lsst-sims",
        "lsst-epo",
        "lsst-camera-dh",
        "LSSTDESC",
        "LSST-strong-lensing",
        "LSST-TVSSC",
        "LSST-SSSC",
        "LSSTScienceCollaborations",
        "lsst-dm",
    ]

    all_docs = []
    for org in orgs:
        docs = scrape_1org(org_name=org, write=True)
        del docs

    return all_docs
