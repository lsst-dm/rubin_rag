import subprocess
import os
from langchain_community.document_loaders import TextLoader
from scrape_github import repos_in_org
import time
import pickle

def clean_file_list(directory='rubin_rag'):
    files = file_list(directory=directory)
    files = [f for f in files if ((os.path.basename(f)[0] != '.') and (f.find('/.') == -1))]
    return files

def file_list(directory='rubin_rag'):
    command = ['find', directory, '-type', 'f']
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True)
    except Exception:
        # this can happen if there's an empty GitHub repo e.g., https://github.com/lsst-it/ittn-041
        return []
    output = process.stdout.strip()
    if output:
        return output.split('\n')
    else:
        return []

def clone_repo(repo_name='lsst/daf_butler'):
    
    repository_url = 'https://github.com/' + repo_name + '.git'
    command = ['git', 'clone', repository_url]
    process = subprocess.run(command,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             text=True)  # Capture output as text

def ingest_1repo(repo_name='lsst/daf_butler'):
    clone_repo(repo_name=repo_name)
    repo_basename = os.path.basename(repo_name)
    flist = clean_file_list(directory=repo_basename)

    # loop over LangChain docs with try/except
    # delete the cloned repo (BE CAREFUL WITH RM -R)
    # return the list of documents

    docs = []    
    for i,f in enumerate(flist):
        print(i, f)   
        loader = TextLoader(f)
        try:
            doc = loader.load()
            doc = doc[0]
            doc.metadata["source_key"] = "github"
            docs.append(doc)
        except Exception:
            print('possible non-text file : ' + f)

    print('REPO BASENAME: ' + repo_basename)
    # delete the git clone !
    if os.path.exists(repo_basename):
        command = ['rm', '-rf', repo_basename]
        process = subprocess.run(command)

    return docs


def scrape_1org(org_name='lsst-dmsst', write=False):
    repos = repos_in_org(org_name)
    
    all_docs = []
    t0 = time.time()
    for i, repo in enumerate(repos):
        print('WORKING ON REPO : ' + repo, i+1, ' of ', len(repos))
        docs = ingest_1repo(repo_name=repo)
        all_docs = all_docs + docs
    
    dt = time.time()-t0
    print('took ' + "{:.1f}".format(dt) + ' seconds to scrape ' + org_name + ' ; ' + str(len(all_docs)) + ' files successfully scraped')

    if write:
        with open(org_name + '_20250502.pickle', 'wb') as handle:
            pickle.dump(all_docs, handle, protocol=pickle.HIGHEST_PROTOCOL)        
        
    return all_docs

def scrape_many_orgs():

    orgs = ["lsst", "lsst-it", "lsst-dmsst", "lsst-pst", "lsst-sqre", "lsst-sqre-testing", "lsst-sitcom", "lsst-ts",
            "lsst-camera-dh", "rubin-observatory", "rubin-dp0", "lsst-sims", "lsst-epo", "lsst-camera-dh", "LSSTDESC",
            "LSST-strong-lensing", "LSST-TVSSC", "LSST-SSSC", "LSSTScienceCollaborations", "lsst-dm"]

    all_docs = []
    for org in orgs:
        docs = scrape_1org(org_name=org, write=True)
        #all_docs += docs # memory concern !!!
        del docs

    return all_docs
