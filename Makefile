PYTHON_VERSION := $(shell cat .python-version)

.PHONY: help
help:
	@echo "Make targets for rubin_rag:"
	@echo "make clean - Remove generated files"
	@echo "make init - Set up dev environment (create .venv, install deps, pre-commit hooks)"
	@echo "make linkcheck - Check for broken links in documentation"
	@echo "make update - Update pre-commit dependencies and run make init"
	@echo "make update-deps - Update pre-commit dependencies"

.PHONY: clean
clean:
	rm -rf .tox
	rm -rf docs/_build
	rm -rf docs/api

.PHONY: init
init:
	pip install --upgrade uv
	test -d .venv || uv venv .venv --prompt rubin-rag
	uv sync --group dev
	uv run pre-commit install
	uv export --frozen --no-dev --no-hashes --no-emit-project --python $(PYTHON_VERSION) -o requirements.txt
	rm -rf .tox

# This is defined as a Makefile target instead of only a tox command because
# if the command fails we want to cat output.txt, which contains the
# actually useful linkcheck output. tox unfortunately doesn't support this
# level of shell trickery after failed commands.
.PHONY: linkcheck
linkcheck:
	sphinx-build --keep-going -n -W -T -b linkcheck docs	\
	    docs/_build/linkcheck				\
	    || (cat docs/_build/linkcheck/output.txt; exit 1)

# update updates uv, pre-commit, and pre-commit hook versions only.
# Other dependencies are intentionally not upgraded here for safety reasons; to upgrade a
# specific package run uv lock --upgrade-package <name>, or
# run uv lock --upgrade to review and update all pinned dependencies.
.PHONY: update
update:
	pip install --upgrade uv
	uv lock --upgrade-package uv
	uv lock --upgrade-package pre-commit
	uv sync --group dev
	uv run pre-commit autoupdate
	uv export --frozen --no-dev --no-hashes --no-emit-project --python $(PYTHON_VERSION) -o requirements.txt


# .PHONY: update-deps
# update-deps:
# 	pip install --upgrade uv
# 	uv lock --upgrade
# 	uv run pre-commit autoupdate
# 	uv export --frozen --no-dev --no-hashes --no-emit-project -o requirements.txt
