#!/bin/bash
ruff --config pyproject.toml check --select I --fix
ruff --config pyproject.toml format .
