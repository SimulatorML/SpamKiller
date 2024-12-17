#!/bin/bash
export PYTHONPATH=$(pwd)/src
poetry run python -m src.app.bot
