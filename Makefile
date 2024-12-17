.PHONY: setup clean run pull-data

CONDA_ENV_NAME=spamkiller
PYTHON_VERSION=3.9

setup:
	@echo "Creating conda environment..."
	conda create -n $(CONDA_ENV_NAME) python=$(PYTHON_VERSION) --yes
	@echo "Installing requirements..."
	conda run -n $(CONDA_ENV_NAME) pip install -r requirements.txt
	@echo "Installing DVC..."
	conda run -n $(CONDA_ENV_NAME) pip install dvc dvc-gdrive

pull-data:
	@echo "Pulling data from DVC..."
	conda run -n $(CONDA_ENV_NAME) dvc pull

run: pull-data
	PYTHONPATH=$(PWD) conda run -n $(CONDA_ENV_NAME) python src/app/bot.py

clean:
	conda env remove -n $(CONDA_ENV_NAME) -y

# Помощь по командам
help:
	@echo "Available commands:"
	@echo "  make setup     : Setup project (create conda env and install requirements)"
	@echo "  make pull-data : Pull data files from DVC storage"
	@echo "  make run      : Run the SpamKiller bot"
	@echo "  make clean    : Remove conda environment" 