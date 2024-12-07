# Define variables
DOCKER_COMPOSE = docker compose
PYTHON = .venv/bin/python
ACTIVATE_VENV = source .venv/bin/activate

# Targets
.PHONY: all up migrate populate embeddings

all: up migrate populate embeddings


up:
	$(DOCKER_COMPOSE) up -d
migrate:
	$(ACTIVATE_VENV) && $(PYTHON) -m alembic upgrade head

populate:
	$(PYTHON) src/database/populate_tables.py
embeddings:
	$(PYTHON) src/models/embeddings.py