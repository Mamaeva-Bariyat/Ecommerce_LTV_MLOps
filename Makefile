.PHONY: install train api frontend docker clean help

help:
	@echo "Available commands:"
	@echo "  make install     - install all dependencies"
	@echo "  make train       - run training pipeline"
	@echo "  make api         - start FastAPI server"
	@echo "  make frontend    - start Streamlit dashboard"
	@echo "  make docker      - build & run with docker-compose"
	@echo "  make clean       - remove caches and temporary files"

install:
	python -m venv .venv
	. .venv/bin/activate && pip install -U pip
	. .venv/bin/activate && pip install -r requirements.txt
	. .venv/bin/activate && pip install -r backend/requirements.txt
	. .venv/bin/activate && pip install -r frontend/requirements.txt

train:
	python -m src.pipeline

api:
	uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	streamlit run frontend/main.py --server.port 8501

docker:
	docker-compose up --build

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
