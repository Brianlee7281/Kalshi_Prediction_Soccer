.PHONY: test lint up down

test:
	python -m pytest tests/ -v

lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

up:
	docker compose up -d

down:
	docker compose down
