.PHONY: setup verify doctor checks lint typecheck test smoke docker-up docker-down

setup:
	bash scripts/setup_env.sh
verify:
	python scripts/verify_install.py
doctor:
	python scripts/doctor.py
smoke:
	python scripts/smoke_local_pipeline.py
checks:
	bash scripts/run_all_checks.sh --full
lint:
	ruff check src/ tests/ scripts/
typecheck:
	mypy src/
test:
	NO_NETWORK=1 pytest tests/ -q --tb=short
docker-up:
	docker compose up -d
docker-down:
	docker compose down
