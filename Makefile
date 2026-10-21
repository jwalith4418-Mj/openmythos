.PHONY: install install-dev test lint fmt check docker-build docker-run eval clean

# ─── Dev setup ──────────────────────────────────────────────────────────────
install:
	pip install -e ".[serve]"

install-dev:
	pip install -e ".[all,dev]"
	pre-commit install

# ─── Code quality ───────────────────────────────────────────────────────────
lint:
	ruff check . --fix
	black --check .

fmt:
	ruff check . --fix
	black .

check: lint
	mypy open_mythos/main.py open_mythos/serving/server.py --ignore-missing-imports

# ─── Tests ──────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short --ignore=tests/chaos -x

test-all:
	pytest tests/ -v --tb=short -x

test-chaos:
	pytest tests/chaos/ -v --tb=short

coverage:
	pytest tests/ --cov=open_mythos --cov-report=html --cov-report=term-missing

# ─── Security ───────────────────────────────────────────────────────────────
audit:
	pip-audit --requirement requirements.txt

sbom:
	cyclonedx-py environment -o sbom.json --of json

# ─── Docker ─────────────────────────────────────────────────────────────────
docker-build:
	docker build -t openmythos:local .

docker-run:
	docker-compose up -d

docker-down:
	docker-compose down

# ─── Evaluation ─────────────────────────────────────────────────────────────
eval:
	python scripts/eval_checkpoint.py \
		--checkpoint $(CHECKPOINT) \
		--tasks mmlu arc_challenge gsm8k \
		--output-html eval_report.html

report:
	python scripts/generate_report.py

# ─── Export ─────────────────────────────────────────────────────────────────
export-hf:
	python scripts/export_hf.py --checkpoint $(CHECKPOINT) --output $(OUTPUT)

# ─── K8s ────────────────────────────────────────────────────────────────────
helm-dry-run:
	helm install openmythos helm/openmythos --dry-run --debug

helm-deploy:
	helm upgrade --install openmythos helm/openmythos \
		--set image.tag=$(TAG) \
		--set secrets.apiKeys="$(API_KEYS)"

# ─── Clean ──────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete
	rm -rf dist build *.egg-info htmlcov .coverage
