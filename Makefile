.PHONY: help setup lint test requirements data base documentos index avaliacao notebooks api clean

PY = uv run python

help:
	@echo "Setup e qualidade:"
	@echo "  setup         instala as dependencias com uv (torch na versao CPU)"
	@echo "  lint          roda o ruff"
	@echo "  test          roda a suite de testes"
	@echo "  requirements  exporta requirements.txt a partir do uv.lock"
	@echo ""
	@echo "Base de conhecimento (na ordem):"
	@echo "  data          baixa a base da Olist do Kaggle para data/raw e grava o manifesto"
	@echo "  base          monta a base de avaliacoes com os metadados do pedido"
	@echo "  documentos    limpa e normaliza os comentarios e gera os documentos do indice"
	@echo "  index         constroi o indice lexical (BM25), o vetorial (FAISS) e o mapa de temas"
	@echo ""
	@echo "Avaliacao (chama o LLM; exige as variaveis do .env):"
	@echo "  avaliacao     pool, julgamento, auditoria, gate e avaliacao da geracao"
	@echo ""
	@echo "Uso:"
	@echo "  api           sobe a API em http://127.0.0.1:8000 (documentacao em /docs)"
	@echo ""
	@echo "Notebooks:"
	@echo "  notebooks     executa os notebooks de ponta a ponta, na ordem"
	@echo "  clean         limpa caches"

setup:
	uv sync

lint:
	uv run ruff check .

test:
	uv run pytest -q

requirements:
	uv export --no-dev --no-hashes --no-emit-project -o requirements.txt
	printf '%s\n' '' '# torch na versao CPU vem do indice do PyTorch' \
		'--extra-index-url https://download.pytorch.org/whl/cpu' >> requirements.txt

data:
	$(PY) -m src.data.download

base:
	$(PY) -m src.data.base

documentos:
	$(PY) -m src.preprocessing.documentos

index:
	$(PY) -m src.indexing.lexical
	$(PY) -m src.indexing.vetorial
	$(PY) -m src.indexing.temas

avaliacao:
	$(PY) -m src.evaluation.julgamento pool
	$(PY) -m src.evaluation.julgamento julgar
	$(PY) -m src.evaluation.julgamento auditoria
	$(PY) -m src.graph.gate coletar
	$(PY) -m src.graph.gate calibrar
	$(PY) -m src.evaluation.geracao responder
	$(PY) -m src.evaluation.geracao julgar
	$(PY) -m src.evaluation.geracao injecao

notebooks:
	export HF_HUB_VERBOSITY=error TQDM_DISABLE=1; \
	for nb in notebooks/0*.ipynb; do \
		uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 $$nb || exit 1; \
	done

api:
	uv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
