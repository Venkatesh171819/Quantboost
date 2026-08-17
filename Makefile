.PHONY: setup app run test clean lab

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

app:
	streamlit run app.py

run:
	python scripts/run_pipeline.py --offline --n-trials 0

run-tuned:
	python scripts/run_pipeline.py --offline --n-trials 40

test:
	pytest -q tests

lab:
	jupyter lab notebooks

clean:
	rm -rf reports/*.csv reports/figures/*.png data/raw/*.parquet .pytest_cache
