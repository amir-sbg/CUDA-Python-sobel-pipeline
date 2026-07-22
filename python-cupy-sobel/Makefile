.PHONY: install install-test test run clean

install:
	python -m pip install -r requirements.txt
	python -m pip install -e .

install-test:
	python -m pip install -r requirements-test.txt
	python -m pip install -e .

test:
	python -m pytest -q

run:
	python -m gpu_edges.pipeline --cpu-only

clean:
	rm -rf outputs reports .pytest_cache src/python_cupy_sobel.egg-info
