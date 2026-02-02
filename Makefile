.PHONY: test benchmark clean

test:
	python -m pytest tests/ -v

benchmark:
	python -m app.run_benchmark --alpha 0.0 0.5 1.0

clean:
	rm -rf outputs/*.csv outputs/*.md __pycache__ */__pycache__ */*/__pycache__
