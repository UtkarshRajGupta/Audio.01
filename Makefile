.PHONY: dry-run plan test

dry-run:
	./run_demo.sh --dry-run

plan:
	./run_demo.sh --plan-only

test:
	./habitat-env/bin/python -m unittest discover -s tests
