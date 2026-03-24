.PHONY: dry-run plan test host-check

dry-run:
	./run_demo.sh --dry-run

plan:
	./run_demo.sh --plan-only

test:
	./habitat-env/bin/python -m unittest discover -s tests

host-check:
	python3 soundspaces_host_check.py
