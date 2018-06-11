PKGNAME=peeemtee

default: build

build:
	@echo "No need to build :)"

install:
	pip install .

install-dev:
	pip install -e .

clean:
	python setup.py clean --all

test:
	py.test --junitxml=./junit.xml peeemtee || true

test-cov:
	py.test --cov ./ --cov-report term-missing --cov-report xml peeemtee || true

test-loop:
	py.test || true
	ptw --ext=.py,.pyx

flake8:
	py.test --flake8 || true

pep8: flake8

docstyle:
	py.test --docstyle  || true

lint:
	py.test --pylint || true

dependencies:
	pip install -Ur requirements.txt


.PHONY: clean build install test test-nocov flake8 pep8 dependencies dev-dependencies
