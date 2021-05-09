# Makefile

all: install

install: venv
	: Install deps 
	. venv/bin/activate && pip install -r requirements.txt

venv:
	: Create virtualenv
	test -d venv || python3 -m venv venv

clean:
	rm -rf venv
	find -iname "*.pyc" -delete