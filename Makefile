VENV = venv
FLASK_APP = app.py
PYTHON_VERSION = python3.11

install:
	$(PYTHON_VERSION) -m venv $(VENV)
	./$(VENV)/bin/pip install --upgrade pip
	./$(VENV)/bin/pip install -r requirements.txt

run:
	FLASK_APP=$(FLASK_APP) FLASK_ENV=development ./$(VENV)/bin/flask run --port 3000

clean:
	rm -rf $(VENV)

reinstall: clean install