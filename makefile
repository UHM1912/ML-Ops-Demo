python = venv\Scripts\python.exe
pip = venv\Scripts\pip.exe

setup:
	python -m venv venv
	$(python) -m pip install --upgrade pip
	$(pip) install -r backend/requirements.txt

run:
	cd backend && ..\$(python) main.py

mlflow:
	cd backend && ..\venv\Scripts\mlflow ui

test:
	cd backend && ..\$(python) -m pytest

clean:
	@if exist backend\steps\__pycache__ (rmdir /s /q backend\steps\__pycache__)
	@if exist backend\__pycache__ (rmdir /s /q backend\__pycache__)
	@if exist .pytest_cache (rmdir /s /q .pytest_cache)
	@if exist backend\tests\__pycache__ (rmdir /s /q backend\tests\__pycache__)

remove:
	@if exist venv (rmdir /s /q venv)