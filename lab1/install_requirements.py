"""Установка библиотек из requirements.txt в текущий интерпретатор (запуск кнопкой Run)."""
import subprocess
import sys
from pathlib import Path

req = Path(__file__).resolve().parent / "requirements.txt"
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req)])
print("\nГотово: библиотеки установлены.")
