"""Полная демонстрация ЛР1: фильтр Собеля.

1) обработка изображения обеими реализациями (main.py);
2) сравнение быстродействия на разных размерах (benchmark.py);
3) модульные тесты;
4) открытие результатов.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)


def step(title, args):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70, flush=True)
    subprocess.run([sys.executable, *args], check=True)


step("1. Фильтр Собеля: OpenCV и ручная реализация", ["main.py", "images/input.png", "--repeats", "5"])
step("2. Сравнение быстродействия (около 20 секунд)", ["benchmark.py"])
step("3. Модульные тесты", ["-m", "unittest", "-v"])

print("\n" + "=" * 70 + "\n4. Результаты\n" + "=" * 70)
for f in ["examples/comparison.png", "examples/benchmark.png"]:
    print("Открываю", f)
    if hasattr(os, "startfile"):
        os.startfile(ROOT / f)
print("\nГотово. Картинки: папки output/ и examples/, отчёт: README.md")
