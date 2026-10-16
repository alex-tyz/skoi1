"""Фильтр Собеля: OpenCV и ручная реализация на Python, сравнение результата и времени.

Пример:  python main.py images/input.png --repeats 5
"""
import argparse
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from sobel import sobel_native, sobel_opencv, to_uint8


def measure(function, repeats):
    """Запустить функцию repeats раз; вернуть результат последнего запуска и медианное время."""
    times = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        times.append(time.perf_counter() - start)
    return result, statistics.median(times)


def main():
    import os
    os.chdir(Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser(description="Sobel filter with OpenCV and pure Python")
    parser.add_argument("input", type=Path, nargs="?", default=Path("images/input.png"), help="Входное изображение")
    parser.add_argument("--repeats", type=int, default=5, help="Число запусков для замера времени")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Папка для результатов")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    gray = cv2.imread(str(args.input), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        parser.error(f"cannot read image {args.input}")

    (gx, gy, mag_cv), t_cv = measure(lambda: sobel_opencv(gray), args.repeats)
    (_, _, mag_py), t_py = measure(lambda: sobel_native(gray), args.repeats)
    mag_py = np.asarray(mag_py)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "gray.png"), gray)
    cv2.imwrite(str(out / "sobel_gx.png"), to_uint8(np.abs(gx)))
    cv2.imwrite(str(out / "sobel_gy.png"), to_uint8(np.abs(gy)))
    cv2.imwrite(str(out / "sobel_opencv.png"), to_uint8(mag_cv))
    cv2.imwrite(str(out / "sobel_python.png"), to_uint8(mag_py))

    h, w = gray.shape
    print(f"Изображение {args.input}: {w}x{h}, запусков: {args.repeats}")
    print(f"OpenCV: {t_cv * 1000:10.3f} мс (медиана)")
    print(f"Python: {t_py * 1000:10.3f} мс (медиана)")
    print(f"Ускорение OpenCV: {t_py / t_cv:.0f} раз")
    print(f"Максимальная разница модуля градиента: {np.abs(mag_cv - mag_py).max():.3g}")


if __name__ == "__main__":
    main()
