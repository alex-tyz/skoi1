"""Сравнение быстродействия реализаций фильтра Собеля на изображениях разного размера.

Сохраняет таблицу examples/benchmark.csv, график examples/benchmark.png
и сравнительную картинку examples/comparison.png.
"""
import csv
import statistics
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sobel import sobel_native, sobel_numpy, sobel_opencv, to_uint8

SIZES = [64, 128, 256, 512, 1024]
OUT = Path("examples")


def median_time(function, image, repeats):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function(image)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def save_examples(src):
    gx, gy, mag_cv = sobel_opencv(src)
    mag_py = np.asarray(sobel_native(src)[2])
    images = [src, to_uint8(np.abs(gx)), to_uint8(np.abs(gy)), to_uint8(mag_cv), to_uint8(mag_py)]
    names = ["input_gray", "sobel_gx", "sobel_gy", "sobel_opencv", "sobel_python"]
    for name, img in zip(names, images):
        cv2.imwrite(str(OUT / f"{name}.png"), img)
    cv2.imwrite(str(OUT / "comparison.png"), np.hstack([cv2.resize(i, (256, 256)) for i in images]))


def main():
    import os
    os.chdir(Path(__file__).resolve().parent)
    src = cv2.imread("images/input.png", cv2.IMREAD_GRAYSCALE)
    OUT.mkdir(exist_ok=True)
    save_examples(src)

    rows = []
    for s in SIZES:
        img = cv2.resize(src, (s, s), interpolation=cv2.INTER_AREA)
        ref = sobel_opencv(img)[2]
        diff = max(np.abs(ref - np.asarray(sobel_native(img)[2])).max(),
                   np.abs(ref - sobel_numpy(img)[2]).max())

        t_cv = median_time(sobel_opencv, img, 51)
        t_np = median_time(sobel_numpy, img, 11)
        t_py = median_time(sobel_native, img, 5 if s <= 256 else 3)
        rows.append((s, t_cv * 1000, t_np * 1000, t_py * 1000, diff))
        print(f"{s:5d}x{s:<5d} OpenCV {t_cv*1000:8.3f} мс | NumPy {t_np*1000:8.3f} мс | "
              f"Python {t_py*1000:9.1f} мс | Python/OpenCV {t_py/t_cv:5.0f}x | разница {diff:.3g}")

    with open(OUT / "benchmark.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["size", "opencv_ms", "numpy_ms", "python_ms", "max_diff"])
        w.writerows(rows)

    sizes = [r[0] for r in rows]
    plt.figure(figsize=(7, 4.5))
    plt.plot(sizes, [r[1] for r in rows], "o-", label="OpenCV (cv2.Sobel)")
    plt.plot(sizes, [r[2] for r in rows], "s-", label="NumPy (вручную)")
    plt.plot(sizes, [r[3] for r in rows], "^-", label="Чистый Python")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xticks(sizes, [f"{s}²" for s in sizes])
    plt.xlabel("Размер изображения, пикс.")
    plt.ylabel("Время, мс (лог. шкала)")
    plt.title("Быстродействие фильтра Собеля (медиана)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "benchmark.png", dpi=120)


if __name__ == "__main__":
    main()
