"""
ЛР1, вариант 1 — фильтр Собеля.

Две реализации одного алгоритма:
  * sobel_opencv  — через встроенную функцию cv2.Sobel;
  * sobel_native  — вручную на чистом Python (циклы по пикселям, без NumPy-свёрток).
Дополнительно: sobel_numpy — ручная векторизованная версия на NumPy (для сравнения).

Все версии используют одинаковую обработку границ (BORDER_REFLECT_101,
как по умолчанию в OpenCV), поэтому их результаты совпадают.
"""
import math

import cv2
import numpy as np

# Ядра Собеля 3x3
KX = [[-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1]]
KY = [[-1, -2, -1],
      [ 0,  0,  0],
      [ 1,  2,  1]]


# ---------------------------------------------------------------- OpenCV
def sobel_opencv(gray):
    """Градиенты и модуль градиента через cv2.Sobel."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return gx, gy, mag


# ---------------------------------------------------------- чистый Python
def _reflect101(i, n):
    """Индекс с отражением без повтора крайнего пикселя: ... 2 1 | 0 1 2 ... n-1 | n-2 ..."""
    if i < 0:
        return -i
    if i >= n:
        return 2 * n - i - 2
    return i


def sobel_native(gray):
    """Фильтр Собеля вручную: свёртка 3x3 двойным циклом по пикселям.

    gray — двумерный массив (или список списков) яркостей.
    Возвращает gx, gy, mag как списки списков float.
    """
    img = gray.tolist() if hasattr(gray, "tolist") else gray
    h, w = len(img), len(img[0])
    gx = [[0.0] * w for _ in range(h)]
    gy = [[0.0] * w for _ in range(h)]
    mag = [[0.0] * w for _ in range(h)]

    for y in range(h):
        rows = [img[_reflect101(y + dy, h)] for dy in (-1, 0, 1)]
        for x in range(w):
            cols = [_reflect101(x + dx, w) for dx in (-1, 0, 1)]
            sx = sy = 0.0
            for ky in range(3):
                row = rows[ky]
                kxr, kyr = KX[ky], KY[ky]
                for kx in range(3):
                    p = row[cols[kx]]
                    sx += kxr[kx] * p
                    sy += kyr[kx] * p
            gx[y][x] = sx
            gy[y][x] = sy
            mag[y][x] = math.sqrt(sx * sx + sy * sy)
    return gx, gy, mag


# ------------------------------------------------- ручная версия на NumPy
def sobel_numpy(gray):
    """Та же свёртка, но сдвигами массивов NumPy (без готового фильтра)."""
    a = np.pad(gray.astype(np.float64), 1, mode="reflect")  # reflect == REFLECT_101
    h, w = gray.shape
    gx = np.zeros((h, w))
    gy = np.zeros((h, w))
    for ky in range(3):
        for kx in range(3):
            win = a[ky:ky + h, kx:kx + w]
            gx += KX[ky][kx] * win
            gy += KY[ky][kx] * win
    return gx, gy, np.hypot(gx, gy)


def to_uint8(mag):
    """Нормировка модуля градиента в 0..255 для сохранения картинки."""
    m = np.asarray(mag, dtype=np.float64)
    if m.max() > 0:
        m = m / m.max() * 255.0
    return m.astype(np.uint8)
