import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt, label

def f_otsu(hist, img):
    va_N = sum(hist)
    va_L = len(hist)
    l_p = [hist[n]/va_N for n in range(va_L)]
    va_max_sigma = 0
    va_k = 0

    for k in range(va_L-1):
        va_w0 = sum(l_p[0:k+1])
        va_w1 = sum(l_p[k+1:va_L])
        va_u0 = sum([i*l_p[i] for i in range(0, k+1)]) / va_w0
        va_u1 = sum([i*l_p[i] for i in range(k+1, va_L)]) / va_w1
        va_sigma = va_w0 * va_w1 * (va_u1 - va_u0)**2

        if va_sigma > va_max_sigma:
            va_max_sigma = va_sigma
            va_k = k

    o_img_bin = np.zeros_like(img)
    o_img_bin[img > va_k] = 1
    o_img_bin[img <= va_k] = 0
    return va_k, o_img_bin


def f_watershed(o_img):
    l_hist, _ = np.histogram(o_img, bins=256, range=(0, 255))
    otsu_val, o_img_bin = f_otsu(l_hist, o_img)

    l_distance = distance_transform_edt(o_img_bin)
    l_local_max = ndi.maximum_filter(l_distance, size=25) == l_distance
    o_markers, _ = label(l_local_max)
    o_img_bin_inv = 1 - o_img_bin

    o_labels = ndi.watershed_ift(o_img_bin_inv.astype(
        np.uint8), o_markers.astype(np.int32))

    return o_labels, o_img_bin, o_markers, l_distance, otsu_val


def f_show(o_img, o_img_bin, o_markers, o_labels):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 4, 1)
    plt.imshow(o_img, cmap='gray')
    plt.title("Oryginał")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(o_img_bin, cmap='gray')
    plt.title(f"Binaryzacja (próg: {otsu_val})")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(o_markers, cmap='nipy_spectral')
    plt.title("Markery")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(o_labels, cmap='nipy_spectral')
    plt.title("Watershed")
    plt.axis('off')

    plt.show()


l_paths = ["Computer vision/lista 6/coin.png","Computer vision/lista 1/a.jfif"]
for va_path in l_paths:
    o_img = cv2.imread(va_path, cv2.IMREAD_GRAYSCALE)
    if o_img is None:
        print(f"Nie można wczytać obrazu: {va_path}")
        continue
    o_labels, o_img_bin, o_markers, l_distance, otsu_val = f_watershed(o_img)
    f_show(o_img, o_img_bin, o_markers, o_labels)