import numpy as np
import cv2
import matplotlib.pyplot as plt

def otsu_threshold(img):
    hist = cv2.calcHist([img], [0], None, [256], [0,256]).flatten()
    total = img.size
    sumB, wB, maximum, sum1 = 0.0, 0.0, 0.0, np.dot(np.arange(256), hist)
    threshold = 0
    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum1 - sumB) / wF
        between = wB * wF * (mB - mF) ** 2
        if between > maximum:
            maximum = between
            threshold = t
    return threshold

def manual_watershed(image, markers):
    h, w = image.shape
    labels = markers.copy()
    changed = True
    while changed:
        changed = False
        for y in range(1, h-1):
            for x in range(1, w-1):
                if labels[y, x] == 0:
                    neighbor_labels = [labels[y-1, x], labels[y+1, x], labels[y, x-1], labels[y, x+1]]
                    neighbor_labels = [l for l in neighbor_labels if l > 0]
                    if len(set(neighbor_labels)) == 1:
                        labels[y, x] = neighbor_labels[0]
                        changed = True
    return labels

def segment_with_otsu_and_manual_watershed(img, visualize=False):
    thresh = otsu_threshold(img)
    binary = (img > thresh).astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
    sure_fg = cv2.erode(binary, kernel, iterations=2)
    sure_bg = cv2.dilate(binary, kernel, iterations=3)
    sure_bg = cv2.bitwise_not(sure_bg)
    markers = np.zeros_like(img, dtype=np.int32)
    markers[sure_bg == 1] = 2
    markers[sure_fg == 1] = 1
    labels = manual_watershed(img, markers)
    if visualize:
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs[0, 0].imshow(img, cmap='gray')
        axs[0, 0].set_title('Oryginał')
        axs[0, 1].imshow(binary, cmap='gray')
        axs[0, 1].set_title(f'Binaryzacja Otsu (prog={thresh})')
        axs[0, 2].imshow(sure_fg, cmap='gray')
        axs[0, 2].set_title('Markery obiektów (erozja)')
        axs[1, 0].imshow(sure_bg, cmap='gray')
        axs[1, 0].set_title('Markery tła (dylatacja)')
        axs[1, 1].imshow(markers, cmap='nipy_spectral')
        axs[1, 1].set_title('Markery (1-obiekt, 2-tło)')
        axs[1, 2].imshow(labels, cmap='nipy_spectral')
        axs[1, 2].set_title('Wynik watershed')
        for ax in axs.ravel():
            ax.axis('off')
        plt.tight_layout()
        plt.show()
    return labels

# Przykład użycia:
img = cv2.imread('Computer vision/lista 6/coin.png', 0)
img2 = cv2.imread('Computer vision/lista 1/a.jfif', 0)
labels = segment_with_otsu_and_manual_watershed(img, visualize=True)
labels2 = segment_with_otsu_and_manual_watershed(img2, visualize=True)
