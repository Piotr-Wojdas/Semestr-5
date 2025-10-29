import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


BASE_DIR = Path(__file__).resolve().parent
path = BASE_DIR / "a.jfif"

def load_image(path):
	img = Image.open(path)
	img_rgb = img.convert('RGB') # konwersja do RGB
	return img_rgb
	
rgb = load_image(path)


# =============================================================================================================================================================


def canny(img):
    initial_low = 50
    initial_high = 150
    array = np.array(img.convert('L')) 	# zamiana na macierz, i zamiana na skalę szarości
    edges = cv2.Canny(array, initial_low, initial_high, L2gradient=True)

    # wizualizacja 
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(left=0.1, bottom=0.25)
    img_plot = ax.imshow(edges, cmap='gray')
    ax.set_title('Canny')
    ax.axis('off')

    # Suwaki
    ax_low = plt.axes([0.1, 0.1, 0.8, 0.03])
    ax_high = plt.axes([0.1, 0.15, 0.8, 0.03])
    slider_low = Slider(ax=ax_low, label='Low', valmin=0, valmax=255, valinit=initial_low)
    slider_high = Slider(ax=ax_high, label='High', valmin=0, valmax=255, valinit=initial_high)

    
    def update(val):
        low = slider_low.val
        high = slider_high.val
        new_edges = cv2.Canny(array, int(low), int(high))
        img_plot.set_data(new_edges)
        fig.canvas.draw_idle()

    slider_low.on_changed(update)
    slider_high.on_changed(update)

    plt.show()
canny(rgb)

'''
Wnioski: 
1. Kiedy high jest wysokie a low niskie, tylko najostrzejsze krawędzie są wykrywane, 
bo przez niski próg low dużo pixeli jest klasyfikowanych jako słabe krawędzie dzięki czemu krawędzie są wydłużone i się nie przerywają

2. Dla niskich wartości low i high bardzo dużo pixeli jest wykrywanych jako krawędzie, nawet najmniejsze zmiany w jasności są widoczne

3. Gdy obie wartości są wysokie, widać jedynie te najmocniejsze krawędzie ale przez niewielkie okno dla słabych krawędzi, są one przerywane
'''


# =============================================================================================================================================================


def Prog(img):

    array = np.array(img)
    gray_img = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    initial_thresh = 127

    # wizualizacja
    fig, ax = plt.subplots(figsize=(10, 8)) # Ustawienie większego rozmiaru okna
    plt.subplots_adjust(left=0.1, bottom=0.25)
    
    _, thresh_img = cv2.threshold(gray_img, initial_thresh, 255, cv2.THRESH_BINARY)
    img_plot = ax.imshow(thresh_img, cmap='gray')
    ax.set_title('Progowanie')
    ax.axis('off')

    # suwak
    ax_thresh = plt.axes([0.1, 0.1, 0.8, 0.03])
    slider_thresh = Slider(ax=ax_thresh, label='Threshold', valmin=0, valmax=255, valinit=initial_thresh)

    def update(val):
        thresh_val = slider_thresh.val
        _, new_thresh_img = cv2.threshold(gray_img, thresh_val, 255, cv2.THRESH_BINARY)
        img_plot.set_data(new_thresh_img)
        fig.canvas.draw_idle()
    slider_thresh.on_changed(update)
    plt.show()
Prog(rgb)

'''
1. Dla wysokiego progu coraz mniej pixeli przechodzi przez próg i nie kwalifikuje na kolor biały więc większość obrazu jest czarna
2. Dla niskiego progu coraz więcej pixeli jest powyżej progu więc obraz posiada więcej białych obszarów
'''


# =============================================================================================================================================================


def gauss(img):
    array = np.array(img)
    initial_kernel = 5

    # Wizualizacja
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(left=0.1, bottom=0.25)

    blurred_img = cv2.GaussianBlur(array, (initial_kernel, initial_kernel), 0)
    img_plot = ax.imshow(blurred_img)
    ax.set_title('Rozmycie Gaussa')
    ax.axis('off')

    # Suwak
    ax_kernel = plt.axes([0.1, 0.1, 0.8, 0.03])
    slider_kernel = Slider(ax=ax_kernel, label='Kernel', valmin=1, valmax=51, valstep=2, valinit=initial_kernel)   # step 2 bo kernel musi być nieparzysty

    def update(val):
        kernel = int(slider_kernel.val)
        new_blurred_img = cv2.GaussianBlur(array, (kernel, kernel), 0)
        img_plot.set_data(new_blurred_img)
        fig.canvas.draw_idle()
    slider_kernel.on_changed(update)
    plt.show()
gauss(rgb)

'''
Im większy kernel, tym obraz jest bardziej rozmazany, ponieważ pixele o większej wartości rozmazują szerszy obszar.
'''