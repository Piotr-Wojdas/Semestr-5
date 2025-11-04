from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np
import cv2

BASE_DIR = Path(__file__).resolve().parent
path = BASE_DIR / "a.jfif"

def load_image(path):
	img = Image.open(path)
	img_rgb = img.convert('RGB') # konwersja do RGB
	return img_rgb
	
rgb = load_image(path)

# --------------------------- 1 ---------------------------
def grayscale(img):
	gray = img.convert('L')
	gray.show()	

#grayscale(rgb)	

# --------------------------- 2 ---------------------------
def gaussian_blur(img, radius=5):		# promień rozmycia, im więcej tym bardziej się rozmywa
	blurr = img.filter(ImageFilter.GaussianBlur(radius))
	blurr.show()

#gaussian_blur(rgb)

# --------------------------- 3 ---------------------------
def Canny(img, low=100, high=200):

	array = np.array(img.convert('L')) 	# zamiana na macierz, i zamiana na skalę szarości
	edges = cv2.Canny(array, low, high, L2gradient=True)

	edges_img = Image.fromarray(edges)	# wracamy do PIL
	edges_img.show()

#Canny(rgb)

# --------------------------- 4 ---------------------------
def threshold_bin(img, thresh=128, maxval=255, method='fixed'):
	gray = img.convert('L')
	array = np.array(gray)
	_, thresh_arr = cv2.threshold(array, int(thresh), int(maxval), cv2.THRESH_BINARY)
	thresh_img = Image.fromarray(thresh_arr)
	thresh_img.show()

#threshold_bin(rgb)

# --------------------------- 5 ---------------------------
def rotate_img(img, angle=45):
	rotated = img.rotate(angle, expand=True)	# expand - nie przycina tylko zwiększa obraz
	rotated.show()

#rotate_img(rgb)

# --------------------------- 6 ---------------------------
def resize(img, multiplier=2): 
	m = float(multiplier)
	w, h = img.size
	new_size = (max(1, int(round(w * m))), max(1, int(round(h * m))))
	resized = img.resize(new_size)
	resized.show()

#resize(rgb)

# --------------------------- 7 ---------------------------
def equal_hist(img):
    rgb = np.array(img)
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)	# konwersja RGB -> YCrCb
    y, cr, cb = cv2.split(ycrcb)	# wyciągamy poziom jasności (y)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge((y_eq, cr, cb))	# łączymy kanały w całość
    rgb_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2RGB)	# konwersja YCrCb -> RGB
    Image.fromarray(rgb_eq).show()

#equal_hist(rgb)

# --------------------------- 8 ---------------------------
def sharpness(img,x=2):
	filter = np.array([

		[0 ,   -x ,   0],
		[-x,   5*x,  -x],
		[0 ,   -x ,   0]

		])

	array = np.array(img)
	sharp = cv2.filter2D(array, -1, filter)	# -1 - zachowujemy głębię oryginału
	Image.fromarray(sharp).show()

#sharpness(rgb)


