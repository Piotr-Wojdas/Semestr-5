import numpy as np
import cv2

def manual_bilinear_interpolation(image, new_h, new_w):
    
    # Pobranie wymiarów oryginalnego obrazu
    h, w, c = image.shape
    
    # Utworzenie pustego obrazu o docelowych wymiarach
    resized_image = np.zeros((new_h, new_w, c), dtype=np.uint8)
    
    # Obliczenie współczynników skalowania
    x_ratio = w / new_w
    y_ratio = h / new_h
    
    # Iteracja przez każdy piksel nowego (przeskalowanego) obrazu
    for y_new in range(new_h):
        for x_new in range(new_w):
            # Mapowanie współrzędnych z nowego obrazu na oryginalny
            x_orig = x_new * x_ratio
            y_orig = y_new * y_ratio
            
            # Znalezienie 4 najbliższych sąsiadów w oryginalnym obrazie (A, B, C, D)
            # A (x1, y1) --- B (x2, y1)
            # |               |
            # C (x1, y2) --- D (x2, y2)

            x1 = int(np.floor(x_orig))
            y1 = int(np.floor(y_orig))
            
            # Używamy min, aby nie wyjść poza granice obrazu
            x2 = min(x1 + 1, w - 1)
            y2 = min(y1 + 1, h - 1)
            
            # Pobranie wartości pikseli sąsiadów
            A = image[y1, x1]
            B = image[y1, x2]
            C = image[y2, x1]
            D = image[y2, x2]
            
            # Obliczenie wag (odległości od punktu (x1, y1))
            x_diff = x_orig - x1
            y_diff = y_orig - y1
            
            # Interpolacja w poziomie
            # Górny wiersz: (A * (1-x_diff)) + (B * x_diff)
            top_interp = A * (1 - x_diff) + B * x_diff
            
            # Dolny wiersz: (C * (1-x_diff)) + (D * x_diff)
            bottom_interp = C * (1 - x_diff) + D * x_diff
            
            # Interpolacja w pionie
            final_pixel = top_interp * (1 - y_diff) + bottom_interp * y_diff
            
            # Przypisanie obliczonego piksela do nowego obrazu
            resized_image[y_new, x_new] = final_pixel.astype(np.uint8)
            
    return resized_image

# --- Funkcja do aktualizacji obrazu na podstawie suwaków ---
def update(val):
    # Pobranie aktualnych wartości z suwaków
    new_width = cv2.getTrackbarPos('Width', 'Controls')
    new_height = cv2.getTrackbarPos('Height', 'Controls')

    # Zabezpieczenie przed wymiarami równymi 0
    if new_width == 0 or new_height == 0:
        return

    # Wywołanie ręcznej interpolacji
    manual_resized = manual_bilinear_interpolation(img, new_height, new_width)
    
    # Wyświetlenie wyniku
    cv2.imshow('Reczna interpolacja dwuliniowa', manual_resized)


# --- Przykładowe użycie ---

# Wczytanie obrazu
img = cv2.imread('Computer vision/lista 1/a.jfif')

original_h, original_w = img.shape[:2]

# Utworzenie okna na suwaki
cv2.namedWindow('Controls')
cv2.resizeWindow('Controls', 600, 100)

# suwaki
# Zakres od 1 do 3x oryginalny wymiar
cv2.createTrackbar('Width', 'Controls', original_w, original_w * 3, update)
cv2.createTrackbar('Height', 'Controls', original_h, original_h * 3, update)

# Wyświetlenie oryginalnego obrazu
cv2.imshow('Oryginalny', img)

# Pierwsze wywołanie, aby wyświetlić obrazy na starcie
update(0)

# Pętla do utrzymania okna i obsługi zamykania
while True:
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
