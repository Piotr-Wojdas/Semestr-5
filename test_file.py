import numpy as np
import cv2
import heapq

def otsu_threshold(image):
    pixel_counts = np.bincount(image.ravel(), minlength=256)    # wektor liczby pikseli dla każdej wartości intensywności
    total_pixels = image.size   
    
    current_max_variance = 0.0
    best_threshold = 0
    
    # Zmienne pomocnicze do szybkiego obliczania średnich
    sum_total = np.dot(np.arange(256), pixel_counts)
    sum_background = 0
    weight_background = 0
    
    # Iteracja przez wszystkie progi (0-255)
    for t in range(256):
        weight_background += pixel_counts[t]
        if weight_background == 0:
            continue
            
        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break
            
        sum_background += t * pixel_counts[t]
        
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        
        # Obliczenie wariancji międzyklasowej (sigma_b^2)
        inter_class_variance = weight_background * weight_foreground * ((mean_background - mean_foreground) ** 2)
        
        # Szukamy maksimum wariancji
        if inter_class_variance > current_max_variance:
            current_max_variance = inter_class_variance
            best_threshold = t

    # Binaryzacja z wyznaczonym progiem
    binary_img = np.zeros_like(image)
    binary_img[image > best_threshold] = 255
    
    return best_threshold, binary_img

def watershed(image, markers):

    # Obliczamy gradient obrazu 
    kernel = np.ones((3,3), np.uint8)
    gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    
    height, width = image.shape
    labels = markers.copy()
    is_visited = np.zeros((height, width), dtype=bool)
    
    # Kolejka priorytetowa: przechowuje krotki (wartość_gradientu, x, y)
    pq = []
    
    # Inicjalizacja: Dodajemy sąsiadów markerów do kolejki    
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    
    # Wstępne wypełnienie kolejki   
    marker_y, marker_x = np.where(labels > 0) # znajdujemy współrzędne markerów za pomocą numpy
    
    for x, y in zip(marker_x, marker_y):
        for i in range(4):  # iterujemy po 4 sąsiadach
            nx, ny = x + dx[i], y + dy[i]
            if 0 <= nx < width and 0 <= ny < height:
                if labels[ny, nx] == 0 and not is_visited[ny, nx]:
                    # Dodajemy do kolejki piksel nieznany, priorytetem jest jego gradient
                    heapq.heappush(pq, (int(gradient[ny, nx]), nx, ny))
                    is_visited[ny, nx] = True
                   
    
    # Reset visited dla nieoznaczonych (powyższa pętla była tylko do znalezienia krawędzi)
    is_visited[:] = False
    pq = []
    
    # Znajdźmy brzegowe piksele markerów
    # Używamy dylatacji markerów, żeby znaleźć ich obwódkę
    dilated_markers = cv2.dilate(labels.astype(np.uint8), kernel)
    boundary_mask = (dilated_markers > 0) & (labels == 0)
    by, bx = np.where(boundary_mask)
    
    for x, y in zip(bx, by):
        heapq.heappush(pq, (int(gradient[y, x]), x, y))
        is_visited[y, x] = True

    print(f"Rozpoczynam algorytm Watershed (kolejka: {len(pq)} elementów)...")

    # 3. Główna pętla zalewania
    while pq:
        val, x, y = heapq.heappop(pq)
        
        # Znajdź etykietę od sąsiada markeru
        neighbor_labels = []
        for i in range(4):
            nx, ny = x + dx[i], y + dy[i]
            if 0 <= nx < width and 0 <= ny < height:
                if labels[ny, nx] > 0:
                    neighbor_labels.append(labels[ny, nx])
        
        if neighbor_labels:
            # Przypisz bierzemy pierwszą znalezioną etykietę sąsiada
            labels[y, x] = neighbor_labels[0]
            
            # Dodaj nieodwiedzonych sąsiadów tego piksela do kolejki
            for i in range(4):
                nx, ny = x + dx[i], y + dy[i]
                if 0 <= nx < width and 0 <= ny < height:
                    if not is_visited[ny, nx] and labels[ny, nx] == 0:
                        heapq.heappush(pq, (int(gradient[ny, nx]), nx, ny))
                        is_visited[ny, nx] = True

    return labels

def process_image_otsu_watershed(image):
    
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    # Binaryzacja Otsu 
    otsu_thresh, binary = otsu_threshold(img)
    print(f"Wyznaczony próg Otsu: {otsu_thresh}")

    # Wyznaczanie markerów - Erozja
    kernel = np.ones((3,3), np.uint8)
    sure_fg = cv2.erode(binary, kernel, iterations=3)

    # Wyznaczanie tła - Dylatacja
    sure_bg_mask = cv2.dilate(binary, kernel, iterations=3)
    
    markers = np.zeros_like(img, dtype=np.int32)

    # Oznaczamy pewne tło wartością  1
    markers[sure_bg_mask == 0] = 1 
    
    # Oznaczamy pewne obiekty wartością 2
    markers[sure_fg == 255] = 2
    
    # Obszar nieznany (pomiędzy pewnym tłem a pewnym obiektem) pozostaje jako 0
    
    
    # Watershed
    result_labels = watershed(img, markers)
    
    # Wizualizacja wyników 
    
    display_result = np.zeros_like(img, dtype=np.uint8)
    display_result[result_labels == 1] = 50   # Tło
    display_result[result_labels == 2] = 255  # Obiekt

    # nałożenie granic na oryginał
    vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)    
    vis_img[display_result == 255] = [0, 255, 0] 

    return binary, markers, display_result

img_binary, img_markers, img_result = process_image_otsu_watershed('Computer vision/lista 1/a.jfif')
cv2.imshow('1. Otsu Binary', img_binary)
cv2.imshow('2. Watershed Result', img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()