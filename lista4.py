import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def rozwiaz_projekt_czesc_1(sciezka_pliku):
    # 1. Wczytanie danych bezpośrednio z formatu Excel (.xlsx)
    # Wykorzystujemy silnik openpyxl do obsługi plików binarnych
    try:
        data = pd.read_excel(sciezka_pliku, engine='openpyxl')
    except Exception as e:
        print(f"Błąd podczas wczytywania pliku: {e}")
        return

    # Przygotowanie osi czasu i obliczenie kroku próbkowania dt
    data['Time'] = pd.to_datetime(data['Time'])
    data['Seconds'] = (data['Time'] - data['Time'].iloc[0]).dt.total_seconds()
    
    # Obliczamy średni odstęp czasu między próbkami
    dt = data['Seconds'].diff().iloc[1:].mean()
    if np.isnan(dt) or dt == 0:
        dt = 1.0  # Wartość domyślna, jeśli próbkowanie wynosi 1s
    
    # Przypisanie wektorów danych
    y = data['y (T)'].values
    u = data['u (Pg)'].values
    z = data['z (Tz)'].values
    N = len(y)

    # 2. Identyfikacja parametrów c i d metodą Najmniejszych Kwadratów (MNK)
    # Układamy równanie w formie Y = X * theta
    # gdzie theta = [c, d]^T
    
    Y_ls = []
    X_ls = []
    
    for n in range(N - 1):
        # Przyrost temperatury znormalizowany przez krok dt
        Y_ls.append((y[n+1] - y[n]) / dt)
        # Składowe modelu: u oraz różnica (z - y)
        X_ls.append([u[n], z[n] - y[n]])
        
    Y_ls = np.array(Y_ls)
    X_ls = np.array(X_ls)
    
    # Rozwiązanie numeryczne macierzowego równania regresji
    theta, residuals, rank, s = np.linalg.lstsq(X_ls, Y_ls, rcond=None)
    c_est, d_est = theta

    print("-" * 30)
    print(f"PARAMETRY MODELU (Identyfikacja):")
    print(f"Parametr c (grzałka): {c_est:.8f}")
    print(f"Parametr d (straty):  {d_est:.8f}")
    print(f"Wyznaczony krok dt:   {dt:.2f} s")
    print("-" * 30)

    # 3. Wygenerowanie serii przewidywanych wyjść (Zad. 3)
    horyzonty = [1, 5, 20]
    predykcje = {}

    for h in horyzonty:
        y_hat = np.zeros(N)
        # Przez pierwsze h kroków model nie ma wystarczającej historii, używamy pomiarów
        y_hat[:h] = y[:h]
        
        for n in range(h, N):
            # Punktem wyjścia jest pomiar sprzed h kroków
            temp_pred = y[n-h]
            # Symulujemy model iteracyjnie h kroków w przód
            for k in range(n-h, n):
                temp_pred = temp_pred + dt * (c_est * u[k] + d_est * (z[k] - temp_pred))
            y_hat[n] = temp_pred
        
        predykcje[h] = y_hat

    # 4. Zobrazowanie wyników na wykresie (Zad. 4)
    plt.figure(figsize=(12, 7))
    
    # Przebieg rzeczywisty
    plt.plot(data['Seconds'], y, color='black', label='Rzeczywisty pomiar $y_n$', linewidth=1.2)
    
    # Przebiegi przewidywane dla różnych h
    colors = ['red', 'green', 'blue']
    for i, h in enumerate(horyzonty):
        plt.plot(data['Seconds'][h:], predykcje[h][h:], color=colors[i], 
                 linestyle='--', label=f'Predykcja $\hat{{y}}_n$ (h={h})', alpha=0.8)
        
    plt.title('Identyfikacja obiektu i weryfikacja predykcji modelu', fontsize=14)
    plt.xlabel('Czas [s]', fontsize=12)
    plt.ylabel('Temperatura [$^\circ$C]', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    # Zapis i wyświetlenie
    plt.savefig('wykres_identyfikacji.png', dpi=300)
    plt.show()

    return c_est, d_est

# Uruchomienie programu - upewnij się, że plik znajduje się w tym samym folderze
# Jeśli Twoja nazwa pliku to "Przebieg normalny-1.xlsx", użyj jej poniżej:
c, d = rozwiaz_projekt_czesc_1('Przebieg normalny.xlsx')