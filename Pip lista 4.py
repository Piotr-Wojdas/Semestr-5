import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --- KONFIGURACJA PLIKÓW ---
FILE_NORMAL = "Przebieg normalny.xlsx"
FILE_DISTURBED = "Przebieg zaburzony.xlsx"

def load_and_prep_data(filename):
    """Wczytuje dane, konwertuje przecinki na kropki i skaluje temperatury."""
    # Wczytanie danych - obsługuje zarówno CSV jak i Excel
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        df = pd.read_excel(filename)
    else:
        # Dla plików CSV spróbuj najpierw bez separatora, potem z średnikiem
        try:
            df = pd.read_csv(filename, encoding='utf-8')
        except (UnicodeDecodeError, FileNotFoundError):
            try:
                df = pd.read_csv(filename, sep=';', encoding='utf-8')
            except:
                # Ostatnia próba - używamy latin1 dla polskich znaków
                df = pd.read_csv(filename, sep=';', encoding='latin1')
    
    # Dane w pliku są jako integery np. 230 -> 23.0 stopnie. Dzielimy przez 10.
    # Kolumny: Time, z (Tz), y (T), u (Pg), Q1 (wentylator)
    
    # Mapowanie nazw kolumn dla wygody
    df = df.rename(columns={
        'z (Tz)': 'z', 
        'y (T)': 'y', 
        'u (Pg)': 'u',
        'Q1 (wentylator)': 'fan'
    })
    
    # Skalowanie
    df['z'] = df['z'] / 10.0
    df['y'] = df['y'] / 10.0
    
    return df

# 1. Wczytanie danych
df_norm = load_and_prep_data(FILE_NORMAL)
df_dist = load_and_prep_data(FILE_DISTURBED)

print(f"Dane normalne: {len(df_norm)} wierszy")
print(f"Dane zaburzone: {len(df_dist)} wierszy")

# --- CZĘŚĆ 1: IDENTYFIKACJA (Zad 1 & 2) ---

# Model: y(n+1) - y(n) = c * (z(n) - y(n)) + d * u(n)
# Y_target = y(n+1) - y(n)
# X_features = [(z(n)-y(n)), u(n)]

# Przygotowanie macierzy X i Y dla regresji
n_samples = len(df_norm) - 1
X = np.zeros((n_samples, 2))
Y = np.zeros(n_samples)

for i in range(n_samples):
    y_curr = df_norm['y'].iloc[i]
    y_next = df_norm['y'].iloc[i+1]
    z_curr = df_norm['z'].iloc[i]
    u_curr = df_norm['u'].iloc[i]
    
    X[i, 0] = z_curr - y_curr  # Cecha 1: różnica temperatur
    X[i, 1] = u_curr           # Cecha 2: sterowanie
    Y[i] = y_next - y_curr     # Cel: przyrost temperatury

# Rozwiązanie metodą najmniejszych kwadratów
reg = LinearRegression(fit_intercept=False) # fit_intercept=False, bo model fizyczny nie ma stałej bias
reg.fit(X, Y)
c, d = reg.coef_

print("\n--- WYNIKI IDENTYFIKACJI ---")
print(f"Parametr c (wpływ otoczenia): {c:.6f}")
print(f"Parametr d (moc grzałki): {d:.6f}")

# Weryfikacja (Zad 2) - Symulacja modelu na danych uczących
y_sim = [df_norm['y'].iloc[0]]
for i in range(n_samples):
    y_curr = y_sim[-1] # Bierzemy POPRZEDNIĄ symulowaną wartość (rekurencja)
    z_curr = df_norm['z'].iloc[i]
    u_curr = df_norm['u'].iloc[i]
    
    y_next_pred = y_curr + c * (z_curr - y_curr) + d * u_curr
    y_sim.append(y_next_pred)

# Obliczenie błędu MSE
mse = mean_squared_error(df_norm['y'], y_sim)
print(f"Błąd średniokwadratowy (MSE) modelu: {mse:.6f}")

# Wykres weryfikacji
plt.figure(figsize=(12, 5))
plt.plot(df_norm['y'], label='Rzeczywista (y)', color='blue', alpha=0.7)
plt.plot(y_sim, label='Model (y_hat)', color='red', linestyle='--', alpha=0.8)
plt.title("Zad 2: Weryfikacja modelu (Przebieg normalny)")
plt.xlabel("Czas [s]")
plt.ylabel("Temperatura [°C]")
plt.legend()
plt.grid(True)
plt.show()

# --- CZĘŚĆ 2: DETEKCJA ANOMALII (Zad 1, 3, 4) ---

def predict_h_steps(df, h, c, d):
    """
    Generuje predykcję y_hat na h kroków w przód dla każdej chwili n.
    Zwraca tablicę residuów: r = y_real - y_pred
    """
    residuals = []
    # Nie możemy robić predykcji dla pierwszych h próbek (brak historii) 
    # ani ostatnich (brak przyszłości w pliku), ale uprościmy:
    # Dla każdego punktu 'i', startujemy symulację od 'i' i idziemy 'h' kroków
    # i porównujemy wynik z prawdziwym y w 'i+h'.
    
    max_idx = len(df) - h
    
    # Tablica samych zer na start, żeby wyrównać długość wykresu
    residuals = [0] * h 
    
    for i in range(max_idx):
        # Start symulacji w chwili i
        y_sim_h = df['y'].iloc[i]
        
        # Symulujemy h kroków w przód używając znanych wejść u i z
        for k in range(h):
            idx = i + k
            z_val = df['z'].iloc[idx]
            u_val = df['u'].iloc[idx]
            y_sim_h = y_sim_h + c * (z_val - y_sim_h) + d * u_val
            
        # Prawdziwa wartość w chwili i+h
        y_real_future = df['y'].iloc[i+h]
        
        # Residuum
        r = y_real_future - y_sim_h
        residuals.append(r)
        
    return np.array(residuals)

horizons = [1, 5, 10, 20]
results = {}

# Obliczenia dla obu zbiorów
for h in horizons:
    r_norm = predict_h_steps(df_norm, h, c, d)
    r_dist = predict_h_steps(df_dist, h, c, d)
    results[h] = (r_norm, r_dist)

# Zad 3 & 4: Wykresy residuów i propozycja progów
fig, axes = plt.subplots(len(horizons), 1, figsize=(12, 15), sharex=False)
fig.suptitle("Zad 3 & 4: Residua dla różnych horyzontów (h)", fontsize=16)

for idx, h in enumerate(horizons):
    r_norm, r_dist = results[h]
    
    # Automatyczna propozycja progu: max z wartości absolutnej normalnego + 20% marginesu
    threshold = np.max(np.abs(r_norm)) * 1.5
    if threshold < 0.2: threshold = 0.2 # Minimalny próg szumu
    
    ax = axes[idx]
    ax.plot(r_norm, label='Normalny', color='green', alpha=0.6)
    ax.plot(r_dist, label='Zaburzony (z wentylatorem)', color='red', alpha=0.7)
    
    # Rysowanie linii progu
    ax.axhline(threshold, color='black', linestyle='--', label=f'Próg +/- {threshold:.2f}')
    ax.axhline(-threshold, color='black', linestyle='--')
    
    # Oznaczenie momentu włączenia wentylatora (jeśli Q1 > 0)
    # Znajdźmy pierwszy moment gdzie wentylator działa
    fan_on_indices = df_dist.index[df_dist['fan'] == 1].tolist()
    if fan_on_indices:
        ax.axvline(fan_on_indices[0], color='orange', linestyle=':', label='Start wentylatora')

    ax.set_title(f"Horyzont h = {h}")
    ax.set_ylabel("Residuum (y_real - y_pred)")
    ax.legend(loc='upper right')
    ax.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()

# --- ZADANIE 6: ALGORYTM RLS (Adaptacja parametrów) ---
# Recursive Least Squares dla danych NORMALNYCH
# Pokażemy jak c i d zbiegają do wartości wyliczonych wcześniej

def run_rls(df):
    N = len(df) - 1
    theta = np.zeros((2, 1)) # [c, d]^T
    P = np.eye(2) * 1000     # Macierz kowariancji, duża na start
    lambda_factor = 0.99     # Współczynnik zapominania (opcjonalny, 1.0 = pamięta wszystko)
    
    c_history = []
    d_history = []
    
    for i in range(N):
        y_curr = df['y'].iloc[i]
        y_next = df['y'].iloc[i+1]
        z_curr = df['z'].iloc[i]
        u_curr = df['u'].iloc[i]
        
        # Wektor cech phi [z-y, u]
        phi = np.array([[z_curr - y_curr], [u_curr]])
        
        # Oczekiwane wyjście (różnica temperatur)
        y_target = y_next - y_curr
        
        # Krok predykcji błędu a priori
        e = y_target - np.dot(phi.T, theta)
        
        # Aktualizacja wzmocnienia
        num = np.dot(P, phi)
        den = lambda_factor + np.dot(np.dot(phi.T, P), phi)
        k = num / den
        
        # Aktualizacja parametrów
        theta = theta + k * e
        
        # Aktualizacja macierzy P
        P = (P - np.dot(np.dot(k, phi.T), P)) / lambda_factor
        
        c_history.append(theta[0, 0])
        d_history.append(theta[1, 0])
        
    return c_history, d_history

c_hist, d_hist = run_rls(df_norm)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(c_hist, color='purple')
plt.axhline(c, color='black', linestyle='--', label='c (statyczne)')
plt.title("Zad 6: Adaptacja parametru c (RLS)")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(d_hist, color='brown')
plt.axhline(d, color='black', linestyle='--', label='d (statyczne)')
plt.title("Zad 6: Adaptacja parametru d (RLS)")
plt.xlabel("Krok czasowy")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()