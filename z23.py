import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --- KONFIGURACJA PLIKÓW ---
FILE_NORMAL = "Przebieg normalny.xlsx"
FILE_DISTURBED = "Przebieg zaburzony.xlsx"

def load_and_prep_data(filename):
    """Wczytuje dane, konwertuje jednostki i mapuje kolumny."""
    try:
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(filename)
        else:
            df = pd.read_csv(filename, sep=';', encoding='utf-8')
    except Exception as e:
        print(f"Błąd ładowania pliku {filename}: {e}")
        return None

    # Mapowanie i skalowanie (230 -> 23.0)
    df = df.rename(columns={
        'z (Tz)': 'z', 
        'y (T)': 'y', 
        'u (Pg)': 'u',
        'Q1 (wentylator)': 'fan'
    })
    df['z'] = df['z'] / 10.0
    df['y'] = df['y'] / 10.0
    
    return df

# 1. Wczytanie danych
df_norm = load_and_prep_data(FILE_NORMAL)
if df_norm is None:
    # Tworzenie pustego DataFrame dla celów demonstracji struktury, jeśli plik nie istnieje
    df_norm = pd.DataFrame(columns=['y', 'z', 'u'])

# --- REALIZACJA ZADANIA 2 PODPUNKT 3 ---

# Przygotowanie danych do regresji (Model: delta_y = c * delta_T + d * u)
# N próbek daje N-1 przyrostów
n_samples = len(df_norm) - 1
if n_samples > 0:
    Y = (df_norm['y'].shift(-1) - df_norm['y']).dropna().values
    X = np.column_stack([
        (df_norm['z'] - df_norm['y']).iloc[:-1].values, # Cecha: z(n) - y(n)
        df_norm['u'].iloc[:-1].values                  # Cecha: u(n)
    ])

    # Identyfikacja (OLS)
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, Y)
    c, d = reg.coef_

    # Symulacja weryfikacyjna (Zad 2)
    y_real = df_norm['y'].values
    z_real = df_norm['z'].values
    u_real = df_norm['u'].values
    
    y_sim = np.zeros(len(df_norm))
    y_sim[0] = y_real[0]
    
    for n in range(n_samples):
        y_sim[n+1] = y_sim[n] + c * (z_real[n] - y_sim[n]) + d * u_real[n]

    mse = mean_squared_error(y_real, y_sim)

    print("-" * 40)
    print("WYNIKI ZADANIA 2:")
    print(f"Kryterium jakości: MSE = {mse:.8f}")
    print(f"Wartość parametru c (otoczenie): {c:.8f}")
    print(f"Wartość parametru d (grzałka):   {d:.8f}")
    print("-" * 40)

    # Wykres dopasowania
    plt.figure(figsize=(10, 5))
    plt.plot(y_real, 'b-', label='Dane pomiarowe (y)', alpha=0.6)
    plt.plot(y_sim, 'r--', label='Model symulowany', linewidth=1.5)
    plt.title(f"Zad 2: Weryfikacja modelu (c={c:.4f}, d={d:.4f})")
    plt.xlabel("Próbka (n)")
    plt.ylabel("Temperatura [°C]")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Brak danych do przeprowadzenia identyfikacji.")