import random
import math

# Cel: Maksymalizacja funkcji f(x) = -0.1*x^4 + 0.8*x^3 - x^2 + 5*sin(2*x) + 3*cos(x) + 20

def fitness_function(x):
    # Ocena osobników: Obliczamy wartość funkcji f(x)
    return -0.1*x**4 + 0.8*x**3 - x**2 + 5*math.sin(2*x) + 3*math.cos(x) + 20

def init_population(size, bounds):
    # Populacja początkowa
    return [random.uniform(bounds[0], bounds[1]) for _ in range(size)]

def selection(population, fitnesses, k=3):
    # selekcja - spośród wylosowanych osobników wybieramy najlepszego
    selection_ix = random.randint(0, len(population)-1)
    for ix in [random.randint(0, len(population)-1) for _ in range(k-1)]:
        if fitnesses[ix] > fitnesses[selection_ix]:
            selection_ix = ix
    return population[selection_ix]

def crossover(p1, p2, r_cross):
    # Krzyżowanie
    c1, c2 = p1, p2
    if random.random() < r_cross:
        alpha = random.random()
        c1 = alpha * p1 + (1 - alpha) * p2
        c2 = (1 - alpha) * p1 + alpha * p2
    return [c1, c2]

def mutation(individual, r_mut, bounds):
    # Z racji że działamy na wartościach float, a nie na bitach, dodajemy szum gaussowski
    if random.random() < r_mut:
        individual += random.gauss(0, 2)  # odchylenie standardowe = 2
        
        individual = max(bounds[0], min(bounds[1], individual))
    return individual

def check_stop_condition(gen, n_iter, current_avg, previous_avg, epsilon=0.01):
    # Warunki stopu. Zatrzymujemy ewolucję jeśli:
    # Warunek 1: Osiągnięto maksymalną liczbę pokoleń
    if gen >= n_iter:
        return True
    
    # Warunek 2: Stagnacja - kolejne pokolenie nie poprawiło wyniku
    if previous_avg is not None:
        if abs(current_avg - previous_avg) <= epsilon:
            return True
    
    return False

def genetic_algorithm():
    # Parametry algorytmu
    bounds = [-5, 10]     # Zakres poszukiwań x 
    n_iter = 500           # Maksymalna liczba pokoleń
    n_pop = 20            # Wielkość populacji
    r_cross = 0.8         # Prawdopodobieństwo krzyżowania
    r_mut = 0.2           # Prawdopodobieństwo mutacji
    epsilon = 0.01        # Próg stagnacji dla warunku stopu

    # Inicjalizacja
    pop = init_population(n_pop, bounds)
    best, best_eval = pop[0], fitness_function(pop[0])
    previous_avg = None   # Średnia ocena poprzedniej populacji
    gen = 0               # Licznik pokoleń

    while True:
        #  Ocena populacji
        fitnesses = [fitness_function(ind) for ind in pop]
        current_avg = sum(fitnesses) / len(fitnesses)  # Średnia ocena populacji
        
        # Sprawdzenie czy znaleźliśmy nowy rekord
        for i in range(n_pop):
            if fitnesses[i] > best_eval:
                best, best_eval = pop[i], fitnesses[i]
                print(f"Pokolenie {gen}: Nowy najlepszy x = {best:.4f}, f(x) = {best_eval:.4f} | "
                      f"średnia: {current_avg:.4f}")

        # Sprawdzenie warunku stopu
        if check_stop_condition(gen, n_iter, current_avg, previous_avg, epsilon):
            if previous_avg is not None and abs(current_avg - previous_avg) <= epsilon:
                print(f"\nZatrzymano: Stagnacja po {gen} pokoleniach")
                print(f"Różnica średnich: |{current_avg:.4f} - {previous_avg:.4f}| = "
                      f"{abs(current_avg - previous_avg):.6f} <= {epsilon}")
            else:
                print(f"\nZatrzymano: Osiągnięto maksymalną liczbę pokoleń ({n_iter})")
            break

        # Selekcja - wybór rodziców
        selected = [selection(pop, fitnesses) for _ in range(n_pop)]

        # Tworzenie następnego pokolenia
        children = []
        for i in range(0, n_pop, 2):
            p1 = selected[i]
            p2 = selected[i+1] if i+1 < n_pop else selected[0]
            # Krzyżowanie i mutacja
            for c in crossover(p1, p2, r_cross):
                child = mutation(c, r_mut, bounds)
                children.append(child)
        
        pop = children[:n_pop]
        
        # Aktualizacja dla następnej iteracji
        previous_avg = current_avg
        gen += 1

    return best, best_eval

best_x, best_f = genetic_algorithm()
print("-" * 30)
print(f"Wynik końcowy:\nNajlepszy osobnik (x): {best_x:.4f}\nWartość funkcji f(x): {best_f:.4f}")