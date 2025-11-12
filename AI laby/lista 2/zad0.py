import numpy as np

# funkcja aktywacji - sigmoidalna
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# pochodna funkcji sigmoidalnej
def sigmoid_derivative(s):
    return s * (1 - s)

# XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# parametry
n_x = 2  # liczba neuronow wejsciowych
n_h = 2  # liczba neuronow w warstwie ukrytej
n_y = 1  # liczba neuronow wyjściowych
lr = 0.1 # wspolczynnik uczenia
epochs = 10000 # liczba epok

# wprowadzenie wag i biasów 
# warstwa ukryta:
W1 = np.random.uniform(size=(n_x, n_h))
b1 = np.random.uniform(size=(1, n_h))

# warstwa wyjściowa:
W2 = np.random.uniform(size=(n_h, n_y))
b2 = np.random.uniform(size=(1, n_y))

# trening sieci
for i in range(epochs):
    # propagacja w przód
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    # propagacja w tył
    error = y - A2
    dZ2 = error * sigmoid_derivative(A2)
    
    error_hidden_layer = dZ2.dot(W2.T)
    dZ1 = error_hidden_layer * sigmoid_derivative(A1)

    # aktualizacja wag i biasów
    W2 += A1.T.dot(dZ2) * lr
    b2 += np.sum(dZ2, axis=0, keepdims=True) * lr

    W1 += X.T.dot(dZ1) * lr
    b1 += np.sum(dZ1, axis=0, keepdims=True) * lr

    if i % (epochs // 10) == 0:
        loss = np.mean(np.square(y - A2))
        print(f"Epoka: {i}, Błąd (Loss): {loss}")

print("\nFinalne wagi warstwy ukrytej (W1):\n", W1)
print("Finalne wagi warstwy wyjściowej (W2):\n", W2)


def nn_for_xor(X_test, W1, b1, W2, b2):
    Z1 = np.dot(X_test, W1) + b1
    A1 = sigmoid(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    for i in range(len(X_test)):
       
        print(f"{X_test[i]} -> {1 if A2[i][0] > 0.5 else 0} (Oczekiwano: {y[i][0]})")
        print("Dokładny wynik:", A2[i][0])

nn_for_xor(X, W1, b1, W2, b2)


