import numpy as np

# funkcja aktyacji - sigmoidalna
def f(x):
    return 1 / (1 + np.exp(-x))

# pochodna tej funkcji
def df_ds(s):
    return s * (1 - s)

# XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# parametry
k = 2  # liczba neuronow wejsciowych
h = 2  # liczba neuronow w warstwie ukrytej
r = 1  # liczba neuronow wyjściowych
lr = 0.1 # wspolczynnik uczenia
epochs = 10000 # liczba epok

# wprowadzenie wag i biasów 
# warstwa ukryta:
hidden_weights = np.random.uniform(size=(k, h))
hidden_bias = np.random.uniform(size=(1, h))

# warstwa wyjściowa:
output_weights = np.random.uniform(size=(h, r))
output_bias = np.random.uniform(size=(1, r))

# trening sieci
for i in range(epochs):
    # propagacja w przód
    s_h = np.dot(X, hidden_weights) + hidden_bias
    hidden_layer_output = f(s_h)

    s_o = np.dot(hidden_layer_output, output_weights) + output_bias
    predicted_output = f(s_o)

    # propagacja w tył
    error = y - predicted_output

    d_predicted_output = error * df_ds(predicted_output)
    
    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * df_ds(hidden_layer_output)

    # aktualizacja wag i biasów
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr

    hidden_weights += X.T.dot(d_hidden_layer) * lr
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr

    if i % (epochs // 10) == 0:
        loss = np.sum(np.square(y - predicted_output))
        print(f"Epoka: {i}, Błąd (Loss): {loss}")

print("\nFinalne wagi warstwy ukrytej:\n", hidden_weights)
print("Finalne wagi warstwy wyjściowej:\n", output_weights)


def nn_for_xor(X, hidden_weights, hidden_bias, output_weights, output_bias):
    s_h = np.dot(X, hidden_weights) + hidden_bias
    hidden_layer_output = f(s_h)

    s_o = np.dot(hidden_layer_output, output_weights) + output_bias
    predicted_output = f(s_o)
    for i in range(len(X)):
       
        print(f"{X[i]} -> {1 if predicted_output[i][0] > 0.5 else 0} (Oczekiwano: {y[i][0]})")
        print("Dokładny wynik:", predicted_output[i][0])

nn_for_xor(np.array([[0, 1]]), hidden_weights, hidden_bias, output_weights, output_bias)


