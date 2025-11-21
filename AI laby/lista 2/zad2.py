import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# funkcja aktywacji - sigmoidalna
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(s):
    return s * (1 - s)



# 1. Przygotowanie danych

train_df = pd.read_csv('AI laby/lista 2/penguins_train.csv')
test_df = pd.read_csv('AI laby/lista 2/penguins_test.csv')


# Cechy wejściowe (X) - 5 cech
train_df['Sex'] = train_df['Sex'].apply(lambda x: 1 if x == 'MALE' else 0)
test_df['Sex'] = test_df['Sex'].apply(lambda x: 1 if x == 'MALE' else 0)

features = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)', 'Sex']
X_train = train_df[features].values
X_test = test_df[features].values

# Etykiety wyjściowe (y) - One-Hot Encoding dla 3 gatunków
one_hot_encoder = OneHotEncoder(sparse_output=False, categories='auto') # konwersja z 1,2,3 na [1,0,0], [0,1,0], [0,0,1]
y_train = one_hot_encoder.fit_transform(train_df[['Species']])
y_test = one_hot_encoder.transform(test_df[['Species']])


# 2. Definicja struktury sieci i parametrów
k = 5  # liczba wejść 
h = 5  # liczba neuronów w warstwie ukrytej
r = 3  # liczba wyjść 

lr = 0.1  # współczynnik uczenia
epochs = 100 # liczba epok

def trening(k,h,r,lr,epochs,isclassify=False,example_penguein=None,is_cmatrix=True):
    # 3. Inicjalizacja wag i biasów
    np.random.seed(42) # Ustawienie ziarna losowości dla powtarzalności wyników
    # Warstwa ukryta (wejście -> warstwa ukryta)
    W1 = np.random.uniform(size=(k, h))
    b1 = np.random.uniform(size=(1, h))

    # Warstwa wyjściowa (warstwa ukryta -> wyjście)
    W2 = np.random.uniform(size=(h, r))
    b2 = np.random.uniform(size=(1, r))

    errors_history = []

    # 4. Trening sieci 
    for i in range(epochs):
        # Propagacja w przód 
        Z1 = np.dot(X_train, W1) + b1
        A1 = sigmoid(Z1)

        Z2 = np.dot(A1, W2) + b2
        A2 = sigmoid(Z2)

        loss = np.mean(np.square(y_train - A2))
        errors_history.append(loss)

        # Propagacja wsteczna

        # Obliczenie gradientu dla warstwy wyjściowej
        error = y_train - A2
        dZ2 = error * sigmoid_derivative(A2)
        
        # Obliczenie gradientu dla warstwy ukrytej
        error_hidden_layer = dZ2.dot(W2.T)
        dZ1 = error_hidden_layer * sigmoid_derivative(A1)

        # Aktualizacja wag i biasów
        W2 += A1.T.dot(dZ2) * lr
        b2 += np.sum(dZ2, axis=0, keepdims=True) * lr

        W1 += X_train.T.dot(dZ1) * lr
        b1 += np.sum(dZ1, axis=0, keepdims=True) * lr

        # if i % (epochs // 10) == 0:
        #     print(f"Epoka: {i}, Błąd (Loss): {loss}")

    # 5. Testowanie sieci
    def predict(X_data, W1, b1, W2, b2):
        Z1 = np.dot(X_data, W1) + b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = sigmoid(Z2)
        return A2

    predictions_raw = predict(X_test, W1, b1, W2, b2)
    # Konwersja predykcji (prawdopodobieństw) na konkretne klasy
    predicted_classes = np.argmax(predictions_raw, axis=1)
    actual_classes = np.argmax(y_test, axis=1)


    # 6. Zapisywanie wyników do pliku
    output_path = 'AI laby/lista 2/wyniki_symulacji_mlp.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f'Struktura sieci: {k}-{h}-{r}\n')
        f.write(f'Liczba epok: {epochs}\n')
        f.write(f'Współczynnik uczenia: {lr}\n\n')
        
        f.write('Błąd średniokwadratowy w kolejnych epokach:\n')
        for i, err in enumerate(errors_history):
            if i % 100 == 0: # Zapisuj co 100-tną epokę, żeby plik nie był zbyt duży
                f.write(f'Epoka {i}: {err}\n')
        
        f.write('\n--- Końcowe Wartości Wag i Biasów ---\n')
        f.write('\nWagi warstwy ukrytej (W1):\n')
        f.write(str(W1))
        f.write('\n\nBias warstwy ukrytej (b1):\n')
        f.write(str(b1))
        f.write('\n\nWagi warstwy wyjściowej (W2):\n')
        f.write(str(W2))
        f.write('\n\nBias warstwy wyjściowej (b2):\n')
        f.write(str(b2))
        
        f.write('\n\n--- Wyniki dla Zbioru Testującego ---\n')
        correct_predictions = np.sum(predicted_classes == actual_classes)
        total_predictions = len(actual_classes)
        accuracy = (correct_predictions / total_predictions) * 100
        f.write(f'Dokładność: {accuracy:.2f}% ({correct_predictions}/{total_predictions} poprawnych predykcji)\n\n')
        
        f.write('Przewidziane -> Rzeczywiste (nazwy gatunków):\n')

        original_labels = {1: 'Chinstrap', 2: 'Adelie', 0: 'Gentoo'}
        index_to_name_map = {i: original_labels.get(label, f"Nieznany {label}") for i, label in enumerate(one_hot_encoder.categories_[0])}
        
        for pred_idx, actual_idx in zip(predicted_classes, actual_classes):
            pred_name = index_to_name_map.get(pred_idx, "Nieznany")
            actual_name = index_to_name_map.get(actual_idx, "Nieznany")
            f.write(f'{pred_name} -> {actual_name}\n')

    print(f"Ostateczna dokładność na zbiorze testowym: {accuracy:.2f}% dla wsp. uczenia: {lr:.2f} , epok: {epochs} oraz liczby neuronów ukrytych: {h}")

    if isclassify == True:
        def classify_penguin(culmen_length, culmen_depth, flipper_length, body_mass, sex, W1, b1, W2, b2, scaler, index_to_name_map):

            sex_numeric = 1 if sex.upper() == 'MALE' else 0
            
            numerical_features = np.array([[culmen_length, culmen_depth, flipper_length, body_mass]])
            
            scaled_numerical_features = scaler.transform(numerical_features)
            
            penguin_features = np.append(scaled_numerical_features, [[sex_numeric]], axis=1)
            
            prediction_raw = predict(penguin_features, W1, b1, W2, b2)
            predicted_index = np.argmax(prediction_raw, axis=1)[0]
            
            predicted_species = index_to_name_map.get(predicted_index, "Nieznany gatunek")
            
            return predicted_species



        scaler = MinMaxScaler()
        scaler.fit(train_df[['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']])

        # Dane przykładowego pingwina do klasyfikacji 
        example_penguin = {
            "culmen_length": 39.1,
            "culmen_depth": 20,
            "flipper_length": 181.0,
            "body_mass": 3750.0,
            "sex": "MALE"
        }

        predicted_species = classify_penguin(
            example_penguin["culmen_length"],
            example_penguin["culmen_depth"],
            example_penguin["flipper_length"],
            example_penguin["body_mass"],
            example_penguin["sex"],
            W1, b1, W2, b2, scaler, index_to_name_map
        )

        print(f"\nPrzykładowy pingwin został sklasyfikowany jako: {predicted_species}")

    if is_cmatrix == True:
        def confusion_matrix(y_true, y_pred, class_names):
    
            num_classes = len(class_names)
            cm = np.zeros((num_classes, num_classes), dtype=int)
            
            for i in range(len(y_true)):
                cm[y_true[i], y_pred[i]] += 1
                
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Przewidziane')
            plt.ylabel('Rzeczywiste')
            plt.title('Macierz pomyłek dla lr: {:.2f}, epok: {}, ukryte neurony: {}'.format(lr, epochs, h))
            plt.show()


        species = {0: 'Gentoo', 1: 'Chinstrap', 2: 'Adelie'}
        class_names = list(one_hot_encoder.categories_[0])
        l = []
        for i in class_names:
            l.append(species[i])
        confusion_matrix(actual_classes, predicted_classes, l)

# for i in np.arange(1,11,1):
#     trening(k,i,r,lr=0.18,epochs=100,is_cmatrix=False)

trening(k,5,r,0.18,epochs=50,is_cmatrix=True)
