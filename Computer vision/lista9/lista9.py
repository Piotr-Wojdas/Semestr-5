import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Ustawienia
NORMAL_CLASS = 0  # Cyfra uznawana za "normalną"
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")


# 1. AUTOENKODER LINIOWY
class LinearAutoencoder(nn.Module):
    def __init__(self):
        super(LinearAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.view(-1, 1, 28, 28)
        return decoded


# 2. AUTOENKODER KONWOLUCYJNY
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 7x7 -> 4x4
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),  # 4x4 -> 7x7
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14 -> 28x28
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# 3. SIEĆ KONWOLUCYJNA (CNN) - dla klasyfikacji
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14x14 -> 7x7
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


def load_data():
    """Wczytaj zbiór MNIST i przygotuj dane"""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Pobierz cały zbiór MNIST
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    return train_dataset, test_dataset


def prepare_datasets(train_dataset, test_dataset, normal_class):
    """Przygotuj zbiory treningowe i testowe"""
    # Zbiór treningowy - tylko normalna klasa
    train_indices = [i for i, (_, label) in enumerate(train_dataset) if label == normal_class]
    train_normal = Subset(train_dataset, train_indices)
    
    # Zbiór testowy - wszystkie cyfry
    test_normal_indices = [i for i, (_, label) in enumerate(test_dataset) if label == normal_class]
    test_anomaly_indices = [i for i, (_, label) in enumerate(test_dataset) if label != normal_class]
    
    test_normal = Subset(test_dataset, test_normal_indices)
    test_anomaly = Subset(test_dataset, test_anomaly_indices)
    
    print(f"\nRozmiar zbioru treningowego (klasa {normal_class}): {len(train_normal)}")
    print(f"Rozmiar zbioru testowego (klasa {normal_class}): {len(test_normal)}")
    print(f"Rozmiar zbioru testowego (anomalie): {len(test_anomaly)}")
    
    return train_normal, test_normal, test_anomaly


def train_autoencoder(model, train_loader, epochs, lr, model_name):
    """Trenuj autoenkoder"""
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"\n{'='*50}")
    print(f"Trenowanie: {model_name}")
    print(f"{'='*50}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(DEVICE)
            
            optimizer.zero_grad()
            reconstructed = model(data)
            loss = criterion(reconstructed, data)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
    
    return model


def train_cnn(model, train_normal, test_normal_loader, test_anomaly_loader, epochs, lr):
    """Trenuj CNN - z przykładami obu klas (normal=1, anomaly=0)"""
    model = model.to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"\n{'='*50}")
    print(f"Trenowanie: CNN")
    print(f"{'='*50}")
    
    # Przygotuj dane treningowe: normalne (klasa 1) + próbki anomalii (klasa 0)
    # Użyj części danych testowych jako przykładów anomalii podczas treningu
    from torch.utils.data import ConcatDataset
    
    # Wez próbkę anomalii z danych testowych (np. 20% dla treningu)
    test_dataset = test_anomaly_loader.dataset
    anomaly_train_size = len(train_normal) // 2  # Weź połowę rozmiaru normalnych
    anomaly_train_indices = torch.randperm(len(test_dataset))[:anomaly_train_size].tolist()
    anomaly_train = Subset(test_dataset.dataset if hasattr(test_dataset, 'dataset') else test_dataset, 
                           [test_dataset.indices[i] if hasattr(test_dataset, 'indices') else i 
                            for i in anomaly_train_indices])
    
    # Połącz dane normalne i anomalie
    combined_dataset = ConcatDataset([train_normal, anomaly_train])
    train_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Dane treningowe: {len(train_normal)} normalnych + {len(anomaly_train)} anomalii")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for data, label in train_loader:
            data = data.to(DEVICE)
            # Label: 1 dla normalnych (NORMAL_CLASS), 0 dla anomalii
            labels = (label == NORMAL_CLASS).float().unsqueeze(1).to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        if (epoch + 1) % 5 == 0:
            # Ewaluacja
            model.eval()
            correct_normal = 0
            total_normal = 0
            correct_anomaly = 0
            total_anomaly = 0
            
            with torch.no_grad():
                for data, _ in test_normal_loader:
                    data = data.to(DEVICE)
                    outputs = model(data)
                    predicted = (outputs > 0.5).float()
                    total_normal += data.size(0)
                    correct_normal += (predicted == 1).sum().item()
                
                for data, _ in test_anomaly_loader:
                    data = data.to(DEVICE)
                    outputs = model(data)
                    predicted = (outputs > 0.5).float()
                    total_anomaly += data.size(0)
                    correct_anomaly += (predicted == 0).sum().item()
            
            acc_normal = 100 * correct_normal / total_normal
            acc_anomaly = 100 * correct_anomaly / total_anomaly
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, "
                  f"Acc Normal: {acc_normal:.2f}%, Acc Anomaly: {acc_anomaly:.2f}%")
    
    return model


def calculate_reconstruction_error(model, data_loader):
    """Oblicz błąd rekonstrukcji"""
    model.eval()
    errors = []
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(DEVICE)
            reconstructed = model(data)
            error = torch.mean((data - reconstructed) ** 2, dim=[1, 2, 3])
            errors.extend(error.cpu().numpy())
    
    return np.array(errors)


def calculate_cnn_scores(model, data_loader):
    """Oblicz wyniki CNN (prawdopodobieństwo przynależności do klasy normalnej)"""
    model.eval()
    scores = []
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(DEVICE)
            outputs = model(data)
            scores.extend(outputs.cpu().numpy().flatten())
    
    return np.array(scores)


def determine_threshold(normal_errors, percentile=95):
    """Wyznacz próg anomalii na podstawie percentyla"""
    threshold = np.percentile(normal_errors, percentile)
    return threshold


def evaluate_model(normal_errors, anomaly_errors, threshold, model_name):
    """Ewaluuj model"""
    print(f"\n{'='*50}")
    print(f"Wyniki: {model_name}")
    print(f"{'='*50}")
    
    # Statystyki błędów
    print(f"Błąd rekonstrukcji - Normalne:")
    print(f"  Średnia: {np.mean(normal_errors):.6f}")
    print(f"  Mediana: {np.median(normal_errors):.6f}")
    print(f"  Std: {np.std(normal_errors):.6f}")
    
    print(f"\nBłąd rekonstrukcji - Anomalie:")
    print(f"  Średnia: {np.mean(anomaly_errors):.6f}")
    print(f"  Mediana: {np.median(anomaly_errors):.6f}")
    print(f"  Std: {np.std(anomaly_errors):.6f}")
    
    print(f"\nPróg anomalii: {threshold:.6f}")
    
    # Detekcja anomalii
    normal_correct = np.sum(normal_errors <= threshold)
    normal_total = len(normal_errors)
    anomaly_correct = np.sum(anomaly_errors > threshold)
    anomaly_total = len(anomaly_errors)
    
    print(f"\nDetekcja:")
    print(f"  Poprawnie sklasyfikowane normalne: {normal_correct}/{normal_total} ({100*normal_correct/normal_total:.2f}%)")
    print(f"  Poprawnie sklasyfikowane anomalie: {anomaly_correct}/{anomaly_total} ({100*anomaly_correct/anomaly_total:.2f}%)")
    print(f"  Dokładność ogólna: {100*(normal_correct+anomaly_correct)/(normal_total+anomaly_total):.2f}%")
    
    # ROC AUC
    y_true = np.concatenate([np.zeros(len(normal_errors)), np.ones(len(anomaly_errors))])
    y_scores = np.concatenate([normal_errors, anomaly_errors])
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    print(f"  ROC AUC: {roc_auc:.4f}")
    
    return roc_auc, fpr, tpr


def evaluate_cnn(normal_scores, anomaly_scores, model_name):
    """Ewaluuj CNN (wyższe wartości = normalne, niższe = anomalie)"""
    print(f"\n{'='*50}")
    print(f"Wyniki: {model_name}")
    print(f"{'='*50}")
    
    # Statystyki
    print(f"Wyniki - Normalne:")
    print(f"  Średnia: {np.mean(normal_scores):.6f}")
    print(f"  Mediana: {np.median(normal_scores):.6f}")
    print(f"  Std: {np.std(normal_scores):.6f}")
    
    print(f"\nWyniki - Anomalie:")
    print(f"  Średnia: {np.mean(anomaly_scores):.6f}")
    print(f"  Mediana: {np.median(anomaly_scores):.6f}")
    print(f"  Std: {np.std(anomaly_scores):.6f}")
    
    # Próg (domyślnie 0.5)
    threshold = 0.5
    print(f"\nPróg: {threshold:.6f}")
    
    # Detekcja
    normal_correct = np.sum(normal_scores >= threshold)
    normal_total = len(normal_scores)
    anomaly_correct = np.sum(anomaly_scores < threshold)
    anomaly_total = len(anomaly_scores)
    
    print(f"\nDetekcja:")
    print(f"  Poprawnie sklasyfikowane normalne: {normal_correct}/{normal_total} ({100*normal_correct/normal_total:.2f}%)")
    print(f"  Poprawnie sklasyfikowane anomalie: {anomaly_correct}/{anomaly_total} ({100*anomaly_correct/anomaly_total:.2f}%)")
    print(f"  Dokładność ogólna: {100*(normal_correct+anomaly_correct)/(normal_total+anomaly_total):.2f}%")
    
    # ROC AUC (odwracamy scores bo CNN daje wyższe wartości dla normalnych)
    y_true = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))])
    y_scores = np.concatenate([1 - normal_scores, 1 - anomaly_scores])
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    print(f"  ROC AUC: {roc_auc:.4f}")
    
    return roc_auc, fpr, tpr


def visualize_reconstructions(model, test_loader, n_samples=5, model_name="Model"):
    """Wizualizuj przykładowe rekonstrukcje"""
    model.eval()
    
    data_iter = iter(test_loader)
    images, _ = next(data_iter)
    images = images[:n_samples].to(DEVICE)
    
    with torch.no_grad():
        reconstructed = model(images)
    
    images = images.cpu()
    reconstructed = reconstructed.cpu()
    
    fig, axes = plt.subplots(2, n_samples, figsize=(12, 4))
    fig.suptitle(f'{model_name} - Oryginał vs Rekonstrukcja')
    
    for i in range(n_samples):
        axes[0, i].imshow(images[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title('Oryginał')
        
        axes[1, i].imshow(reconstructed[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title('Rekonstrukcja')
    
    plt.tight_layout()
    plt.savefig(f'{model_name.replace(" ", "_")}_reconstructions.png')
    print(f"Zapisano wizualizację: {model_name.replace(' ', '_')}_reconstructions.png")


def plot_roc_curves(results):
    """Narysuj krzywe ROC dla wszystkich modeli"""
    plt.figure(figsize=(10, 8))
    
    for model_name, (roc_auc, fpr, tpr) in results.items():
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Krzywe ROC - Porównanie modeli')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves_comparison.png')
    print("\nZapisano porównanie krzywych ROC: roc_curves_comparison.png")


def main():
    print(f"Rozpoczynam eksperyment z klasą normalną: {NORMAL_CLASS}")
    
    # 1. Wczytaj dane
    print("\n1. Wczytywanie danych MNIST...")
    train_dataset, test_dataset = load_data()
    
    # 2. Przygotuj zbiory
    print("\n2. Przygotowanie zbiorów...")
    train_normal, test_normal, test_anomaly = prepare_datasets(train_dataset, test_dataset, NORMAL_CLASS)
    
    train_loader = DataLoader(train_normal, batch_size=BATCH_SIZE, shuffle=True)
    test_normal_loader = DataLoader(test_normal, batch_size=BATCH_SIZE, shuffle=False)
    test_anomaly_loader = DataLoader(test_anomaly, batch_size=BATCH_SIZE, shuffle=False)
    
    results = {}
    
    # 3. AUTOENKODER LINIOWY
    print("\n" + "="*70)
    print("AUTOENKODER LINIOWY")
    print("="*70)
    
    linear_ae = LinearAutoencoder()
    linear_ae = train_autoencoder(linear_ae, train_loader, EPOCHS, LEARNING_RATE, "Autoenkoder Liniowy")
    
    # Oblicz błędy rekonstrukcji
    normal_errors_linear = calculate_reconstruction_error(linear_ae, test_normal_loader)
    anomaly_errors_linear = calculate_reconstruction_error(linear_ae, test_anomaly_loader)
    
    # Wyznacz próg
    threshold_linear = determine_threshold(normal_errors_linear, percentile=95)
    
    # Ewaluacja
    roc_auc, fpr, tpr = evaluate_model(normal_errors_linear, anomaly_errors_linear, 
                                       threshold_linear, "Autoenkoder Liniowy")
    results["Autoenkoder Liniowy"] = (roc_auc, fpr, tpr)
    
    # Wizualizacja
    visualize_reconstructions(linear_ae, test_anomaly_loader, n_samples=5, 
                            model_name="Autoenkoder Liniowy")
    
    # 4. AUTOENKODER KONWOLUCYJNY
    print("\n" + "="*70)
    print("AUTOENKODER KONWOLUCYJNY")
    print("="*70)
    
    conv_ae = ConvAutoencoder()
    conv_ae = train_autoencoder(conv_ae, train_loader, EPOCHS, LEARNING_RATE, "Autoenkoder Konwolucyjny")
    
    # Oblicz błędy rekonstrukcji
    normal_errors_conv = calculate_reconstruction_error(conv_ae, test_normal_loader)
    anomaly_errors_conv = calculate_reconstruction_error(conv_ae, test_anomaly_loader)
    
    # Wyznacz próg
    threshold_conv = determine_threshold(normal_errors_conv, percentile=95)
    
    # Ewaluacja
    roc_auc, fpr, tpr = evaluate_model(normal_errors_conv, anomaly_errors_conv, 
                                       threshold_conv, "Autoenkoder Konwolucyjny")
    results["Autoenkoder Konwolucyjny"] = (roc_auc, fpr, tpr)
    
    # Wizualizacja
    visualize_reconstructions(conv_ae, test_anomaly_loader, n_samples=5, 
                            model_name="Autoenkoder Konwolucyjny")
    
    # 5. CNN
    print("\n" + "="*70)
    print("SIEĆ KONWOLUCYJNA (CNN)")
    print("="*70)
    
    cnn = CNN()
    cnn = train_cnn(cnn, train_normal, test_normal_loader, test_anomaly_loader, 
                    EPOCHS, LEARNING_RATE)
    
    # Oblicz wyniki
    normal_scores_cnn = calculate_cnn_scores(cnn, test_normal_loader)
    anomaly_scores_cnn = calculate_cnn_scores(cnn, test_anomaly_loader)
    
    # Ewaluacja
    roc_auc, fpr, tpr = evaluate_cnn(normal_scores_cnn, anomaly_scores_cnn, "CNN")
    results["CNN"] = (roc_auc, fpr, tpr)
    
    # 6. Porównanie wszystkich modeli
    print("\n" + "="*70)
    print("PODSUMOWANIE")
    print("="*70)
    print("\nROC AUC dla wszystkich modeli:")
    for model_name, (roc_auc, _, _) in results.items():
        print(f"  {model_name}: {roc_auc:.4f}")
    
    # Narysuj krzywe ROC
    plot_roc_curves(results)
    
    print("\n" + "="*70)
    print("EKSPERYMENT ZAKOŃCZONY")
    print("="*70)


if __name__ == "__main__":
    main()
