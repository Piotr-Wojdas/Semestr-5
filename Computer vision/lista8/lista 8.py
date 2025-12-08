import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np


def add_gaussian_noise(image, variance=0.1):
    """
    Dodaje szum gaussowski do obrazu.
    
    Args:
        image: Tensor obrazu o wymiarach [C, H, W] znormalizowany do [0, 1]
        variance: Wariancja szumu gaussowskiego
    
    Returns:
        Tensor obrazu z dodanym szumem, obcięty do zakresu [0, 1]
    """
    noise = torch.randn_like(image) * (variance ** 0.5)
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0.0, 1.0)


def add_salt_and_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05):
    """
    Dodaje szum typu salt-and-pepper do obrazu.
    
    Args:
        image: Tensor obrazu o wymiarach [C, H, W] znormalizowany do [0, 1]
        salt_prob: Prawdopodobieństwo wystąpienia piksela białego (salt)
        pepper_prob: Prawdopodobieństwo wystąpienia piksela czarnego (pepper)
    
    Returns:
        Tensor obrazu z dodanym szumem
    """
    noisy_image = image.clone()
    
    # Salt (białe piksele)
    salt_mask = torch.rand_like(image) < salt_prob
    noisy_image[salt_mask] = 1.0
    
    # Pepper (czarne piksele)
    pepper_mask = torch.rand_like(image) < pepper_prob
    noisy_image[pepper_mask] = 0.0
    
    return noisy_image

d
class NoisyFashionMNIST(Dataset):
    """
    Dataset zwracający pary obrazów: zaszumiony (input) i czysty (target).
    """
    def __init__(self, root='./data', train=True, noise_type='gaussian', 
                 variance=0.1, salt_prob=0.05, pepper_prob=0.05):
        """
        Args:
            root: Ścieżka do katalogu z danymi
            train: Czy użyć zbioru treningowego (True) czy testowego (False)
            noise_type: Typ szumu - 'gaussian' lub 'salt_and_pepper'
            variance: Wariancja dla szumu gaussowskiego
            salt_prob: Prawdopodobieństwo salt dla szumu salt-and-pepper
            pepper_prob: Prawdopodobieństwo pepper dla szumu salt-and-pepper
        """
        # Transformacja: konwersja do tensora (automatycznie normalizuje do [0, 1])
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # Wczytanie Fashion-MNIST
        self.dataset = torchvision.datasets.FashionMNIST(
            root=root,
            train=train,
            download=True,
            transform=transform
        )
        
        self.noise_type = noise_type
        self.variance = variance
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        clean_image, label = self.dataset[idx]
        
        # Dodanie szumu
        if self.noise_type == 'gaussian':
            noisy_image = add_gaussian_noise(clean_image, self.variance)
        elif self.noise_type == 'salt_and_pepper':
            noisy_image = add_salt_and_pepper_noise(
                clean_image, self.salt_prob, self.pepper_prob
            )
        else:
            raise ValueError(f"Nieznany typ szumu: {self.noise_type}")
        
        return noisy_image, clean_image, label


class ConvAutoencoder(nn.Module):
    """
    Konwolucyjny autoenkoder do odszumiania obrazów.
    """
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Enkoder - kompresja obrazu
        self.encoder = nn.Sequential(
            # Input: 1x28x28
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 32x14x14
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64x7x7
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 128x4x4
            nn.ReLU(True)
        )
        
        # Dekoder - rekonstrukcja obrazu
        self.decoder = nn.Sequential(
            # Input: 128x4x4
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x8x8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),  # 32x15x15
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 1x30x30
            nn.Sigmoid()  # Normalizacja do [0, 1]
        )
        
        # Dodatkowa warstwa do dopasowania wymiarów 30x30 -> 28x28
        self.final_crop = nn.AdaptiveAvgPool2d((28, 28))
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_crop(x)
        return x


def train_autoencoder(model, train_loader, num_epochs=5, learning_rate=0.001, device='cpu'):
    """
    Trenuje autoenkoder do odszumiania obrazów.
    
    Args:
        model: Model autoenkodera
        train_loader: DataLoader z danymi treningowymi
        num_epochs: Liczba epok treningu
        learning_rate: Współczynnik uczenia
        device: Urządzenie ('cpu' lub 'cuda')
    
    Returns:
        Lista strat dla każdej epoki
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_history = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (noisy_imgs, clean_imgs, _) in enumerate(train_loader):
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            
            # Forward pass
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            
            # Backward pass i optymalizacja
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i + 1) % 100 == 0:
                print(f'Epoka [{epoch+1}/{num_epochs}], Krok [{i+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)
        print(f'Epoka [{epoch+1}/{num_epochs}] zakończona. Średnia strata: {epoch_loss:.4f}')
    
    return loss_history


def evaluate_autoencoder(model, test_dataset, num_samples=5, device='cpu'):
    """
    Ewaluacja autoenkodera - wizualizacja wyników odszumiania.
    
    Args:
        model: Wytrenowany model autoenkodera
        test_dataset: Dataset testowy
        num_samples: Liczba próbek do wyświetlenia
        device: Urządzenie ('cpu' lub 'cuda')
    """
    model.eval()
    model.to(device)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 2*num_samples))
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    with torch.no_grad():
        for i in range(num_samples):
            noisy_img, clean_img, label = test_dataset[i]
            
            # Predykcja
            noisy_input = noisy_img.unsqueeze(0).to(device)
            denoised_img = model(noisy_input).cpu().squeeze()
            
            # Konwersja do numpy
            clean_np = clean_img.squeeze().numpy()
            noisy_np = noisy_img.squeeze().numpy()
            denoised_np = denoised_img.squeeze().numpy()
            
            # Czysty obraz (oryginalny)
            axes[i, 0].imshow(clean_np, cmap='gray')
            axes[i, 0].set_title(f'Oryginalny\n{class_names[label]}')
            axes[i, 0].axis('off')
            
            # Zaszumiony obraz
            axes[i, 1].imshow(noisy_np, cmap='gray')
            axes[i, 1].set_title('Zaszumiony')
            axes[i, 1].axis('off')
            
            # Obraz po odszumieniu
            axes[i, 2].imshow(denoised_np, cmap='gray')
            axes[i, 2].set_title('Po odszumieniu')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_loss_history(loss_history):
    """
    Wyświetla wykres historii strat podczas treningu.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
    plt.xlabel('Epoka')
    plt.ylabel('Średnia strata (MSE)')
    plt.title('Historia strat podczas treningu')
    plt.grid(True)
    plt.show()


def visualize_samples(dataset, num_samples=5):
    """
    Wizualizacja próbek z datasetu - porównanie obrazów zaszumionych i czystych.
    """
    fig, axes = plt.subplots(num_samples, 2, figsize=(6, 2*num_samples))
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    for i in range(num_samples):
        noisy_img, clean_img, label = dataset[i]
        
        # Konwersja do numpy i usunięcie wymiaru kanału
        noisy_np = noisy_img.squeeze().numpy()
        clean_np = clean_img.squeeze().numpy()
        
        # Zaszumiony obraz
        axes[i, 0].imshow(noisy_np, cmap='gray')
        axes[i, 0].set_title(f'Zaszumiony - {class_names[label]}')
        axes[i, 0].axis('off')
        
        # Czysty obraz
        axes[i, 1].imshow(clean_np, cmap='gray')
        axes[i, 1].set_title(f'Czysty - {class_names[label]}')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Ustawienie urządzenia
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Używane urządzenie: {device}\n")
    
    # ===== CZĘŚĆ 1: Przygotowanie danych =====
    print("="*60)
    print("CZĘŚĆ 1: Przygotowanie danych")
    print("="*60)
    
    # Przykład 1: Szum gaussowski
    print("\n1. Tworzenie datasetu z szumem gaussowskim...")
    train_dataset_gaussian = NoisyFashionMNIST(
        root='./data',
        train=True,
        noise_type='gaussian',
        variance=0.1
    )
    
    test_dataset_gaussian = NoisyFashionMNIST(
        root='./data',
        train=False,
        noise_type='gaussian',
        variance=0.1
    )
    
    print(f"Rozmiar datasetu treningowego: {len(train_dataset_gaussian)}")
    print(f"Rozmiar datasetu testowego: {len(test_dataset_gaussian)}")
    print("\nWizualizacja próbek z szumem gaussowskim:")
    visualize_samples(train_dataset_gaussian, num_samples=5)
    
    # Przykład 2: Szum salt-and-pepper
    print("\n2. Tworzenie datasetu z szumem salt-and-pepper...")
    train_dataset_sp = NoisyFashionMNIST(
        root='./data',
        train=True,
        noise_type='salt_and_pepper',
        salt_prob=0.05,
        pepper_prob=0.05
    )
    
    test_dataset_sp = NoisyFashionMNIST(
        root='./data',
        train=False,
        noise_type='salt_and_pepper',
        salt_prob=0.05,
        pepper_prob=0.05
    )
    
    print(f"Rozmiar datasetu treningowego: {len(train_dataset_sp)}")
    print(f"Rozmiar datasetu testowego: {len(test_dataset_sp)}")
    print("\nWizualizacja próbek z szumem salt-and-pepper:")
    visualize_samples(train_dataset_sp, num_samples=5)
    
    # ===== CZĘŚĆ 2: Trening autoenkodera dla szumu gaussowskiego =====
    print("\n" + "="*60)
    print("CZĘŚĆ 2: Trening autoenkodera - szum gaussowski")
    print("="*60)
    
    # Tworzenie DataLoader
    train_loader_gaussian = DataLoader(
        train_dataset_gaussian,
        batch_size=128,
        shuffle=True,
        num_workers=0
    )
    
    # Inicjalizacja modelu
    model_gaussian = ConvAutoencoder()
    print(f"\nArchitektura modelu:")
    print(model_gaussian)
    
    # Trening
    print(f"\nRozpoczynanie treningu (5 epok)...")
    loss_history_gaussian = train_autoencoder(
        model=model_gaussian,
        train_loader=train_loader_gaussian,
        num_epochs=5,
        learning_rate=0.001,
        device=device
    )
    
    # Wykres strat
    print("\nWyświetlanie wykresu strat...")
    plot_loss_history(loss_history_gaussian)
    
    # ===== CZĘŚĆ 3: Ewaluacja dla szumu gaussowskiego =====
    print("\n" + "="*60)
    print("CZĘŚĆ 3: Ewaluacja - szum gaussowski")
    print("="*60)
    
    print("\nWyświetlanie wyników odszumiania...")
    evaluate_autoencoder(
        model=model_gaussian,
        test_dataset=test_dataset_gaussian,
        num_samples=8,
        device=device
    )
    
    # ===== CZĘŚĆ 4: Trening autoenkodera dla szumu salt-and-pepper =====
    print("\n" + "="*60)
    print("CZĘŚĆ 4: Trening autoenkodera - szum salt-and-pepper")
    print("="*60)
    
    # Tworzenie DataLoader
    train_loader_sp = DataLoader(
        train_dataset_sp,
        batch_size=128,
        shuffle=True,
        num_workers=0
    )
    
    # Inicjalizacja modelu
    model_sp = ConvAutoencoder()
    
    # Trening
    print(f"\nRozpoczynanie treningu (5 epok)...")
    loss_history_sp = train_autoencoder(
        model=model_sp,
        train_loader=train_loader_sp,
        num_epochs=5,
        learning_rate=0.001,
        device=device
    )
    
    # Wykres strat
    print("\nWyświetlanie wykresu strat...")
    plot_loss_history(loss_history_sp)
    
    # ===== CZĘŚĆ 5: Ewaluacja dla szumu salt-and-pepper =====
    print("\n" + "="*60)
    print("CZĘŚĆ 5: Ewaluacja - szum salt-and-pepper")
    print("="*60)
    
    print("\nWyświetlanie wyników odszumiania...")
    evaluate_autoencoder(
        model=model_sp,
        test_dataset=test_dataset_sp,
        num_samples=8,
        device=device
    )
    
    # Zapisanie modeli
    print("\n" + "="*60)
    print("Zapisywanie wytrenowanych modeli...")
    torch.save(model_gaussian.state_dict(), 'autoencoder_gaussian.pth')
    torch.save(model_sp.state_dict(), 'autoencoder_salt_pepper.pth')
    print("Modele zapisane jako 'autoencoder_gaussian.pth' i 'autoencoder_salt_pepper.pth'")
    print("="*60)
