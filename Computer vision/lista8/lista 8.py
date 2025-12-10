import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np


def add_gaussian_noise(image, variance=0.1):

    noise = torch.randn_like(image) * (variance ** 0.5)
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0.0, 1.0)


def add_salt_and_pepper_noise(image, salt_prob=0.15, pepper_prob=0.15):
    noisy_image = image.clone()
    salt_mask = torch.rand_like(image) < salt_prob
    noisy_image[salt_mask] = 1.0
    pepper_mask = torch.rand_like(image) < pepper_prob
    noisy_image[pepper_mask] = 0.0
    return noisy_image


class NoisyFashionMNIST(Dataset):
    def __init__(self, root='Computer vision/lista 8/data', train=True, noise_type='gaussian', 
                 variance=0.1, salt_prob=0.15, pepper_prob=0.15):
       
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

device = 'cpu' if not torch.cuda.is_available() else 'cuda'

# pętla treningowa
def train_autoencoder(model, train_loader, num_epochs=5, learning_rate=0.001, device=device):
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
            
            
        print(f'Epoka [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)
        print(f'Epoka [{epoch+1}/{num_epochs}] zakończona. Średnia strata: {epoch_loss:.4f}')
    
    return loss_history


def evaluate_autoencoder(model, test_dataset, num_samples=5, device=device, save_path=None):
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
    
    # Zapisz obraz do pliku, jeśli podano ścieżkę
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

def visualize_samples(dataset, num_samples=5):
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
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    # Przygotowanie danych 
    
    # Szum gaussowski
    train_dataset_gaussian = NoisyFashionMNIST(
        root='.Computer vision/lista8/data',
        train=True,
        noise_type='gaussian',
        variance=0.1
    )
    
    test_dataset_gaussian = NoisyFashionMNIST(
        root='.Computer vision/lista8/data',
        train=False,
        noise_type='gaussian',
        variance=0.1
    )
    
    print(f"Rozmiar datasetu treningowego: {len(train_dataset_gaussian)}")
    print(f"Rozmiar datasetu testowego: {len(test_dataset_gaussian)}")
    visualize_samples(train_dataset_gaussian, num_samples=5)
    
    # Przykład 2: Szum salt-and-pepper
    train_dataset_sp = NoisyFashionMNIST(
        root='.Computer vision/lista8/data',
        train=True,
        noise_type='salt_and_pepper',
        salt_prob=0.05,
        pepper_prob=0.05
    )
    
    test_dataset_sp = NoisyFashionMNIST(
        root='.Computer vision/lista8/data',
        train=False,
        noise_type='salt_and_pepper',
        salt_prob=0.15,
        pepper_prob=0.15
    )
    
    visualize_samples(train_dataset_sp, num_samples=5)
    
    # Trening autoenkodera dla szumu gaussowskiego
    
    # DataLoader
    train_loader_gaussian = DataLoader(
        train_dataset_gaussian,
        batch_size=128,
        shuffle=True,
        num_workers=0
    )
    
    # Inicjalizacja modelu
    model_gaussian = ConvAutoencoder()
    
    # Trening
    loss_history_gaussian = train_autoencoder(
        model=model_gaussian,
        train_loader=train_loader_gaussian,
        num_epochs=5,
        learning_rate=0.001,
        device=device
    )
    
    # Ewaluacja dla szumu gaussowskiego    
    evaluate_autoencoder(
        model=model_gaussian,
        test_dataset=test_dataset_gaussian,
        num_samples=8,
        device=device,
        save_path='Computer vision/lista8/odszumianie_gaussian.png'
    )
    
    # Trening autoenkodera dla szumu salt-and-pepper
    
    # DataLoader
    train_loader_sp = DataLoader(
        train_dataset_sp,
        batch_size=128,
        shuffle=True,
        num_workers=0
    )
    
    # Inicjalizacja modelu
    model_sp = ConvAutoencoder()
    
    # Trening
    loss_history_sp = train_autoencoder(
        model=model_sp,
        train_loader=train_loader_sp,
        num_epochs=5,
        learning_rate=0.001,
        device=device
    )
    
    # Ewaluacja dla szumu salt-and-pepper
    evaluate_autoencoder(
        model=model_sp,
        test_dataset=test_dataset_sp,
        num_samples=8,
        device=device,
        save_path='Computer vision/lista8/odszumianie_salt_and_pepper.png'
    )
    
    