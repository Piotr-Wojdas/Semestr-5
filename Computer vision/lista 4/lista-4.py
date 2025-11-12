import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

def salt_and_pepper_noise(image, amount):
    """
    Adds salt and pepper noise to an image.
    
    :param image: The input image.
    :param amount: The amount of noise (0.0 to 1.0).
    :return: The noisy image.
    """
    noisy_image = np.copy(image)
    # Salt
    num_salt = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1], :] = (255, 255, 255)

    # Pepper
    num_pepper = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1], :] = (0, 0, 0)
    
    return noisy_image

def gaussian_noise(image, sigma):
    """
    Adds Gaussian noise to an image.
    
    :param image: The input image.
    :param sigma: The standard deviation of the noise.
    :return: The noisy image.
    """
    row, col, ch = image.shape
    mean = 0
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image.astype(np.float32) + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)


# Load the image
image_path = r'c:\Users\RODO\Desktop\Semestr 5\Computer vision\lista 1\a.jfif'
original_image = cv2.imread(image_path)

if original_image is None:
    print(f"Error: Could not load image from {image_path}")
else:
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Create the figure and subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(left=0.25, bottom=0.35)

    ax1.set_title('Oryginalny obraz')
    ax1.imshow(original_image_rgb)
    ax1.axis('off')

    ax2.set_title('Zaszumiony obraz')
    im_noisy = ax2.imshow(original_image_rgb)
    ax2.axis('off')

    ax3.set_title('Odszumiony obraz')
    im_denoised = ax3.imshow(original_image_rgb)
    ax3.axis('off')

    # --- Create UI Elements ---

    # Radio buttons for noise type
    ax_noise_radio = plt.axes([0.05, 0.7, 0.15, 0.15])
    radio_noise = RadioButtons(ax_noise_radio, ('Sól i pieprz', 'Gaussowski'))

    # Radio buttons for filter type
    ax_filter_radio = plt.axes([0.05, 0.5, 0.15, 0.15])
    radio_filter = RadioButtons(ax_filter_radio, ('Gaussowski', 'Medianowy'))

    # Sliders
    ax_sp_intensity = plt.axes([0.25, 0.20, 0.65, 0.03])
    slider_sp_intensity = Slider(ax_sp_intensity, 'Intensywność (Sól i pieprz)', 0.0, 1.0, valinit=0.05)

    ax_gauss_sigma = plt.axes([0.25, 0.15, 0.65, 0.03])
    slider_gauss_sigma = Slider(ax_gauss_sigma, 'Sigma (Gauss)', 0, 100, valinit=20)

    ax_gauss_kernel = plt.axes([0.25, 0.10, 0.65, 0.03])
    slider_gauss_kernel = Slider(ax_gauss_kernel, 'Rozmiar jądra (Gauss)', 1, 31, valinit=5, valstep=2)

    ax_median_kernel = plt.axes([0.25, 0.05, 0.65, 0.03])
    slider_median_kernel = Slider(ax_median_kernel, 'Rozmiar jądra (Medianowy)', 1, 31, valinit=5, valstep=2)

    def update(val):
        # --- Apply Noise ---
        noise_type = radio_noise.value_selected
        if noise_type == 'Sól i pieprz':
            intensity = slider_sp_intensity.val
            noisy_image = salt_and_pepper_noise(original_image_rgb, intensity)
        else: # Gaussian
            sigma = slider_gauss_sigma.val
            noisy_image = gaussian_noise(original_image_rgb, sigma)
        im_noisy.set_data(noisy_image)

        # --- Apply Filter ---
        filter_type = radio_filter.value_selected
        if filter_type == 'Gaussowski':
            kernel_size = int(slider_gauss_kernel.val)
            denoised_image = cv2.GaussianBlur(noisy_image, (kernel_size, kernel_size), 0)
        else: # Median
            kernel_size = int(slider_median_kernel.val)
            # Median blur kernel must be odd
            if kernel_size % 2 == 0:
                kernel_size += 1
            denoised_image = cv2.medianBlur(noisy_image, kernel_size)
        
        im_denoised.set_data(denoised_image)
        fig.canvas.draw_idle()

    # Attach update function to all widgets
    slider_sp_intensity.on_changed(update)
    slider_gauss_sigma.on_changed(update)
    slider_gauss_kernel.on_changed(update)
    slider_median_kernel.on_changed(update)
    radio_noise.on_clicked(update)
    radio_filter.on_clicked(update)

    # Initial call to display images
    update(None)

    plt.show()
