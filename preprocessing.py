import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_image(file_path):
    """
    Load an image from a file path.

    Args:
        file_path (str): Path to the image file.

    Returns:
        numpy.ndarray: Loaded image as a numpy array.
    """
    return np.array(Image.open(file_path))

def combine_images(image1, image2):
    """
    Combine two images side by side.

    Args:
        image1 (numpy.ndarray): First image array.
        image2 (numpy.ndarray): Second image array.

    Returns:
        numpy.ndarray: Combined image array.
    """
    return np.concatenate([image1, image2], axis=1)

def save_image(image_array, path):
    """
    Save an image array to a specified path.

    Args:
        image_array (numpy.ndarray): Image array to save.
        path (str): Output path to save the image.
    """
    Image.fromarray(image_array).save(path)

def process_images(gt_folder, hazy_folder, output_folder):
    """
    Process and combine ground truth and hazy images.

    Args:
        gt_folder (str): Path to the folder containing ground truth images.
        hazy_folder (str): Path to the folder containing hazy images.
        output_folder (str): Path to the output folder where combined images will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)

    gt_files = sorted(os.listdir(gt_folder))
    hazy_files = sorted(os.listdir(hazy_folder))

    for gt_file, hazy_file in zip(gt_files, hazy_files):
        gt_path = os.path.join(gt_folder, gt_file)
        hazy_path = os.path.join(hazy_folder, hazy_file)

        gt_image = load_image(gt_path)
        hazy_image = load_image(hazy_path)

        combined_image = combine_images(hazy_image, gt_image)
        combined_image_path = os.path.join(output_folder, f"{os.path.splitext(gt_file)[0]}_combined.png")
        save_image(combined_image, combined_image_path)

        if gt_files.index(gt_file) % 1000 == 0:
            display_combined_image(combined_image, combined_image_path)

def display_combined_image(image, path):
    """
    Display the combined image and log the path.

    Args:
        image (numpy.ndarray): Combined image array to display.
        path (str): Path where the combined image is saved.
    """
    plt.imshow(image)
    plt.title("Combined GT and Hazy")
    plt.axis("off")
    plt.show()
    plt.close()
    print(f"Combined image saved as: {path}")

# Paths configuration
gt_folder = "./train/GT"
hazy_folder = "./train/hazy"
output_folder = "./data/train"

# Process the images
process_images(gt_folder, hazy_folder, output_folder)
