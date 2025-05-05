import os
import random
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import shutil
# Function to generate a single image with a geometric shape
def generate_shape_image(shape, size, rotation, img_size=(128, 128), noise=False):
    """
    Generate an image with a single geometric shape.

    Args:
        shape (str): The shape to draw ('circle', 'square', 'triangle').
        size (int): The size of the shape.
        rotation (int): The rotation angle of the shape (degrees).
        img_size (tuple): The size of the image (width, height).
        noise (bool): Whether to add Gaussian noise to the image.

    Returns:
        PIL.Image: The generated image.
    """
    img = Image.new('RGB', img_size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    #center = (img_size[0] // 2, img_size[1] // 2)

    # Randomly select the center position of the shape within the image
    center_x = random.randint(size // 2, img_size[0] - size // 2)
    center_y = random.randint(size // 2, img_size[1] - size // 2)
    center = (center_x, center_y)
    
    # Randomly select the center position of the shape within the image
    shape_color = tuple(random.randint(0, 255) for _ in range(3))  # Random color for the shape
    if shape == 'circle':
        radius = size // 2
        draw.ellipse([center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius], fill=shape_color)
    elif shape == 'square':
        half_size = size // 2
        draw.rectangle([center[0] - half_size, center[1] - half_size, center[0] + half_size, center[1] + half_size], fill=shape_color)
    elif shape == 'triangle':
        half_size = size // 2
        points = [
            (center[0], center[1] - half_size),
            (center[0] - half_size, center[1] + half_size),
            (center[0] + half_size, center[1] + half_size)
        ]
        draw.polygon(points, fill=shape_color)

    if noise:
        img = add_gaussian_noise(img)

    return img.rotate(rotation)

# Function to add Gaussian noise to an image
def add_gaussian_noise(image, mean=0, std=25):
    """
    Add Gaussian noise to an image.

    Args:
        image (PIL.Image): The input image.
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.

    Returns:
        PIL.Image: The image with added noise.
    """
    np_image = np.array(image)
    noise = np.random.normal(mean, std, np_image.shape).astype(np.int16)
    noisy_image = np.clip(np_image + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

# Custom PyTorch Dataset
class ShapeDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        Args:
            data (list): List of image file paths.
            labels (list): List of labels corresponding to the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        # Dynamically load the image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Convert image to tensor
        #image = torch.tensor(np.array(image).transpose(2, 0, 1), dtype=torch.float32) / 255.0
        return image, label

# Function to generate the dataset and save it to a folder
def generate_dataset(output_dir="dataset", num_images_per_class=1000, img_size=(128, 128), noise=True):
    """
    Generate a dataset of images with geometric shapes and save them to a folder.

    Args:
        output_dir (str): Directory to save the dataset.
        num_images_per_class (int): Number of images per shape class.
        img_size (tuple): Size of the images (width, height).
        noise (bool): Whether to add Gaussian noise to the images.

    Returns:
        list, list: List of image file paths and corresponding labels.
    """
    shapes = ['circle', 'square', 'triangle']
    data = []
    labels = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for label, shape in enumerate(shapes):
        shape_dir = output_dir / shape
        shape_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_images_per_class):
            size = random.randint(20, 80)  # Random size
            rotation = random.randint(0, 360)  # Random rotation
            img = generate_shape_image(shape, size, rotation, img_size, noise)

            # Save image to file
            img_path = shape_dir / f"{shape}_{i}.png"
            img.save(img_path)

            data.append(str(img_path))  # Store file path as string
            labels.append(label)

    return data, labels

# Function to load the dataset from a folder
def load_dataset_from_folder(folder="dataset"):
    """
    Load a dataset of image file paths and labels from a folder.

    Args:
        folder (str): Directory containing the dataset.

    Returns:
        list, list: List of image file paths and corresponding labels.
    """
    folder = Path(folder)
    shapes = ['circle', 'square', 'triangle']
    data = []
    labels = []

    for label, shape in enumerate(shapes):
        shape_dir = folder / shape
        if not shape_dir.exists():
            raise ValueError(f"Shape folder {shape_dir} does not exist. Please generate the dataset first.")

        for img_path in shape_dir.glob("*.png"):
            data.append(str(img_path))  # Store file path as string
            labels.append(label)

    return data, labels

# Main function to prepare the dataset and DataLoader
def prepare_dataloaders(batch_size=32, test_size=0.2, val_size=0.1, num_images_per_class=10000, img_size=(128, 128), noise=False, dataset_dir="dataset"):
    """
    Prepare PyTorch DataLoaders for training, validation, and testing.

    Args:
        batch_size (int): Batch size for the DataLoader. Default is 32.
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        val_size (float): Proportion of the dataset to include in the validation split. Default is 0.1.
        num_images_per_class (int): Number of images to generate per shape class. Default is 1000.
        img_size (tuple): Size of the images (width, height). Default is (128, 128).
        noise (bool): Whether to add Gaussian noise to the generated images. Default is False.
        dataset_dir (str): Directory to save/load the dataset. Default is "dataset".

    Returns:
        tuple: A tuple containing three PyTorch DataLoaders:
            - train_loader (DataLoader): DataLoader for the training set.
            - val_loader (DataLoader): DataLoader for the validation set.
            - test_loader (DataLoader): DataLoader for the testing set.
    """
    # Define data augmentation transforms
    train_transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(), #Randomly flips the image horizontally with a 50% probability, simulating mirrored versions of the data.
        transforms.RandomVerticalFlip(), # Randomly flips the image vertically with a 50% probability, adding further variability
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), #Randomly adjusts the brightness, contrast, saturation, and hue of the image within specified ranges, making the model more robust to lighting and color variations.
        transforms.ToTensor() #Converts the image from a PIL image or NumPy array into a PyTorch tensor and scales pixel values to the range [0, 1].
    ])
    
    val_test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Generate the dataset no matter if it exists or not
    if not Path(dataset_dir).exists():
        print(f"Dataset folder '{dataset_dir}' not found. Generating dataset...")
        generate_dataset(output_dir=dataset_dir, num_images_per_class=num_images_per_class, img_size=img_size, noise=noise)
    else:
        shutil.rmtree(dataset_dir)  # Remove existing dataset folder
        generate_dataset(output_dir=dataset_dir, num_images_per_class=num_images_per_class, img_size=img_size, noise=noise)
    
    data, labels = load_dataset_from_folder(folder=dataset_dir)

    # Split the dataset into train, validation, and test sets
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_size, stratify=labels) # stratify ensures that the split maintains the same proportion of classes
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=val_size, stratify=train_labels)

    # Create PyTorch Datasets
    train_dataset = ShapeDataset(train_data, train_labels, train_transform)
    val_dataset = ShapeDataset(val_data, val_labels, val_test_transform)
    test_dataset = ShapeDataset(test_data, test_labels, val_test_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
    

# Example usage
if __name__ == "__main__":
    train_loader, val_loader, test_loader = prepare_dataloaders(dataset_dir="dataset",noise=True)
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of testing batches: {len(test_loader)}")