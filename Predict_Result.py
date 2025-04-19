from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from Training_Testing_Moel import SimpleCNN
def predict_single_sample(model, image_path, actual_label, device='cpu', class_names=['Circle', 'Square', 'Triangle']):
    """
    Predict a single sample and display the result.

    Args:
        model (nn.Module): The trained model.
        image_path (str): Path to the image file.
        actual_label (int): The actual label of the image.
        device (str): Device to perform the prediction on ('cpu' or 'cuda').
        class_names (list): List of class names.
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = image.resize((128, 128))  # Resize to match model input size
    image_tensor = torch.tensor(np.array(image).transpose(2, 0, 1), dtype=torch.float32) / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension

    # Perform prediction
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_label = torch.max(output, 1)

    # Display the result
    predicted_label = predicted_label.item()
    print(f"Actual: {class_names[actual_label]}, Predicted: {class_names[predicted_label]}")

    # Visualize the image
    plt.imshow(image)
    plt.title(f"Actual: {class_names[actual_label]}\nPredicted: {class_names[predicted_label]}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    model = SimpleCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.to(device)
    model.eval()
    # Example usage
    sample_image_path = "dataset/circle/circle_0.png"
    actual_label = 0  # Actual label of the image (0 for Circle, 1 for Square, 2 for Triangle)
    #device = 'cpu'  # Change to 'cuda' if using a GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predict_single_sample(model, sample_image_path, actual_label, device)