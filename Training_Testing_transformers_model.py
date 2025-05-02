import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from Shape_Classifier import *

# Define a simple Vision Transformer model
class VisionTransformer(nn.Module):
    def __init__(self, img_size=128, patch_size=16, num_classes=3, dim=128, depth=6, heads=8, mlp_dim=256, dropout=0.1):
        super(VisionTransformer, self).__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size  # 3 channels (RGB)
        self.dim = dim

        # Patch embedding
        self.patch_embedding = nn.Linear(self.patch_dim, dim)

        # Positional embeddings
        self.positional_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))

        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, heads, mlp_dim, dropout),
            num_layers=depth
        )

        # MLP head for classification
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        # Split image into patches
        batch_size, channels, height, width = x.shape
        patch_size = int((self.patch_dim // channels) ** 0.5)
        x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        x = x.contiguous().view(batch_size, -1, self.patch_dim)  # Flatten patches

        # Patch embedding
        x = self.patch_embedding(x)

        # Add class token and positional embeddings
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.positional_embedding

        # Transformer encoder
        x = self.transformer(x)

        # Classification head
        cls_output = x[:, 0]  # Extract class token
        return self.mlp_head(cls_output)

# ...existing code...

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    model.to(device)
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(val_loader))
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    return train_losses, val_losses

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(0, 2)  # Set y-axis limit to not exceed 2
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_plot.png')

def save_model(model, path='model.pth'):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def test_model(model, test_loader, device='cpu'):
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy and confusion matrix
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1', 'Class 2'], yticklabels=['Class 0', 'Class 1', 'Class 2'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")
# Main function to train and test the model
if __name__ == "__main__":
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare data loaders
    train_loader, val_loader, test_loader = prepare_dataloaders(
        batch_size=64, 
        test_size=0.2, 
        val_size=0.1, 
        num_images_per_class=1000, 
        img_size=(128, 128), 
        noise=False, 
        dataset_dir="dataset"
    )

    # Initialize model, loss function, and optimizer
    model = VisionTransformer(img_size=128, patch_size=16, num_classes=3, dim=128, depth=6, heads=8, mlp_dim=256, dropout=0.1)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

    # Train the model
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30, device=device)

    # Plot losses
    plot_losses(train_losses, val_losses)
    save_model(model, path='vit_from_scratch.pth')

    # Test the model
    test_model(model, test_loader=test_loader, device=device)