import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from Shape_Classifier import *
# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        #self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  #Input: 3x128x128 Output: 32x64x64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  #Input: 3x128x128 Output: 32x64x64
            nn.ReLU(),
            nn.Dropout(0.6)
        )
        #self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  #Input: 32x64x64 Output: 64x32x132
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  #Input: 32x64x64 Output: 64x32x32
            nn.ReLU(),
            nn.Dropout(0.6)
        )
        #self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) #Input: 64x32x32 Output: 128x16x16 
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), #Input: 64x32x32 Output: 128x16x16
            nn.ReLU(),
            nn.Dropout(0.6)
        )
        #self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1) #Input: 128x16x16 Output: 256x8x8 
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), #Input: 128x16x16 Output: 256x8x8
            nn.ReLU(),
            nn.Dropout(0.6)
        )
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1) #Input: 256x8x8 Output: 512x4x4 

        # Calculate the flattened size after all convolutions and pooling
        self.flattened_size = 512 * 4 * 4  # Output after conv5 and pooling (512x4x4)

        self.fc1 = nn.Linear(self.flattened_size, 1024)  # Fully connected layer
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 3)  # Output: 3 classes (circle, square, triangle)
        
        
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # Downsample: 32x64x64
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # Downsample: 64x32x32
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)  # Downsample: 128x16x16
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)  # Downsample: 256x8x8
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 2)  # Downsample: 512x4x4

        x = x.view(x.size(0), -1)  # Flatten: 512*4*4
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Training function
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

# Testing function
def test_model(model, test_loader, device='cpu'):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    
    sample_images = []
    sample_labels = []
    sample_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        # Collect samples for visualization
        if len(sample_images) < 5:
            sample_images.extend(images.cpu().numpy())
            sample_labels.extend(labels.cpu().numpy())
            sample_preds.extend(preds.cpu().numpy())
            
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}")
    visualize_samples(sample_images[:5], sample_labels[:5], sample_preds[:5])

    # Confusion matrix
    '''
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Circle', 'Square', 'Triangle'], yticklabels=['Circle', 'Square', 'Triangle'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    '''
def visualize_samples(images, labels, preds,class_names = ['Circle', 'Square', 'Triangle']):
    num_samples = len(images)
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        img = images[i].transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.title(f"Actual: {class_names[labels[i]]}\nPredicted: {class_names[preds[i]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Plot training and validation loss
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(0, 2)  # Set y-axis limit to not exceed 2
    plt.title('Training and Validation Loss')
    plt.legend()
    #plt.show()
    plt.savefig('loss_plot.png')

def save_model(model,path='model.pth'):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
    
# Main function to train and test the model
if __name__ == "__main__":
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Prepare data loaders
    train_loader, val_loader, test_loader = prepare_dataloaders(
        batch_size=32, 
        test_size=0.2, 
        val_size=0.1, 
        num_images_per_class=2000, 
        img_size=(128, 128), 
        noise=False, 
        dataset_dir="dataset"
    )

    # Initialize model, loss function, and optimizer
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=0.0001)

    # Train the model
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30, device=device)

    # Plot losses
    plot_losses(train_losses, val_losses)
    save_model(model, path='model.pth')
    # Test the model
    test_model(model,test_loader=test_loader, device=device)