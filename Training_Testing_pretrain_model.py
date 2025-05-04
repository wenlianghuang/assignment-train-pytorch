import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTFeatureExtractor

import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast  # For mixed precision training

from Shape_Classifier import *


# Hyperparameters
batch_size = 16  # Reduced batch size
learning_rate = 1e-4  # Lower learning rate
num_epochs = 30  # Reduced number of epochs
#num_classes = 10  # Adjust based on your dataset
num_classes = 3  # Adjust based on your dataset
'''
# Data preparation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
val_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
'''
train_loader, val_loader, test_loader = prepare_dataloaders(
    batch_size=256,  # Reduced batch size
    test_size=0.1, 
    val_size=0.2, 
    num_images_per_class=1000, 
    img_size=(224, 224), 
    noise=True, 
    dataset_dir="dataset"
)
# Model setup
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
model.classifier = nn.Linear(model.classifier.in_features, num_classes)  # Replace the classification head

# Freeze all layers except the classifier
for param in model.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
scaler = torch.amp.GradScaler('cuda')  # For mixed precision training

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        
        with torch.amp.autocast("cuda"):  # Corrected mixed precision training
            outputs = model(images).logits
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        '''
        outputs = model(images)
        loss = criterion(outputs, labels)
        '''
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images,labels = images.to(device), labels.to(device)
            
            with torch.amp.autocast("cuda"):  # Corrected mixed precision validation
                outputs = model(images).logits
                loss = criterion(outputs, labels)
            val_loss += loss.item()
            #_,predicted = outputs.max(1)
    val_loss /= len(val_loader)
    #train_accuracy = 100. * correct / total
    #print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")
    print(f"Epoch [{epoch+1}/{num_epochs}],Training Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
# Testing phase
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images).logits
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

test_accuracy = 100. * correct / total
print(f"Testing Accuracy: {test_accuracy:.2f}%")

# Example usage of predict function
# from PIL import Image
# image = Image.open('path_to_image.jpg')
# prediction = predict(image)
# print(f"Predicted class: {prediction}")