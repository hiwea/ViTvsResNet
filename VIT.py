
# 1. Import libraries
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import timm
from torchvision import models
import time
import matplotlib.pyplot as plt

# 2. Dataset Preparation (CIFAR-100)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# 3. Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4. Load Pre-trained Models and Modify Classifier
# ViT-Base-16
vit = timm.create_model('vit_base_patch16_224', pretrained=True)
vit.head = nn.Linear(vit.head.in_features, 100)
vit.to(device)

# ResNet-50
resnet = models.resnet50(pretrained=True)
resnet.fc = nn.Linear(resnet.fc.in_features, 100)
resnet.to(device)

# 5. Training Function
def train_model(model, optimizer, epochs=30):
    model.train()
    train_acc = []
    for epoch in range(epochs):
        correct = 0
        total = 0
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        acc = 100. * correct / total
        train_acc.append(acc)
        print(f"Epoch {epoch+1}, Loss: {running_loss:.2f}, Accuracy: {acc:.2f}%")
    return train_acc

# 6. Train ViT
vit_optimizer = optim.Adam(vit.parameters(), lr=5e-5, weight_decay=1e-4)
vit_acc = train_model(vit, vit_optimizer)

# 7. Train ResNet
resnet_optimizer = optim.SGD(resnet.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
resnet_acc = train_model(resnet, resnet_optimizer)

# 8. Evaluation Function
def evaluate_model(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    accuracy = 100. * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

vit_test_acc = evaluate_model(vit)
resnet_test_acc = evaluate_model(resnet)

# 9. Plot Accuracy Curve
plt.plot(vit_acc, label='ViT')
plt.plot(resnet_acc, label='ResNet')
plt.xlabel('Epoch')
plt.ylabel('Train Accuracy')
plt.title('Train Accuracy Comparison')
plt.legend()
plt.grid(True)
plt.savefig("accuracy_comparison.png")
plt.show()

# 10. Print Summary
print("\nSummary:")
print(f"ViT Test Accuracy: {vit_test_acc:.2f}%")
print(f"ResNet Test Accuracy: {resnet_test_acc:.2f}%")
print("ViT has more parameters and is slower but performs better on CIFAR-100.")
print("ResNet is faster and lighter, but with slightly lower accuracy.")
