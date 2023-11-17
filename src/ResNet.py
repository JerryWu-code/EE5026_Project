import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
import torch.nn.functional as F
from PIL import Image
from config import *
from data_loader import *


# Define the ResNet model
class ResNet(nn.Module):
    def __init__(self, num_classes=26):
        super(ResNet, self).__init__()
        # Load a pre-trained ResNet18 model with new weights parameter
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.model = models.resnet18(weights=weights)
        # Modify the input layer to accept grayscale images
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Modify the output layer to match the number of classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)


#####
# Function to prepare data loaders
def get_data_loaders():
    # Training data transformations
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=1),
        ###
        # Data Augmentation
        transforms.ColorJitter(brightness=0.2),
        ###
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load training data
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

    # Load testing data
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True)

    return train_loader, test_loader


# Function to train the model
def train_model(res_model, train_loader, criterion, optimizer, scheduler, num_epochs):
    loss_history = []
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = res_model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            loss_history.append(loss.item())
            loss.backward()
            optimizer.step()

            # Print loss every 10 steps
            if i % 10 == 0:
                if epoch_loss == 'average':
                    average_loss = total_loss / len(train_loader)
                    log1 = 'Epoch [{}/{}], Average Loss: {:.4f}'.format(epoch + 1, num_epochs, average_loss)

                elif epoch_loss == 'last_batch':
                    log1 = 'Epoch [{}/{}], Last Batch Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item())

                with open(resnet18_log_dir, 'a') as f:
                    f.write(log1 + '\n')
                print(log1)

        # update the learning rate
        scheduler.step()

        with open(resnet18_log_dir, 'a') as f:
            f.write('=' * 50 + '\n')
        print('=' * 50)  # Print a dividing line after each epoch

    return loss_history


# Function to calculate predictions and accuracy
def get_predictions_and_accuracy(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    all_predictions = []

    with torch.no_grad():  # Gradient calculation is not required
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.tolist())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return all_predictions, accuracy


# Function to preprocess a single image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image


# Function to predict the class of a single image
def predict_single_image(model, image_path):
    # Preprocess the image
    input_image = preprocess_image(image_path)

    # Perform prediction
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        output = model(input_image)

    # Interpret the model output
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

    return predicted_class


#####

def main():
    resnet_model = ResNet()
    criterion = nn.CrossEntropyLoss()
    if resnet18_norm_gradient:
        torch.nn.utils.clip_grad_norm_(resnet_model.parameters(), max_norm=1.0)
    if resnet18_L2_norm:
        optimizer = optim.Adam(resnet_model.parameters(), lr=learning_rate_, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(resnet_model.parameters(), lr=learning_rate_)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma_)
    train_loader, test_loader = get_data_loaders()
    loss_history = train_model(resnet_model, train_loader, criterion, optimizer, scheduler, num_epochs=num_epochs)

    # save the loss history and model
    save_loss_history_to_csv(loss_history, resnet18_loss_dir)
    torch.save(resnet_model.state_dict(), resnet18_model_dir)

    # Load the model and evaluate on test data
    resnet_model.load_state_dict(torch.load(resnet18_model_dir))
    test_predictions, accuracy = get_predictions_and_accuracy(resnet_model, test_loader)
    # print(test_predictions)

    log2 = 'Accuracy on test set: {:.2f}%'.format(accuracy)
    with open(resnet18_log_dir, 'a') as f:
        f.write(log2)
    print(log2)


if __name__ == "__main__":
    set_random_seed(seed)
    with open(resnet18_log_dir, 'w') as f:
        f.write('Model: ResNet18\n\n')
    main()
