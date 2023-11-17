import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
# Import configurations, e.g., epoch_loss from a config file
from config import *


# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Define the layers of the model
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(50 * 5 * 5, 500)
        self.fc2 = nn.Linear(500, 26)

    def forward(self, x):
        # Forward pass of the model
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 50 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Load testing data
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_loader, test_loader


# Function to train the model
def train_model(cnn_model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = cnn_model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Print loss every 10 steps
            if i % 10 == 0:
                if epoch_loss == 'average':
                    average_loss = total_loss / len(train_loader)
                    print('Epoch [{}/{}], Average Loss: {:.4f}'.format(epoch + 1, num_epochs, average_loss))
                elif epoch_loss == 'last_batch':
                    print('Epoch [{}/{}], Last Batch Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
        print('=' * 50)  # Print a dividing line after each epoch


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

# Main function to orchestrate the training and testing
def main():
    cnn_model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
    train_loader, test_loader = get_data_loaders()
    train_model(cnn_model, train_loader, criterion, optimizer, num_epochs=10)
    torch.save(cnn_model.state_dict(), model_dir)

    # Load the model and evaluate on test data
    cnn_model.load_state_dict(torch.load(model_dir))
    test_predictions, accuracy = get_predictions_and_accuracy(cnn_model, test_loader)
    # print("All predicted labels on the test set:")
    print(test_predictions)
    print('Accuracy on test set: {:.2f}%'.format(accuracy))


if __name__ == "__main__":
    main()

    # Load the model to predict a single image
    # cnn_model.load_state_dict(torch.load('../data/cnn_model.pth'))
    # image_path = '../data/test/1/3.jpg'
    # predicted_class = predict_single_image(cnn_model, image_path)
    # print("The predicted class for the image is: {}".format(predicted_class))