# DL-Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional neural network (CNN) classification model for the given dataset.

## THEORY
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28×28 pixels. The task is to classify these images into their respective digit categories. CNNs are particularly well-suited for image classification tasks as they can automatically learn spatial hierarchies of features through convolutional layers, pooling layers, and fully connected layers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS

STEP 1:
Import the required libraries such as PyTorch, Torchvision, NumPy, Matplotlib, Seaborn, and Scikit-Learn metrics.

STEP 2:
Load the MNIST handwritten digit dataset and apply preprocessing transformations such as tensor conversion and normalization.

STEP 3:
Create DataLoaders for the training and testing datasets to enable batch processing.

STEP 4:
Design a Convolutional Neural Network (CNN) consisting of convolution layers, activation functions, pooling layers, and fully connected layers.

STEP 5:
Define the loss function (CrossEntropyLoss) and optimizer (Adam).

STEP 6:
Train the CNN model using the training dataset and update weights using backpropagation.

STEP 7:
Evaluate the trained model using the testing dataset.

STEP 8:
Compute the classification accuracy, confusion matrix, and classification report.

STEP 9:
Visualize the confusion matrix to analyze model performance.

STEP 10:
Predict the class label of a sample handwritten digit image and compare it with the actual label.

## PROGRAM

### Name: Arunsamy D

### Register Number: 212224240016

```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.fc1 = nn.Linear(
            32 * 7 * 7,
            128
        )

        self.fc2 = nn.Linear(
            128,
            10
        )


    def forward(self, x):
        x = self.pool(
            torch.relu(
                self.conv1(x)
            )
        )

        x = self.pool(
            torch.relu(
                self.conv2(x)
            )
        )

        x = x.view(
            x.size(0),
            -1
        )

        x = torch.relu(
            self.fc1(x)
        )

        x = self.fc2(x)

        return x

# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier()

if torch.cuda.is_available():
    model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# Train the Model
def train_model(model, train_loader, num_epochs=3):
    print('Name: Arunsamy D')
    print('Register Number: 212224240016')
    
    model.train()

    for epoch in range(num_epochs):

        running_loss = 0.0

        for images, labels in train_loader:

            if torch.cuda.is_available():
                images = images.to(device)
                labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

## OUTPUT

### Dataset Information
<img width="393" height="391" alt="image" src="https://github.com/user-attachments/assets/3cba9d81-c995-411d-b182-1da9020d50fe" />

### CNN Model Summary
<img width="955" height="424" alt="image" src="https://github.com/user-attachments/assets/85fffd98-b49b-406a-981f-ee98b93141b7" />

### Training Loss per Epoch

<img width="967" height="340" alt="image" src="https://github.com/user-attachments/assets/336fe4ac-9baf-4e2b-9bb1-f4017d51a379" />

### Confusion Matrix
<img width="707" height="597" alt="download" src="https://github.com/user-attachments/assets/be16e4f8-b282-4b4e-9592-3b54647d1cad" />

### Classification Report
<img width="1035" height="426" alt="image" src="https://github.com/user-attachments/assets/8a085381-422e-4afa-9d36-6aa0d48b1678" />

### Sample Image Prediction
<img width="1102" height="632" alt="image" src="https://github.com/user-attachments/assets/d42dbfe3-6d26-4a3e-b6fd-0a08d9387241" />


## RESULT

Thus, a Convolutional Neural Network (CNN) was successfully developed and trained using the MNIST handwritten digit dataset. The model was evaluated using test accuracy, confusion matrix, and classification report, and it successfully classified handwritten digit images with high accuracy.
