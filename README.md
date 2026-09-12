# exp 05 : Developing a Recurrent Neural Network Model for Stock Prediction

## Aim

To develop a Recurrent Neural Network (RNN) model using PyTorch to predict stock prices from previous closing price data.

## Algorithm

1. Import the required Python libraries.
2. Load the training and testing stock datasets from `trainset.csv` and `testset.csv`.
3. Select the `Close` column as the stock price.
4. Normalize the training and testing data using `MinMaxScaler`.
5. Create sequences of 60 previous stock prices to predict the next price.
6. Convert the sequences into PyTorch tensors.
7. Create a `DataLoader` with a batch size of 64.
8. Define an RNN model with an input size of 1, hidden size of 50, and output size of 1.
9. Use a linear layer to produce the predicted stock price.
10. Use `MSELoss` as the loss function and Adam optimizer with learning rate 0.001.
11. Train the RNN model for 100 epochs.
12. Predict the stock prices from the test data.
13. Convert the predicted and actual values back to the original price scale.
14. Plot the actual and predicted stock prices and display the final prediction.

## Program

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Load and Preprocess Data
df_train = pd.read_csv('trainset.csv')
df_test = pd.read_csv('testset.csv')

# Use closing prices
train_prices = df_train['Close'].values.reshape(-1, 1)
test_prices = df_test['Close'].values.reshape(-1, 1)

# Normalize the data based on training set only
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_prices)
scaled_test = scaler.transform(test_prices)

# Create sequences
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

seq_length = 60
x_train, y_train = create_sequences(scaled_train, seq_length)
x_test, y_test = create_sequences(scaled_test, seq_length)

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create dataset and dataloader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Step 2: Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

model = RNNModel()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Step 3: Train the Model
num_epochs = 100
train_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_epoch_loss)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')

# Plot training loss
print('Name:                 ')
print('Register Number:     ')

plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

# Step 4: Make Predictions on Test Set
model.eval()

with torch.no_grad():
    predicted = model(x_test_tensor.to(device)).cpu().numpy()
    actual = y_test_tensor.cpu().numpy()

# Inverse transform the predictions and actual values
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(actual)

# Plot the predictions vs actual prices
print('Name:                 ')
print('Register Number:     ')

plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Prediction using RNN')
plt.legend()
plt.show()

print(f'Predicted Price: {predicted_prices[-1]}')
print(f'Actual Price: {actual_prices[-1]}')
```

The notebook creates **1,199 training sequences** and **65 testing sequences**, each using a sequence length of 60.

The RNN model contains **2,701 trainable parameters**.

## Output

<img width="631" height="420" alt="image" src="https://github.com/user-attachments/assets/15bb1fc6-e09a-47c4-87e6-9bde5774258a" />

<img width="1047" height="522" alt="image" src="https://github.com/user-attachments/assets/ce20a04e-f05b-4b48-9d60-7c78ce10057a" />


Final output from the notebook:

```text
Predicted Price: [1105.8026]
Actual Price: [1115.65]
```

The model was trained for 100 epochs, and the loss printed at the end was **0.0003**.

The notebook plots the actual and predicted prices and prints the final predicted and actual prices.

## Result

Thus, the Recurrent Neural Network model was developed successfully using PyTorch for stock price prediction. The final predicted price was **1105.8026**, while the actual price was **1115.65**.
