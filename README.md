# Neural Network for Iris Flower Classification

## Overview 🌸
This project implements a **Neural Network** using **PyTorch** to classify iris flowers into three species: *Setosa, Versicolor, and Virginica*. The model is trained on the **Iris Dataset**, using a feedforward neural network with two hidden layers.

## Features 🚀
- Implements a custom **PyTorch-based neural network**.
- Uses **CrossEntropyLoss** for multi-class classification.
- **Adam optimizer** for efficient training.
- **Data Preprocessing**: Converts species labels to numeric values.
- **Train/Test Split** for model validation.
- **Loss Curve Visualization** using Matplotlib.
- **Model Saving and Loading** for future inference.

## Prerequisites 🛠
Ensure the following dependencies are installed:

```bash
pip install torch pandas scikit-learn matplotlib
```

## Dataset 📊
The **Iris dataset** consists of 150 samples, each with **four features**:
1. Sepal length
2. Sepal width
3. Petal length
4. Petal width

The target variable contains **three species**:
- `0` → *Iris-setosa*
- `1` → *Iris-versicolor*
- `2` → *Iris-virginica*

## Model Architecture 🏗️
The neural network consists of:
- **Input Layer**: 4 neurons (corresponding to the 4 features)
- **Hidden Layer 1**: 8 neurons (ReLU activation)
- **Hidden Layer 2**: 9 neurons (ReLU activation)
- **Output Layer**: 3 neurons (one for each class)

## Implementation Details 📜

### 1️⃣ Model Definition
The model is defined using PyTorch's `nn.Module`:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
```

### 2️⃣ Data Preprocessing
- Load the dataset using Pandas
- Convert categorical labels into numerical values
- Perform a **train-test split**
```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("/kaggle/input/iris/Iris.csv")
df['Species'] = df['Species'].replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
X = df.drop(['Species', 'Id'], axis=1)
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)
```

### 3️⃣ Convert Data to Tensors
```python
X_train = torch.FloatTensor(X_train.values)
X_test = torch.FloatTensor(X_test.values)
y_train = torch.LongTensor(y_train.values)
y_test = torch.LongTensor(y_test.values)
```

### 4️⃣ Model Training 🎯
```python
model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100
losses = []
for i in range(epochs):
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print(f'Epoch: {i}, Loss: {loss.item()}')
```

### 5️⃣ Loss Curve Visualization 📉
```python
import matplotlib.pyplot as plt
plt.plot(range(epochs), losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()
```

### 6️⃣ Model Evaluation 🏆
```python
with torch.no_grad():
    y_eval = model(X_test)
    loss = criterion(y_eval, y_test)
print(f'Test Loss: {loss.item()}')
```

### 7️⃣ Model Inference & Prediction 🔍
Test the model with **new unseen data**:
```python
new_iris = torch.tensor([3.2, 3.1, 1.2, 0.2])
with torch.no_grad():
    print(model(new_iris))
```

### 8️⃣ Model Saving & Loading 💾
Save the trained model:
```python
torch.save(model.state_dict(), 'iris_model.pth')
```
Load the model later:
```python
new_model = Model()
new_model.load_state_dict(torch.load('iris_model.pth'))
new_model.eval()
```

## Results & Insights 📈
- The model **successfully classifies** iris species with high accuracy.
- Training loss **decreases over epochs**, showing effective learning.
- The model can be fine-tuned by **adjusting hyperparameters** like the number of neurons, learning rate, and epochs.

## Future Improvements 🔮
- Implement **dropout layers** to prevent overfitting.
- Use **batch normalization** for better convergence.
- Train on **larger datasets** for improved generalization.
- Implement **hyperparameter tuning** for optimal performance.

## License 📜
This project is **open-source**. Feel free to modify and improve it!

---
🔍 **Contributions and feedback are welcome!** 🚀
