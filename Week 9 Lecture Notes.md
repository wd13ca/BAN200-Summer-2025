# Week 9 Lecture Notes

## Review: Linear and Logistic Regression

Before we move on to neural networks, let’s take a step back and review the models we've built so far — **linear regression** and **logistic regression**.

These two models illustrate a powerful general recipe for building machine learning systems:

### The Four Ingredients of a Model

1. **Data** — What are the inputs and outputs?
2. **Architecture** — What kind of model are we using? (e.g., linear function, softmax layer)
3. **Loss Function** — How do we measure how wrong the predictions are?
4. **Training Algorithm** — How do we improve the model? (usually with gradient descent)


### Linear Regression

**Goal**: Predict a single number (e.g., a house price)

#### 1. Data  

- Input: vector `x` of features (e.g., size, location)  
- Output: scalar `y` (e.g., price)

#### 2. Architecture  

A simple linear model:

```
ŷ = w · x + b
```

Where:
- `w` is a weight vector  
- `b` is a scalar bias  
- `ŷ` is the predicted value

#### 3. Loss Function  

We use **Mean Squared Error (MSE)**:

```
L = (ŷ - y)²
```

#### 4. Training Algorithm  

Use gradient descent to minimize the loss by updating `w` and `b`.

### Logistic Regression

**Goal**: Predict a **class**, not a number

#### 1. Data  

- Input: vector `x` of features (e.g., tokenized message)  
- Output: vector `y` of class labels

#### 2. Architecture  

A linear model plus an activation function.

**Binary Classification:**

```
z = w · x + b  
ŷ = sigmoid(z)
```

**Multiclass Classification:**

```
z = w · x + b  
ŷ = softmax(z)
```

Where:
- `w` is now a **matrix** (one row of weights per class)  
- `ŷ` is a **vector of probabilities**, one per class
- `softmax` or `sigmoid` is the activation function

#### 3. Loss Function  

We use **Cross-Entropy Loss**:

```
L = -log(ŷ_y)
```

Where `ŷ_y` is the predicted probability for the true class `y`.

#### 4. Training Algorithm  

Use gradient descent to minimize the loss by updating `w` and `b`.

## From Logistic Regression to Neural Networks

So far, we’ve seen how a **logistic regression model** can take an input vector `x`, apply a linear transformation, and then use **softmax** to produce a prediction.

But what if the data isn’t linearly separable? What if we want the model to learn **interactions between features**, or **nonlinear patterns**?

To solve that, we introduce the next step in our journey:

**Multi-Layer Networks (Neural Networks)**

### Key Idea

Add **layers** of computation between the input and the output.

These layers apply **nonlinear functions** to the data — giving the model more flexibility and power to learn complex patterns.

### Neural Network Architecture

The simplest kind of neural network is called a **feedforward network** or **multi-layer perceptron (MLP)**.

It looks like this:

```
input x
   ↓
Linear: z₁ = W₁ · x + b₁
   ↓
Nonlinearity: h = ReLU(z₁)
   ↓
Linear: z₂ = W₂ · h + b₂
   ↓
Softmax: ŷ = softmax(z₂)
```

Each step transforms the data a bit more, allowing the model to build up increasingly abstract representations.

### Hidden Layers

A **hidden layer** is any layer that comes between the input and the output.

- It has its own weights and biases (`W₁`, `b₁`)
- It applies a **nonlinear activation function** like **ReLU**, **tanh**, or **sigmoid**
- It produces a new internal representation `h`, which becomes the input to the next layer

### ReLU Activation Function

One of the most popular activation functions is **ReLU (Rectified Linear Unit)**:

```
ReLU(z) = max(0, z)
```

Why ReLU?

- It introduces **nonlinearity**, so the model can learn more than just lines or planes
- It's **simple to compute**
- It helps with **gradient flow** in deeper networks

### Final Layer: Softmax

As before, we use **softmax** at the output to turn raw scores into probabilities:

```
ŷ = softmax(W₂ · h + b₂)
```

Where:

- `h` is the hidden layer output  
- `W₂`, `b₂` are the weights and biases of the final layer  
- `ŷ` is a probability distribution over classes

### Updated 4 Ingredients

We’re still following the same recipe — just with more expressive power:

1. **Data** — still the same input-output pairs  
2. **Architecture** — now has **multiple layers** and **nonlinearities**  
3. **Loss Function** — still **cross-entropy** for classification  
4. **Training Algorithm** — still **gradient descent**, now applied to **every layer**

We’ll need to compute gradients for all weights using a technique called **backpropagation** — a generalization of the chain rule from calculus.

### Summary

- A **neural network** is a stack of linear layers and nonlinear activations  
- Each layer transforms the data into a new space  
- Nonlinear activations (like ReLU) give the model the power to learn complex patterns  
- At the output, we use softmax to predict class probabilities  
- Training is done using gradient descent — just like logistic regression

## Backpropagation: How Neural Networks Learn

Now that we've introduced multi-layer networks, the big question is:

**How do we train all these layers?**

In logistic regression, we computed the gradient of the loss with respect to the weights and biases using the chain rule. In a neural network, we do the same thing — just across **multiple layers**.

This process is called **backpropagation**.

### What Is Backpropagation?

**Backpropagation** is a method for computing the gradient of the loss with respect to **every parameter in the network**.

It’s based on two ideas:

1. **Chain Rule** (from calculus):  
   If a function is made of multiple parts, we can compute its derivative by multiplying the derivatives of each part.

2. **Reusing Intermediate Results**:  
   Instead of recomputing everything from scratch, we compute gradients layer by layer, starting from the output and working backward.

That’s why it’s called **back**propagation — we go **backwards** through the network.

### The 4 Ingredients (Again)

Let’s revisit the model-building recipe:

1. **Data** — inputs `x`, targets `y`
2. **Architecture** — multiple layers with weights and activations
3. **Loss Function** — measures how wrong our prediction is
4. **Training Algorithm** — this is where backpropagation lives

Backpropagation is how we implement gradient descent for **multi-layer networks**.

### Backprop: Step by Step

For a simple neural network like this:

```
x → Linear → ReLU → Linear → Softmax → Loss
```

Backpropagation proceeds as follows:

1. **Forward pass**:  
   Compute predictions and loss just like normal.

2. **Backward pass**:  
   - Start with the gradient of the loss (∂L/∂ŷ)
   - Use the **chain rule** to compute gradients for the last layer  
     (e.g., ∂L/∂W₂ and ∂L/∂b₂)
   - Pass the error **backward** through the ReLU  
   - Continue to compute gradients for the first layer  
     (e.g., ∂L/∂W₁ and ∂L/∂b₁)

3. **Update weights**:  
   Use gradient descent to adjust all weights and biases.

### Why It Works

Each layer of the network is just a function — usually linear followed by a nonlinearity. Because we know how to differentiate each part, we can use the chain rule to compute the full gradient.

This allows us to **learn all the weights in the network**, no matter how many layers there are.

### Summary

- Backpropagation is a generalization of gradient descent for multi-layer models
- It uses the **chain rule** to compute gradients from output to input
- It allows us to train all layers of a neural network — not just the last one
- It’s the foundation of modern deep learning

You don’t need to memorize the math — the key idea is that we can compute gradients for each layer automatically and use them to improve the model.

## What Is PyTorch?

**PyTorch** is a popular open-source deep learning library developed by **Meta (Facebook AI Research)**. It provides flexible tools for:

- Working with **tensors** (multi-dimensional arrays)
- Building **neural networks**
- Running models on **GPUs** for fast computation
- Automatically computing **gradients** using a system called **autograd**

PyTorch is used by researchers, engineers, and companies around the world — from small prototypes to large-scale production systems.

### Why Are We Using It Now?

Building models from scratch helped us:

- Understand how predictions, loss, gradients, and updates actually work
- See how matrix math underpins all learning

But in practice, almost no one writes models completely from scratch.  
Instead, people use **high-level libraries** like PyTorch and Google's TensorFlow.

These tools save time, reduce bugs, and make it easy to scale up to larger models.

> From here on, we'll use PyTorch to build and train our neural networks — but now you’ll understand exactly what’s going on under the hood.

## SMS Spam Collection

We’re returning to the **SMS Spam Collection** — a dataset we’ve used a few times already to explore text classification.

Each row in the dataset contains:
- A **label**: either `"spam"` or `"ham"` (not spam)
- A **message**: the text content of an SMS

Previously, we used:
- **Naive Bayes** to classify messages using word counts
- **Logistic Regression** (from scratch) with TF-IDF vectors

This time, we’ll build a **multi-layer neural network** to perform the same task — using **PyTorch** to manage tensors, layers, gradients, and optimization.

> Same dataset, but now a more powerful model — and a much more scalable framework.

## Preparing the Data

Before we can train a neural network, we need to turn our SMS messages into numeric input vectors that PyTorch can understand. In this section, we:

1. **Load the dataset**
2. **Split into training and test sets**
3. **Tokenize the messages**
4. **Compute TF‑IDF scores**
5. **Convert messages into PyTorch tensors**
6. **Wrap everything in a Dataset and DataLoader**

### Step 1: Load the Dataset

We use `pandas` to read the SMS Spam Collection file from a URL. Each row contains a label (`"spam"` or `"ham"`) and a message.

```python
import pandas as pd
df = pd.read_csv(
    "https://wd13ca.github.io/BAN200-Summer-2025/SMSSpamCollection.txt",
    sep="\t", header=None, names=["label", "message"]
)
```

### Step 2: Train-Test Split

We randomly split the dataset: 80% for training, 20% for testing.

```python
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=13)
```

### Step 3: Tokenize the Text

We define a simple tokenizer that:
- Lowercases the text
- Removes stop words (like "the", "and", etc.)
- Extracts word tokens using regular expressions

```python
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def tokenize(text):
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if t not in ENGLISH_STOP_WORDS]
```

### Step 4: Compute IDF Scores

We compute the **inverse document frequency (IDF)** of each word in the training set. Words that appear in many messages get lower weights; rare words get higher weights.

```python
import math

N = len(train_df)
doc_freq = {}
for msg in train_df["message"]:
    for tok in set(tokenize(msg)):
        doc_freq[tok] = doc_freq.get(tok, 0) + 1

idf = {tok: math.log(N / df) for tok, df in doc_freq.items()}
```

### Step 5: Vectorize Each Message

Finally, we create a tokenizer that converts a message into a dense PyTorch tensor. Each element of the vector corresponds to a word in the vocabulary, weighted by its TF‑IDF score.

```python
vocab = [word for word, _ in sorted(doc_freq.items(), key=lambda item: item[1], reverse=True)]
word2idx = {w: i for i, w in enumerate(vocab)}

def vectorize(message):
    vec = torch.zeros(len(vocab))
    for tok in tokenize(message):
        if tok in idf:
            vec[word2idx[tok]] += idf[tok]
    return vec
```

### Step 6: Create a Dataset and DataLoader

To use PyTorch effectively, we wrap our data in a `Dataset` class and feed it to a `DataLoader`. This handles batching and shuffling automatically.


```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

label2index = {'spam': 0, 'ham': 1}

class SMSDataset(Dataset):
    def __init__(self, df):
        self.x = torch.stack([vectorize(m) for m in df["message"]])
        self.y = torch.tensor([label2index[lbl] for lbl in df["label"]])
    def __len__(self): 
        return len(self.y)
    def __getitem__(self, idx): 
        return self.x[idx], self.y[idx]

batch_size = 64
train_loader = DataLoader(SMSDataset(train_df), batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(SMSDataset(test_df),  batch_size=batch_size)
```

> At this point, we have a working PyTorch pipeline: raw text → TF‑IDF vector → PyTorch tensor → mini-batches for training.

## Defining the Model

Now that our data is ready, let’s define the neural network we’ll use to classify SMS messages.

We’ll build a **two-layer feedforward neural network**:

1. A **linear layer** that maps from the input size to a hidden dimension  
2. A **ReLU activation** to introduce nonlinearity  
3. A second **linear layer** that maps from the hidden dimension to 2 output units (spam or ham)

### Why 2 Outputs?

Because we encoded the labels as `[1.0, 0.0]` for spam and `[0.0, 1.0]` for ham, the model must output two numbers — one for each class. We'll later apply **softmax** or **cross-entropy loss** to turn those into probabilities.

### PyTorch Model Definition

We’ll use `torch.nn.Sequential` to stack the layers.

```python
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)
```

### Instantiate the Model

We define:
- `input_dim`: the size of the TF‑IDF vector (i.e. vocab size)
- `hidden_dim`: number of hidden units (you can tune this)
- `output_dim`: 2 (for spam and ham)

```python
input_dim = len(vocab)
hidden_dim = 100
output_dim = 2

model = TwoLayerNet(input_dim, hidden_dim, output_dim)
```

> This model has learnable weights in both layers and can learn complex patterns in the input — much more powerful than logistic regression.

Next, we’ll train this model using gradient descent.

## Training the Model

To train the model, we need two key components:

1. A **loss function** to measure how wrong the predictions are  
2. An **optimizer** to update the model’s weights using gradients

We’ll use:
- `nn.CrossEntropyLoss()` — combines softmax + log-loss
- `torch.optim.Adam` — a good default optimizer

```python
import torch.nn as nn
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

Now we train the model over multiple epochs:

```python
epochs = 10

for epoch in range(epochs):
    # --- Training phase ---
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(yb)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += len(yb)

    avg_train_loss = total_loss / total
    train_acc = correct / total

    # --- Evaluation phase ---
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            test_loss += loss.item() * len(yb)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += len(yb)

    avg_test_loss = test_loss / total
    test_acc = correct / total

    print(f"Epoch {epoch+1}: "
          f"Train Loss = {avg_train_loss:.4f}, Acc = {train_acc:.2%} | "
          f"Test Loss = {avg_test_loss:.4f}, Acc = {test_acc:.2%}")
```

After training, we’ll evaluate the model’s accuracy on the test set.

## Overfitting and Early Stopping

As we build more powerful models — especially neural networks with multiple layers — we also increase the risk of **overfitting**.

### What Is Overfitting?

**Overfitting** happens when a model learns to perform very well on the **training data**, but fails to generalize to **new, unseen data**. In other words, the model starts to memorize the training set instead of learning useful patterns.

### Why Does It Happen?

Overfitting is more likely when:
- The model has **many parameters** (e.g., deep or wide networks)
- The dataset is **small** or **noisy**
- Training runs for **too many epochs**

The model becomes so flexible that it can "explain" the training data perfectly — including the noise — but it performs poorly on the test set.

You’ll often see this pattern:

- **Training loss keeps decreasing**
- **Validation (test) loss starts increasing**

This is a clear sign of overfitting.

### One Solution: Early Stopping

**Early stopping** is a simple and effective way to avoid overfitting.

Here’s how it works:
- During training, monitor performance on the **validation set**
- If validation loss stops improving for several epochs in a row, **stop training early**
- Keep the model from the epoch with the **lowest validation loss**

Early stopping helps prevent the model from "going too far" and starting to memorize the training data.

> Overfitting is a sign that your model is too powerful for your data — regularization, more data, or simpler models can help, but early stopping is often the easiest place to start.


## Evaluate the Model

### Predict Function

Let's create a `predict()` function that takes an unlabeled message and returns a predicted label:

```python
def predict(message):
    """
    Predict the class label for a raw input message (string).
    - message: the input message (e.g., "Free entry now!!!")

    Returns:
        predicted_label: the class label with highest probability
    """
    model.eval()
    with torch.no_grad():
        vec = vectorize(message).unsqueeze(0)  # add batch dimension
        logits = model(vec)                   # raw scores (1 x num_classes)
        predicted_class = logits.argmax(dim=1).item()
    return list(label2index.keys())[list(label2index.values()).index(predicted_class)]

```

### Evaluate Model Accuracy on the Test Set

Now that we have a working `predict()` function, we’ll apply it to every message in the test set.

Then we’ll compare the predicted labels to the actual labels and calculate the model’s accuracy.

```python
# Predict all messages in the test set
predictions = []

for _, row in test_df.iterrows():
    message = row["message"]
    prediction = predict(message)
    predictions.append(prediction)

# Actual labels
actual = test_df["label"].tolist()

# Compute accuracy
correct = sum([pred == truth for pred, truth in zip(predictions, actual)])
accuracy = correct / len(test_df)

print(f"Accuracy: {accuracy:.2%}")
```

### Confusion Matrix

Now let's take a look at the confusion matrix:

```python
from sklearn.metrics import confusion_matrix

# Generate confusion matrix
cm = confusion_matrix(actual, predictions, labels=["spam", "ham"])

# Display as a readable table
print("Confusion Matrix:")
print(f"               Predicted")
print(f"             | spam | ham ")
print(f"Actual spam  |  {cm[0][0]:4} | {cm[0][1]:4}")
print(f"Actual ham   |  {cm[1][0]:4} | {cm[1][1]:4}")
```

### Precision, Recall, and F1-Score

Here are the precision, recall, and F1-scores:

```python
from sklearn.metrics import classification_report

print(classification_report(actual, predictions, target_names=["spam", "ham"]))
```

### Error Analysis

Let’s look at some misclassified messages — where the model's prediction didn't match the true label.

This helps us understand:

- Where the model is confused
- Whether certain types of spam are being missed
- If it’s too aggressive (labeling ham as spam)

```python
# Show the first 10 misclassified messages
for i in range(len(test_df)):
    if predictions[i] != actual[i]:
        print(f"\n--- Misclassified Message ---")
        print(f"Actual:    {actual[i]}")
        print(f"Predicted: {predictions[i]}")
        print(f"Message:   {test_df.iloc[i]['message']}")
```

## Exercises

Try these exercises to deepen your understanding of neural network architecture and PyTorch:

1. Change the Number of Hidden Units
2. Add an Extra Layer
3. Remove the Hidden Layer
4. Try a New Dataset

