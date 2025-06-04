# Week 6 Lecture Notes

This week, we will continue our introduction of neural networks. Today, we will be building a logistic regression on the SMS Spam Collection using a neural network approach. This will set the stage for the more advanced models we will be working with after study week.

## Multiclass Logistic Regression

So far, we’ve looked at **linear regression**, where the model predicts a **single number**. Now let’s extend this idea to a classification task, where the goal is to predict **which category** something belongs to.

### From Regression to Classification

In classification, instead of predicting a single continuous value, we want to predict one of several **classes**. 

For example, we might want to classify a customer email as:

- `0` = Billing  
- `1` = Sales  
- `2` = Tech Support

### Outputs as a Vector

Instead of outputting a single number `ŷ`, the model now outputs a **vector of values** — one for each class.

For example, the model might output a vector like this:

```
ŷ = [0.1, 0.7, 0.2]
```

We can interpret this as:

- 10% chance it's Billing  
- 70% chance it's Sales  
- 20% chance it's Tech Support  

And we would classify this example as **Sales**, since it has the highest score.

### Model Equation

Our model equation is now:

```
ŷ = softmax(w · x + b)
```

Let’s break it down:

- `x` is the **input vector** (with `n` features)
- `w` is the **weight matrix**, with shape `k × n`  
  (one row of weights for each of the `k` classes)
- `b` is the **bias vector**, with one bias per class
- The result `w · x + b` is a **vector of raw scores** (called **logits**)
- We apply a function called **softmax** to turn those scores into **probabilities**

### Softmax: From Scores to Probabilities

Softmax is a type of **activation function** — a function applied at the output of a model to shape or interpret the result in some useful way.

The softmax function takes in a vector of raw scores and returns a vector of probabilities that sum to 1. It’s defined as:

```
softmax(zᵢ) = exp(zᵢ) / Σⱼ exp(zⱼ)
```

Where:
- `zᵢ` is the raw score for class `i` (the `i`th element of the logits vector)
- `exp(zᵢ)` makes all the scores positive and amplifies larger ones
- The denominator ensures the outputs sum to 1

> This lets us interpret the outputs as a **probability distribution** over the classes.

### Why Is `w` a Matrix?

In linear regression, we had:

```
ŷ = w · x + b
```

Here, `w` was a vector (1D), and `ŷ` was a single number.

But in classification, we want **one output per class**, so we need **a different set of weights for each class**. That means `w` becomes a **matrix**:

- Each **row** of `w` corresponds to one class
- The dot product between that row and the input `x` gives a **score** for that class

In effect, it’s like running **multiple linear regressions in parallel** — one for each class — and then using softmax to pick the most likely one.

### Summary

Multiclass logistic regression is a natural extension of linear regression:

- We go from a **single output** to a **vector of outputs**
- We go from a **weight vector** to a **weight matrix**
- We apply a **nonlinear activation function** (softmax) to interpret the outputs as probabilities

## Deriving the Gradient: Multiclass Logistic Regression

We’ll now derive the gradient for a **softmax classifier** trained with **cross-entropy loss**.

### Model Output

The model computes:

```
z = w · x + b         # raw scores (logits)
ŷ = softmax(z)        # predicted probabilities
```

Where:
- `x` is the input vector (length `n`)
- `w` is a weight matrix with shape `k × n` (k = number of classes)
- `b` is a bias vector (length `k`)
- `ŷ` is a vector of class probabilities (length `k`)

The softmax function for class `i` is:

```
ŷᵢ = exp(zᵢ) / sum_j exp(zⱼ)
```

### Loss Function: Cross-Entropy

If the true class label is `y` (an integer from `0` to `k-1`), the loss is:

```
L = -log(ŷ_y)        # negative log of the predicted probability for the true class
```

### Goal

We want to compute the gradients:

- ∂L/∂wᵢⱼ — how the loss changes with respect to each weight
- ∂L/∂bᵢ  — how the loss changes with respect to each bias

### Gradient of the Loss

We apply the chain rule. The derivative of the loss with respect to the logit `zᵢ` is:

```
∂L/∂zᵢ = ŷᵢ - 1   if i == y
∂L/∂zᵢ = ŷᵢ       otherwise
```

This tells us that:

- For the **correct class**, we subtract 1 from the predicted probability
- For all other classes, the gradient is just the predicted probability

### Final Gradients

Now that we have ∂L/∂zᵢ, we compute the gradients with respect to the weights and biases:

```
∂L/∂wᵢⱼ = (ŷᵢ - 1[y = i]) * xⱼ
∂L/∂bᵢ  = (ŷᵢ - 1[y = i])
```

Where `1[y = i]` is 1 if `i` is the correct class, otherwise 0.

### Summary

- The **error** for each class is:  
  ```
  errorᵢ = ŷᵢ - 1[y = i]
  ```
- We multiply that error by each input feature to get the gradient:
  ```
  dwᵢⱼ = errorᵢ * xⱼ
  dbᵢ  = errorᵢ
  ```

This gives us the gradients we need to perform gradient descent on a multiclass softmax classifier.

## SMS Spam Collection

To see how logistic regressions work, let's try building one. We'll use the same dataset we used in Week 4. 

### Dataset, Tokenizer, and Vectorizer

First, we'll download the data and split it into test and train datasets.

```python
# Download the dataset
!wget https://wd13ca.github.io/BAN200-Summer-2025/SMSSpamCollection.txt

# Load it into a pandas DataFrame
import pandas as pd

df = pd.read_csv("SMSSpamCollection.txt", sep="\t", header=None, names=["label", "message"])

from sklearn.model_selection import train_test_split

# Split the dataset: 80% training, 20% testing
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Show the number of messages in each set
print("Training messages:", len(train_df))
print("Test messages:", len(test_df))
```

Next, we'll recreate our tokenizer.

```python
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def tokenize(text):
    lowercase_text = text.lower()
    tokens = re.findall(r'\b\w+\b', lowercase_text)
    return [t for t in tokens if t not in ENGLISH_STOP_WORDS]
```

We'll also need a vectorizer. Let's use the one from Week 3. 

```python
from math import log

N = len(train_df) # total number of documents

doc_freq = {} # document frequency for each word
for _, row in train_df.iterrows():
  tokens = tokenize(row["message"])
  unique_tokens = set(tokens) # only count once per document
  for token in unique_tokens:
      doc_freq[token] = doc_freq.get(token, 0) + 1

idf_dict = {}
for token, df in doc_freq.items():
    idf_dict[token] = log(N / df)

def vectorize_tf_idf(tokens):
    """
    Takes a list of tokens and an IDF dictionary,
    returns a dictionary representing the TF-IDF vector.
    """
    tf = {}
    for token in tokens:
        if token in tf:
            tf[token] += 1
        else:
            tf[token] = 1

    tf_idf = {}
    for token, freq in tf.items():
        if token in idf_dict:
            tf_idf[token] = freq * idf_dict[token]

    return tf_idf
```

### Defining a Model Class

To build a multiclass classification model from scratch, we’ll use **multiclass logistic regression** with:

- A **softmax activation function** to produce probabilities
- A **cross-entropy loss** to measure how wrong the predictions are
- **Gradient descent** to update the weights and bias

We’ll define a Python class called `BoWLogisticClassifier` to encapsulate everything: prediction (`forward`), gradient computation (`backward`), and model parameters (`w` and `b`).


```python
import math

class BoWLogisticClassifier:
    """
    A sparse multiclass logistic regression model with softmax activation.
    Input vectors x are sparse dictionaries: {feature: value}.
    """

    def __init__(self, vocab, classes):
        """
        Initialize model with zero weights and zero biases.
        - vocab: a set of input feature names (e.g., words)
        - classes: a set of class labels (e.g., 'spam', 'ham')
        """
        self.vocab = sorted(vocab)
        self.classes = sorted(classes)
        self.input_dim = len(self.vocab)
        self.output_dim = len(self.classes)

        # Map vocab and classes to index
        self.vocab_index = {word: i for i, word in enumerate(self.vocab)}
        self.class_index = {label: i for i, label in enumerate(self.classes)}

        # Initialize weights and biases
        self.w = [[0.0 for _ in range(self.input_dim)] for _ in range(self.output_dim)]
        self.b = [0.0 for _ in range(self.output_dim)]

    def softmax(self, logits):
        """
        Apply softmax to logits for numerical stability.
        """
        max_logit = max(logits)
        exp_scores = [math.exp(z - max_logit) for z in logits]
        sum_exp = sum(exp_scores)
        return [s / sum_exp for s in exp_scores]

    def forward(self, x_sparse):
        """
        Compute prediction for sparse input x_sparse: dict {feature: value}.
        Returns probability distribution over classes.
        """
        logits = []
        for k in range(self.output_dim):
            score = self.b[k]
            for word, value in x_sparse.items():
                if word in self.vocab_index:
                    j = self.vocab_index[word]
                    score += self.w[k][j] * value
            logits.append(score)
        return self.softmax(logits)

    def backward(self, x_sparse, y_true_label):
        """
        Compute gradients w.r.t weights and biases using cross-entropy loss.
        - x_sparse: input features as dict {feature: value}
        - y_true_label: true class label (e.g., 'spam')

        Returns:
            dw: sparse gradient of weights (dict of dicts)
            db: dense gradient of biases (list)
        """
        y_true = self.class_index[y_true_label]
        y_pred = self.forward(x_sparse)

        # Gradients
        dw = {k: {} for k in range(self.output_dim)}
        db = [0.0 for _ in range(self.output_dim)]

        for k in range(self.output_dim):
            error = y_pred[k] - (1 if k == y_true else 0)
            for word, value in x_sparse.items():
                if word in self.vocab_index:
                    j = self.vocab_index[word]
                    dw[k][word] = error * value
            db[k] = error

        return dw, db
```

### Training the Classifier

Now let’s train our logistic regression model using gradient descent.

```python
from collections import defaultdict
import random

# Build vocabulary and class labels from training set
all_tokens = [token for message in train_df["message"] for token in tokenize(message)]
vocab = set(all_tokens)
classes = set(train_df["label"])

# Initialize model
model = BoWLogisticClassifier(vocab=vocab, classes=classes)

# Prepare training data as (x_sparse, y_label) pairs
train_data = []
for _, row in train_df.iterrows():
    tokens = tokenize(row["message"])
    x_sparse = vectorize_tf_idf(tokens)
    y_label = row["label"]
    train_data.append((x_sparse, y_label))

# Training loop
def train(model, data, lr=0.1, epochs=5):
    for epoch in range(epochs):
        random.shuffle(data)
        total_loss = 0.0

        for x_sparse, y_label in data:
            y_true_idx = model.class_index[y_label]
            y_pred = model.forward(x_sparse)

            # Cross-entropy loss
            loss = -math.log(y_pred[y_true_idx] + 1e-12)
            total_loss += loss

            # Compute gradients
            dw, db = model.backward(x_sparse, y_label)

            # Update weights and biases
            for k in range(model.output_dim):
                for word, grad in dw[k].items():
                    j = model.vocab_index[word]
                    model.w[k][j] -= lr * grad
                model.b[k] -= lr * db[k]

        avg_loss = total_loss / len(data)
        print(f"Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}")

# Train the model
train(model, train_data, lr=0.1, epochs=10)
```

### Predict Function

Let's create a `predict()` function that takes an unlabeled message and returns a predicted label:

```python
def predict(message, model, tokenizer, vectorizer):
    """
    Predict the class label for a raw input message (string).
    - message: the input message (e.g., "Free entry now!!!")
    - tokenizer: a function that tokenizes the message
    - vectorizer: a function that turns tokens into sparse vector

    Returns:
        predicted_label: the class label with highest probability
    """
    tokens = tokenizer(message)
    x_sparse = vectorizer(tokens)
    probs = model.forward(x_sparse)
    predicted_idx = probs.index(max(probs))
    return model.classes[predicted_idx]

```

### Evaluate Model Accuracy on the Test Set

Now that we have a working `predict()` function, we’ll apply it to every message in the test set.

Then we’ll compare the predicted labels to the actual labels and calculate the model’s accuracy.

```python
# Predict all messages in the test set
predictions = []

for _, row in test_df.iterrows():
    message = row["message"]
    prediction = predict(message,model,tokenize,vectorize_tf_idf)
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
## Logistic Regression vs. Naive Bayes: Why Logistic Regression Sometimes Performs Worse

You may have noticed that our **logistic regression model** doesn't perform quite as well as the **Naive Bayes model** on the SMS Spam Collection dataset. This might seem surprising at first — after all, logistic regression is more flexible and learned its weights directly from the data. So what’s going on?

### Why Logistic Regression Might Underperform

Here are a few possible reasons:

1. **Naive Bayes is surprisingly strong for text classification**  
   Even though it makes strong independence assumptions (i.e. that words are independent given the class), those assumptions actually hold *reasonably well* in many text datasets. This makes Naive Bayes hard to beat, especially on short documents like SMS messages.

2. **Logistic regression is more sensitive to sparse features**  
   Logistic regression must learn one weight per feature, and when the vocabulary is large and many features appear rarely (as in text), it can be harder for the model to learn stable estimates — especially with limited training data.

3. **No regularization in our implementation**  
   Our logistic regression model is very basic: it doesn’t use techniques like **L2 regularization**, which can prevent overfitting and improve generalization — especially with high-dimensional sparse inputs like TF-IDF vectors.

4. **Naive Bayes builds in strong priors**  
   Naive Bayes directly incorporates class priors (how common spam vs. ham messages are), and it’s especially effective when those priors are unbalanced. Logistic regression can learn this too, but it requires more data and care.

### Why Learn Logistic Regression Anyway?

Even though Naive Bayes outperforms logistic regression *here*, logistic regression is still **worth learning — and essential going forward**:

- It introduces the idea of **learning weights from data**, not from counts or rules  
- It teaches the concept of **gradient descent**, which we’ll use to train more complex models  
- It prepares us for **deep learning**, where logistic regression is effectively the **final layer** of most neural networks used for classification  
- It scales better to complex, high-dimensional data once regularization and optimization improvements are added

> **Key takeaway:**  
> Logistic regression might not win on this dataset — but it’s a critical stepping stone toward more powerful and general models, like deep neural networks and transformers.

## Group Project Reminder

Project Proposals are due next week. Class time will be used for groups to meet with the Professor to discuss their Project Proposals. 

## Midterm Reminder

The Midterm will take place the week after Study Week. There will be no class that week. The Midterm will be online and open book. Students will have the entire week to complete it. It will cover all lecture material. (Material covered in the textbook will not be covered directly on the midterm - the textbook is just a resource to help you understand the code used in class.)