# Week 5 Lecture Notes

This week, we introduce one of the most powerful families of models in modern machine learning: **neural networks**. We’ll start by building intuition around how neural networks work, why they’re called "neural", and how they learn from data. Along the way, we’ll revisit linear regression from a new perspective, explore the mathematics of learning via **gradient descent**, and extend our toolkit to include **logistic regression for classification**. By the end of this lecture, you’ll see how these foundational ideas set the stage for more advanced models like deep learning.


## What Is a Neural Network?

A **neural network** is a type of machine learning model that can learn patterns from data to make predictions.

Like the other models we've looked at, a neural network is really just a **mathematical equation**. It takes some **input** (represented as numbers), processes it through a series of steps, and produces some **output** (also represented as numbers).

What makes neural networks different is **how they learn**. Instead of using:
- Hard-coded rules (like we did with lexicon-based sentiment analysis)
- Predefined probabilities based on counting (like we did with Naive Bayes)

Neural networks learn their own internal rules using a method called **gradient descent** — a process that gradually adjusts the model’s parameters to reduce error over time.

### Why Are They Called Neural Networks?

The neural networks we'll study in this course are more specifically called **artificial neural networks**, named after the **biological neural networks** found in animal brains.

While the original idea was inspired by how real neurons fire and connect, artificial neural networks work quite differently. In practice, they are better thought of as **powerful pattern-matching systems** — mathematical models that learn from data, not true simulations of the brain.

## Tensor

Before we go further into neural networks, we need to talk about the data structures they use — specifically something called a **tensor**.

### Tensors Are Just Generalized Arrays

At a basic level, a **tensor** is just a container for numbers — kind of like a list, array, or table. In fact:

- A **scalar** (a single number) is a **0D tensor**
- A **vector** (a column or row of numbers) is a **1D tensor**
- A **matrix** (a rectangle of numbers) is a **2D tensor**
- A cube of numbers is a **3D tensor**

Tensors can be extended into higher dimensions as well.

### Input and Output Examples

Tensors are used to represent **both inputs and outputs** in neural networks. Depending on the type of data we're working with, tensors can have different dimensions.

Here are a few common examples:

- **Regression output (0D tensor)**  
  If your model predicts a single number (e.g., house price or sentiment score), the output is a scalar (0D tensor)

- **Document input (1D tensor)**  
  A document represented using bag-of-words or TF-IDF is a 1D tensor — a simple list of numbers

- **Classification output (1D tensor)**  
  In a classification model, the output can be represented as a 1D tensor with one element for each class  
  For example, if you're building a model to route incoming customer emails into one of three categories — **Billing**, **Sales**, or **Tech Support** — the model might output a vector like:  
  `[0.7, 0.1, 0.2]`  
  These numbers can be interpreted as **probabilities**. In this case, the model is most confident the email is about a billing issue

- **Black and white image input (2D tensor)**  
  A grayscale image is stored as a 2D tensor with shape: height × width

- **Colour image input (3D tensor)**  
  A colour image is stored as a 3D tensor with shape: height × width × colour channels  
  For example, a 64×64 RGB image would be represented as: 64 × 64 × 3

- **Video input (4D tensor)**  
  A short colour video can be represented as a 4D tensor with shape: time × height × width × channels  
  For example, a 10-frame video of 64×64 RGB images: 10 × 64 × 64 × 3

## Linear Regression

A **linear regression** is one of the simplest types of neural networks. It’s a model that learns to **predict a single number** (like a price, score, or value) by combining input features using a weighted sum.

The mathematical form of a linear regression is:

```
ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

- `x₁, x₂, ..., xₙ` are the input features  
- `w₁, w₂, ..., wₙ` are the feature weights the model learns  
- `b` is the bias weight (intercept) the model learns   
- `ŷ` is the predicted output (a single number)

This is also known as a **weighted sum** or a **linear combination**.

### Example: Predicting a House Price

Suppose we want to predict the price of a house based on three features:

- `x₁ = 1200` (square footage)  
- `x₂ = 3` (number of bedrooms)  
- `x₃ = 1` (has garage: 1 = yes, 0 = no)

Let’s say our model has learned the following weights:

- `w₁ = 150`  
- `w₂ = 10,000`  
- `w₃ = 5,000`  
- `b = 20,000` (the base price)

We plug these into the equation:

```
ŷ = (150 × 1200) + (10,000 × 3) + (5,000 × 1) + 20,000  
  = 180,000 + 30,000 + 5,000 + 20,000  
  = 235,000
```

So, the model predicts that the house is worth **$235,000**.

### Vector Form

You can also write this equation in **vector form**:

```
ŷ = w · x + b
```

- `w` is a **1D tensor** (vector) of weights  
- `x` is a **1D tensor** (vector) of input features  
- `·` represents the **dot product**, which multiplies each pair of values and sums the results  
- `b` is still the bias  
- `ŷ` is the output — a single number

This vectorized version does exactly the same thing: it multiplies each input by its corresponding weight, adds them all together, and then adds the bias.

It’s mathematically identical to the expanded version — just more compact and efficient, especially when working with large models or datasets.

## Loss Functions

So far, we've talked about how a linear regression model makes predictions using weights (`w`) and a bias (`b`). But how do we know if a particular set of weights is **good or bad**?

To train a model, we need to define what it means for a model to perform well. This is where **loss functions** come in.

### What Are We Trying to Do?

At the end of the day, we want to find weights and a bias that make the model's predictions as accurate as possible. That means we need a way to **measure how far off the predictions are** from the actual answers.

That’s the job of a **loss function**: it assigns a number (the **loss**) to each prediction, telling us how bad it was. Lower is better.

> **Key idea:**  
> A **loss function** tells us how well the model is doing.  
> We want to find the weights (`w`) and bias (`b`) that **minimize the loss**.

### Mean Squared Error (MSE)

For regression tasks (predicting a number), a common loss function is **Mean Squared Error (MSE)**:

```
MSE = average((ŷ - y)²)
```

- `ŷ` is the predicted value from the model  
- `y` is the true value from the training data  
- We subtract them to get the **error**, then square it (to make all errors positive and penalize bigger mistakes more)

#### Example:

If the true house price is `$300,000` and the model predicts `$280,000`, the squared error is:

```
(280,000 - 300,000)² = 400,000,000
```

If we do this for every house in the dataset and take the average, we get the MSE — a single number that summarizes how far off our predictions are.

### Why Not Just Take the Mean Error?

You might wonder: why not just compute the **mean error** like this?

```
average(ŷ - y)
```

This won’t work — because **positive and negative errors will cancel each other out**.  

For example, if the model overpredicts one example by 20 and underpredicts another by 20, the average error would be 0 — even though both predictions were wrong.

To avoid this cancellation, we square each error. This makes all the errors positive and ensures that **larger mistakes are penalized more heavily**.

### Why Not Just Take the Mean Absolute Error?

Another option is to use **Mean Absolute Error (MAE)**:

```
MAE = average(|ŷ - y|)
```

Instead of squaring the error, we take the **absolute value** — which also avoids cancellation and gives a clear measure of the average size of the errors.

So why don’t we use this as the default?

#### A Historical Reason: It Was Too Hard to Calculate

When linear regression was first developed in the early 1800s — long before calculators or computers — all the math had to be done **by hand**.  

And it turns out that **minimizing absolute error is much harder to do by hand** than minimizing squared error.

So squared error became the standard — not because it was the only option, but because it was the one that was **mathematically convenient** at the time. And it stuck.

That said, today **mean absolute error** is often used instead of MSE.

## Gradient Descent

Now that we know how to measure how "good" a set of weights is — using a **loss function** — the next question is:

> **How do we actually find good weights?**

There are infinitely many possible values for the weights and bias. We can't try them all, and there's no obvious formula that tells us the best ones.  
So instead of guessing randomly or exhaustively searching, we need an **intelligent way to search** for a good solution.

That strategy is called **gradient descent**.

### The Basic Idea

Imagine you're standing somewhere on a hilly landscape, and your goal is to get to the bottom of the valley — where the **loss** is lowest.

You can’t see the whole terrain, but you can feel which direction the ground slopes. So you take a step downhill. Then another. Then another.  
Eventually, you get closer and closer to the lowest point.

This is exactly what **gradient descent** does — but instead of navigating a landscape, it’s searching through possible weights for your model.  
It moves step-by-step in the direction that **reduces the loss**.

> **Gradient descent is an optimization algorithm that tweaks the model’s weights to reduce the loss.**

### How It Works

At each training step:

1. The model makes a prediction using the current weights and bias
2. The prediction is compared to the true answer using the **loss function**
3. The **gradient** of the loss is computed — how much the loss would change if we nudged each weight a little
4. The weights are updated by taking a small step in the direction that reduces the loss

This step is done for **each weight** and the bias.

#### Update Rule

The standard update formula looks like this:

```
wᵢ ← wᵢ - η * ∂L/∂wᵢ
```

- `wᵢ` is the current weight  
- `∂L/∂wᵢ` is the gradient of the loss with respect to that weight  
- `η` (eta) is the **learning rate** — a small constant that controls the step size  
- The new weight is the old weight **minus** a small step in the direction of the gradient

We do the same for the bias `b`:

```
b ← b - η * ∂L/∂b
```

> The minus sign is important — it tells us to move **downhill**, not uphill.

### The Learning Rate

The learning rate `η` is a small number like `0.01` or `0.001`. It controls **how big each step is**.

- If it's too small, training will be slow  
- If it's too big, training can overshoot the minimum and never settle

Finding a good learning rate is part of tuning the training process.

### A Simple Example

Let’s say we have a model with just one weight:

- The current weight is `w = 2.0`  
- The loss function tells us that increasing `w` increases the loss  
- The gradient is `∂L/∂w = 4.0`  
- We’re using a learning rate of `η = 0.1`

We update the weight:

```
w ← 2.0 - 0.1 * 4.0 = 1.6
```

We’ve taken a small step **toward lower loss**.

On the next step, we repeat the process — compute a new prediction, get the new loss and gradient, and update again.  
This process continues until the loss stops improving (or we reach a set number of steps).

### Why It Works

The magic of gradient descent is that it **uses the slope of the loss function** to guide the search.

- If the slope is steep, the weight update is bigger  
- If we’re close to the minimum, the slope flattens, and the updates become smaller  
- Over time, the model settles into a **minimum-loss** solution

> **Key idea:**  
> Gradient descent turns learning into a process of repeated small improvements,  
> guided by math, not guesswork.

### Will It Always Find the Best Solution?

Not necessarily.

Gradient descent is a powerful tool, but it doesn’t **guarantee** that we’ll find the best possible set of weights — the true global minimum of the loss function.

Why not?

- The **loss landscape** can be complicated, especially for deep neural networks
- It might have **multiple valleys** (local minima), **flat regions**, or **noisy slopes**
- Gradient descent can get stuck in a **local minimum**, or just settle somewhere “good enough”

In practice, this usually isn’t a big problem. For many real-world tasks, getting to a **low-loss** solution (even if it's not the lowest possible) is good enough — and gradient descent gets us there surprisingly well.

> So while gradient descent doesn’t always find *the best*, it often finds something **very useful** — and it does it fast.

### Batch Gradient Descent

When we train a neural network using gradient descent, we need to compute how wrong the model is — the **loss** — and then use that to update the weights.

But here’s an important question:

> **How much data should we use to compute that update?**

There are actually a few different strategies, and they each have tradeoffs.

#### 1. Full-Batch Gradient Descent

This is the most straightforward idea:

- Use the **entire training dataset** to compute the average loss  
- Then update the weights once, using the overall gradient


#### 2. Stochastic Gradient Descent (SGD)

At the opposite extreme, we can update the model using **just one example at a time**:

- Pick one training example  
- Compute the loss and gradient  
- Update the weights  
- Repeat

This is called **stochastic gradient descent** — "stochastic" meaning random. It updates the weights using **noisy**, fast updates.

#### 3. Mini-Batch Gradient Descent (the most common)

Stochastic Gradient Descent (SGD) updates the model using one training example at a time, which makes each update very fast. This can be a big advantage when working with large datasets. Because it's based on just a single example, SGD introduces some randomness ("noise") into the updates — which can help the model escape local minimums but also makes training less stable. Full-batch gradient descent, on the other hand, uses the entire dataset to compute each update. This gives a very accurate gradient and makes learning more stable and predictable — but each update is much slower, and it requires a lot more memory. In practice, most modern training uses a compromise: **mini-batch gradient descent**, which combines the speed of SGD with the stability of full-batch.

> **Mini-batch gradient descent** uses a small number of examples at a time — usually 32, 64, or 128.

Each step works like this:

- Pick a small **batch** of examples (e.g. 64 messages)
- Compute the average loss and gradient for just that batch
- Update the weights
- Move to the next batch

### Epochs

When we say we’ve done one **epoch** of training, it means we’ve gone through the **entire training dataset once**. Training usually takes **multiple epochs**, so the model gets many chances to improve. Over time, the weights are refined to make better and better predictions.


## Deriving the Gradient of MSE Loss for Linear Regression

Let’s derive the gradient we need for training a linear regression model using gradient descent.

### Step 1: The Model

We start with the linear regression equation:

**ŷ = w · x + b**

Where:  
- `x` is the input vector (a 1D tensor of features)  
- `w` is the weight vector (same size as `x`)  
- `b` is the bias (a scalar)  
- `ŷ` is the predicted output (a scalar)

### Step 2: The Loss Function

We’ll use **Mean Squared Error (MSE)** as our loss function.  
For a single training example:

**L = (ŷ - y)²**

Where:  
- `ŷ = w · x + b` is the predicted value  
- `y` is the true value from the training data

We want to compute the gradients of `L` with respect to `w` and `b`.

### Step 3: Gradient with Respect to Weights (w)

We use the chain rule:

**dL/dw = dL/dŷ · dŷ/dw**

First:

**dL/dŷ = 2(ŷ - y)**

Then:

**dŷ/dw = x**

Putting it together:

**dL/dw = 2(ŷ - y) · x**

### Step 4: Gradient with Respect to Bias (b)

Again, using the chain rule:

**dL/db = dL/dŷ · dŷ/db**

We already have:

**dL/dŷ = 2(ŷ - y)**

And:

**dŷ/db = 1**

So:

**dL/db = 2(ŷ - y)**

### Final Gradient Equations

For one training example:

- **Gradient of the weights:**  
  **∇w L = 2(ŷ - y) · x**

- **Gradient of the bias:**  
  **∇b L = 2(ŷ - y)**

These gradients tell us how to update the weights and bias in the direction that reduces the loss.

### Note on Averaging (for Mini-Batches)

If you're working with a **batch of examples**, you would average the gradients over the batch:

- **∇w L = average over batch of [2(ŷ - y) · x]**  
- **∇b L = average over batch of [2(ŷ - y)]**

This ensures that the update reflects the overall trend across the examples, not just a single case.

## Predicting Ice Cream Sales

Imagine you run an **ice cream shop**, and you’ve recorded your **daily ice cream sales** alongside the **temperature** on that day.

We want to build a model that can predict **ice cream sales (`y`)** based on the **temperature (`x`)**.

### Dataset

We'll use a simple **synthetic dataset** that simulates our real-world scenario. The dataset is generated using this rough formula:

```
y ≈ 126 + 13x + noise
```

- `x` is the temperature (in degrees Celsius), randomly sampled between 0 and 30  
- `y` is the number of ice cream cones sold, with some random noise added to simulate real-world variability  
- The noise makes the relationship more realistic — it's not perfectly linear, just mostly linear

This synthetic data allows us to:
- Focus on the **core ideas** of linear regression
- Keep the math and code simple
- Easily **visualize** the data and the model, since it has only one feature

We’ve wrapped the dataset in a small Python class called `IceCreamDataSet` that lets us generate and plot the data easily.

```python
import random
import matplotlib.pyplot as plt

random.seed(13)

class IceCreamDataSet:

  def __init__(self):
    self.n = 100
    self.x = [random.uniform(0,30) for _ in range(self.n)]
    self.y = [round(max(0,126 + 13*xi + random.normalvariate(0,26))) for xi in self.x]

  def plot_data(self):
    plt.figure(figsize=(8,6))
    plt.scatter(self.x,self.y)
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Ice Cream Sales")
    plt.title("Ice Cream Sales vs. Temperature")
    plt.show()

  def plot_model(self,model):
    plt.figure(figsize=(8,6))
    plt.scatter(self.x,self.y)
    line_x = [min(self.x), max(self.x)]
    line_y = [model.forward(xi) for xi in line_x]
    plt.plot(line_x,line_y,color='red')
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Ice Cream Sales")
    plt.title("Model Prediction vs. Data")
    plt.show()
```

Here, we create an instance of the synthetic dataset, simulating 100 days of ice cream sales based on daily temperature.

```python
dataset = IceCreamDataSet()
```

Let's show a scatterplot of the data, where each point represents one day's temperature and the corresponding number of ice cream cones sold.

```python
dataset.plot_data()
```

### Building a Simple Linear Regression Model

Now that we have a dataset, we need a model to learn from it.

We’ll create a simple **linear regression model** from scratch. The model will predict ice cream sales using this formula:

```
ŷ = w * x + b
```

Where:

- `x` is the input (temperature)
- `ŷ` is the predicted output (sales)
- `w` is the **weight** (slope), and `b` is the **bias** (intercept) — both are parameters the model will learn

Here’s a class that defines the model and its two core operations:

- `forward`: Makes a prediction
- `backward`: Computes gradients of the loss with respect to `w` and `b`, using Mean Squared Error (MSE)

```python
class LinearModel:
    """
    A simple linear regression model:
    ŷ = w * x + b
    """

    def __init__(self):
        # Initialize weights and bias to zero
        self.w = 0
        self.b = 0

    def forward(self, x):
        """
        Predict the output ŷ given input x.
        """
        return self.w * x + self.b

    def backward(self, x, y):
        """
        Compute gradients of the loss with respect to w and b
        using Mean Squared Error (MSE) for a single example.
        """
        y_pred = self.forward(x)
        error = y_pred - y
        dw = 2 * error * x   # gradient w.r.t. w
        db = 2 * error       # gradient w.r.t. b
        return dw, db
```

Let's use the class to create a model. 

```python
model = LinearModel()
```

Let's take a look at the initial weights:

```python
dataset.plot_model(model)
```

#### Why Is It Called "Backward"?

The function that computes the gradient is called `backward` because it’s about working **backward from the error** to figure out how to adjust the model.

In the **forward** step, the model takes an input (like temperature), runs it through the equation (`ŷ = w * x + b`), and produces a prediction.  

Then we compare that prediction to the true answer and calculate how wrong it was — the **loss**.

In the **backward** step, we take that loss and ask:  
> *How should we change the weights and bias to reduce this error next time?*

To answer that, we work backward through the equation, computing how much each part (each parameter) contributed to the mistake. This is the start of a process called **backpropagation** — short for *backward propagation of errors*.

In more complex networks, this process involves the **chain rule** from calculus, which allows us to pass gradients backward through layers of functions.  
We’ll see that later — but even now, you’re seeing the core idea:  
> *To learn, a model must look at the error and trace it back to the parts that caused it.*

### Training the Model (One Example at a Time)

Now that we have a model and a dataset, let’s train the model using **gradient descent**.

We'll use a very simple form of training called **stochastic gradient descent**. That means we’ll update the model **one example at a time** — using just a single `(x, y)` pair per update.

For each training example:

1. Make a prediction using the model  
2. Compare it to the actual value to get the error  
3. Use the `backward()` method to compute the gradients  
4. Adjust the model’s parameters (`w` and `b`) to reduce the error

To visualize the training process, we’ll also plot the model’s current line after each epoch, and pause briefly so we can see it evolve.

```python
import time
from IPython.display import clear_output

# Set the learning rate — controls how big each update is
learning_rate = 0.001

# Train for 100 passes (epochs) over the dataset
for epoch in range(100):
    for i in range(dataset.n):
        # Get one example (temperature and sales)
        x = dataset.x[i]
        y = dataset.y[i]

        # Forward pass: make a prediction
        y_pred = model.forward(x)

        # Backward pass: compute gradients
        dw, db = model.backward(x, y)

        # Update weights and bias using gradient descent
        model.w -= learning_rate * dw
        model.b -= learning_rate * db

    # Clear previous plot/output
    clear_output(wait=True)

    # Print current progress
    print("Epoch:", epoch)
    print("Learned weight (w):", model.w)
    print("Learned bias (b):", model.b)

    # Plot the current model line
    dataset.plot_model(model)

    # Pause so we can see the update before moving on
    time.sleep(0.5)
```

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

To see how logistic regressions work, let's try building one. We'll use the same dataset we used last week. 

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

## Summary

This week, we explored the building blocks of modern neural networks and machine learning through hands-on implementations and intuitive explanations:

- We introduced **artificial neural networks** as powerful pattern-matching systems
- We learned how **tensors** serve as the basic data structure for neural models
- We re-framed **linear regression** as a single-layer neural network
- We studied **loss functions** — especially **mean squared error (MSE)** — to measure how wrong a model is
- We implemented **gradient descent**, the algorithm that adjusts model weights to reduce error
- We trained a model to predict **ice cream sales from temperature**, step by step
- We extended linear models to **multiclass classification** with **softmax** and **cross-entropy loss**
- We built a complete **logistic regression classifier** for spam detection using sparse TF-IDF vectors

Even though our logistic regression model didn’t outperform Naive Bayes on this dataset, it marks an important transition: we are now building models that **learn directly from data**, laying the foundation for deep learning and neural architectures we’ll explore in future weeks.

## Exercises

Try the following exercises to reinforce this week’s concepts:

1. **Manual Prediction**  
   Using the linear regression example from class, compute the prediction for a new house with features:  
   `x₁ = 1500`, `x₂ = 4`, `x₃ = 0`  
   Use the same weights as in the house price example.

2. **Derive the Gradient**  
   Walk through the derivation of the MSE gradient for a single data point using the chain rule.  
   Explain what each part means in plain English.

3. **Experiment with Learning Rate**  
   Modify the ice cream training code to try different learning rates. What happens if you set it to `0.0001` or `0.1`?

4. **Inspect the Weights**  
   Print the 10 largest and smallest weights from your logistic regression model.  
   What kinds of words are getting high/low weights?

5. **Backpropagation Reflection**  
   In your own words, explain why the `.backward()` function is called "backward".  
   What’s being traced backward, and why is that important?

6. **Try a New Dataset**  
   Replace the SMS spam dataset with a small custom dataset (e.g., short labeled tweets or comments) and try retraining the logistic regression model.


## Group Project Reminder

Each group must meet with me to discuss their Project Proposal. Meetings will take place during the Week 7 class. Not all members of the group need to attend. If no one from your group is able to attend during class time, please email me by June 8 to make an accommodation.

