# Week 4 Lecture Notes

This week, we built our first **supervised machine learning classifier** from scratch — a Naive Bayes model for **spam detection**. Along the way, we compared rule-based and learning-based approaches, reviewed key ML concepts, and connected everything back to ideas we've already explored in sentiment analysis. By the end of this unit, you’ll understand not just how Naive Bayes works, but why it works — and how surprisingly similar it is to the lexicon-based classifier we built earlier.

## Rule-Based vs. Machine Learning–Based Systems

Before we dive into building our first *machine learning model*, let’s review the difference between two major approaches to analyzing text:

### Rule-Based Systems

Rule-based systems rely on **handcrafted logic** to analyze text. You define the rules, and the system applies them.

**Examples:**

- If a message contains the word “refund”, label it as a complaint.
- If a review contains more positive than negative words (from a lexicon), predict it as positive.

We’ve already seen some rule-based systems in this course:

- **Information retrieval** using cosine similarity and TF-IDF is rule-based — it follows a fixed mathematical formula to rank documents.
- **Lexicon-based sentiment analysis** is rule-based — it assigns scores to documents based on a predefined dictionary of word sentiments. *Note: you could use machine learning to create the lexicon (e.g., by training on labeled data), but our approach was purely rule-based.*

**Pros:**

- Simple, interpretable, and easy to build
- No training data needed
- Good for clear-cut or highly structured tasks

**Cons:**

- Rigid and brittle — can break with new wording or phrasing
- Can’t handle nuance or subtlety (e.g., sarcasm, negation)
- Hard to scale and maintain as complexity grows

### Machine Learning–Based Systems

Machine learning systems **learn patterns from labeled examples** instead of being explicitly told what to do.

You give the model **input texts** and the correct **output labels**, and it finds patterns that connect them.

**Examples:**

- Train a model on 10,000 emails labeled “spam” or “not spam”
- Use word frequency patterns to automatically learn what spam looks like
- Predict new labels for unseen messages based on learned patterns

We’ve already seen a machine learning system:

- **K-Means clustering** is an unsupervised ML model — it learns to group similar documents without predefined categories.

**Pros:**

- Flexible and adaptive — learns from data, not rules
- Can capture subtle statistical signals
- Scales well to large datasets and complex problems

**Cons:**

- Needs labeled data to train (in supervised learning)
- Can be harder to interpret
- May make mistakes in unexpected ways

## Supervised vs. Unsupervised Learning

Now that we’ve reviewed machine learning systems, let’s look at two major types of learning: **supervised** and **unsupervised**.

### Unsupervised Learning

In **unsupervised learning**, the model is given input data **without any labels**. Its goal is to **find patterns or structure** in the data on its own.

**Example: K-Means Clustering**

- We gave K-Means a set of review vectors — but **no labels** or categories.
- The model grouped similar reviews into clusters based on shared vocabulary.
- We didn’t tell the model what the clusters should be — it discovered patterns automatically.

Unsupervised learning is useful when:
- You don’t have labeled data
- You want to explore or summarize large datasets
- You want to discover hidden structure or themes

### Supervised Learning

In **supervised learning**, the model is trained on input data **with known labels**. It learns the relationship between the input and the correct output, so it can make predictions on new data.

**What We’re Doing Today: Naive Bayes**

- We'll build a **spam detection model** using a dataset of messages labeled as **spam** or **not spam**.
- The model will learn what kinds of words are commonly found in spam messages vs. regular ones.
- Once trained, the model can classify **new, unseen messages** as spam or not.

> **Key Idea**:  
> Supervised learning requires labeled data and predicts known outcomes.  
> Unsupervised learning looks for patterns without any guidance.


## Introducing Naive Bayes: A Machine Learning Approach to Text Classification

So far, we’ve built a **lexicon-based sentiment classifier**:  

- Every word in a review had a predefined **sentiment score** (positive, negative, or neutral).  
- To score a review, we **added up the scores of the words it contained**.  
- Based on the total score, we classified the review as **positive, neutral, or negative**.

Today, we’ll build a new kind of text classifier — using **machine learning**.  

Our task is to classify messages as either **spam** or **not spam** (also called *ham*).  

We’ll use a model called **Naive Bayes**, which is commonly used for spam detection.

### Similarities to the Lexicon Model

In both models:

- Each message is represented as a **bag of words**
- Each word contributes to the **final message score**
- The final score determines the predicted **class label**:
  - Positive vs. Negative (in sentiment analysis)
  - Spam vs. Not Spam (in today’s example)

### Key Difference: Where the Word Scores Come From

- In the **lexicon model**, the word scores were **manually defined** in a sentiment dictionary.  
- In the **Naive Bayes model**, the word scores will be **learned from data**.  
  - We’ll train the model on **labeled messages** — ones that are already marked as spam or not.
  - The model will learn which words are more likely to appear in spam vs. ham.

This is what makes Naive Bayes a **supervised machine learning model**.

> **Key Idea**:  
> Just like our lexicon-based model, Naive Bayes uses word-level scores to make a prediction —  
> but it **learns** those scores automatically by analyzing a labeled dataset.

## Deriving Naive Bayes for Spam Detection

To understand how our model works, we’ll start with **Bayes’ Rule** — a fundamental idea in probability theory.

### Bayes’ Rule

Bayes’ Rule helps us reverse conditional probabilities:

**P(A | B) = P(B | A) * P(A) / P(B)**

In our case:

- **A** is the class (e.g., "spam" or "not spam")
- **B** is the message text

We want to calculate:

**P(spam | message)** — the probability that a message is spam, given its contents.

Using Bayes’ Rule:

**P(spam | message) = P(message | spam) * P(spam) / P(message)**

And likewise:

**P(ham | message) = P(message | ham) * P(ham) / P(message)**

To classify the message, we compare these two probabilities and choose the class with the higher value.

### The Naive Assumption

The tricky part is computing **P(message | spam)** — the probability of the entire message, given that it’s spam.

Since messages are made up of many words, this would normally be very hard to compute.  
So we make a simplifying assumption:

> **Naive Bayes assumes that all words in a message are independent, given the class.**

That means:

**P(message | spam)**  
≈ **P(w1 | spam) * P(w2 | spam) * ... * P(wn | spam)**

We do the same for ham:

**P(message | ham)**  
≈ **P(w1 | ham) * P(w2 | ham) * ... * P(wn | ham)**

### The Final Classification Rule

Now we can skip the denominator (P(message)) because it’s the same for both classes.

To classify a message, we compute:

- **spam score = P(spam) * P(w1 | spam) * P(w2 | spam) * ...**
- **ham score = P(ham) * P(w1 | ham) * P(w2 | ham) * ...**

Then we pick the class with the higher score.

### Take the Log

Multiplying lots of small probabilities (like 0.01 × 0.005 × 0.0008...) can lead to **very tiny numbers** that computers have trouble storing.  

Instead, we take the **logarithm** of the scores, which turns multiplication into addition:

**log(spam score) = log(P(spam)) + log(P(w1 | spam)) + log(P(w2 | spam)) + ...**

And the same for ham:

**log(ham score) = log(P(ham)) + log(P(w1 | ham)) + log(P(w2 | ham)) + ...**

This gives us a much more stable calculation — and it’s also easier to interpret.

### Final Classification Rule

We subtract the two log-scores and check the sign of the result:

**score = log(spam score) - log(ham score)**  
= **log(P(spam) / P(ham))**  
  + **sum over words: log(P(word | spam) / P(word | ham))**

This gives us a **single number** that summarizes how "spammy" the message is.

### How to Make a Prediction

- If the score is **greater than 0**, the message is **more likely spam**
- If the score is **less than 0**, the message is **more likely ham (not spam)**
- If the score is **exactly 0**, the model is completely uncertain (rare)


### Why This Looks Like a Lexicon Model

- In lexicon-based sentiment analysis, we added up the **sentiment scores** of each word.
- In Naive Bayes (after taking logs), we add up the **spaminess scores** of each word:
  - For each word: **log(P(word | spam) / P(word | ham))**

This becomes our **spam lexicon**, but instead of being handcrafted, it's **learned from data**.

### What About the Constant?

In our log-based Naive Bayes classifier, the full formula looks like this:

**score = log(P(spam) / P(ham))  
  + sum over words: log(P(word | spam) / P(word | ham))**

The first part — **log(P(spam) / P(ham))** — is a **constant**. It doesn’t depend on the message or its words. It just reflects how common spam is in the training data compared to ham.

#### What Does It Mean?

- If spam is more common than ham in the training set, the constant will be **positive**
- If ham is more common, the constant will be **negative**

This constant shifts the final score **up or down** and acts as a **bias term**:
- A high constant makes the model more likely to predict spam (unless the words strongly suggest otherwise)
- A low constant makes the model more conservative (only labeling as spam if the words are very spammy)

#### Why Don’t We Have This in Lexicon-Based Sentiment?

In our sentiment model, we only use the sum of word scores — there’s **no class prior** built into the model.

That means:
- We assume that positive and negative reviews are equally likely by default
- There's **no bias toward one label or the other** unless the words push it in that direction

If we wanted to, we could **add a constant** to our sentiment model to reflect prior expectations — for example, if we know most reviews tend to be positive — but in practice, we usually leave it out for simplicity.

## SMS Spam Collection

Now that we understand how the Naive Bayes model works, let’s put it into practice.

We’ll be using a real-world dataset called the **SMS Spam Collection**. It contains **5,574 text messages**, each labeled as either:

- **"spam"** — unwanted commercial messages, scams, or promotions
- **"ham"** — regular, non-spam messages (e.g., from friends, family, or service providers)

This dataset is widely used to teach and evaluate text classification models, and it's perfect for our goal today.

### Our Objective

We want to **train a Naive Bayes classifier** that can automatically determine whether a new message is spam or not.

To do that, we’ll:

1. **Load the dataset**
2. **Tokenize the messages**
3. **Estimate word probabilities for each class**
4. **Compute log-scores for new messages**
5. **Make predictions and evaluate accuracy**

This is a classic example of **supervised learning**:

- We train the model on a set of **labeled examples**
- The model learns which words are more likely to appear in spam vs. ham
- It uses this to classify **new, unseen messages**

Let’s get started by loading and exploring the data.

## Download and Load the Dataset

The SMS Spam Collection is available as a **tab-delimited text file** where:

- The first column is the **label** (`ham` or `spam`)
- The second column is the **message text**

We’ll download the file using `wget`, then read it into a DataFrame using `pandas`.

```python
# Download the dataset
!wget https://storage.googleapis.com/wd13/SMSSpamCollection.txt

# Load it into a pandas DataFrame
import pandas as pd

df = pd.read_csv("SMSSpamCollection.txt", sep="\t", header=None, names=["label", "message"])

# Display the first few rows
df.head()
```

Each row represents one SMS message, labeled as either `"ham"` or `"spam"`. We'll use this DataFrame as our training data for the Naive Bayes classifier.

### A Quick Note on Pandas and DataFrames

In this example, we're using a Python library called **pandas**. It's one of the most popular tools for working with data in Python.

The main structure in pandas is the **DataFrame** — a table of data with **rows and columns**, similar to a spreadsheet or a SQL table.

When we load our SMS dataset using `pd.read_csv()`, we get a DataFrame where:

- Each **row** is one text message
- Each **column** holds a different kind of information (like the label or the message text)

For example:

| label | message                                  |
|-------|------------------------------------------|
| ham   | Go until jurong point, crazy..           |
| ham   | Ok lar... Joking wif u oni...            |
| spam  | Free entry in 2 a wkly comp to win FA... |

We can inspect the first few rows of any DataFrame using `.head()`.

## Split Into Training and Test Sets

Before we train our Naive Bayes model, we need to split our dataset into two parts:

- **Training set**: The portion of the data we use to **teach** the model. It learns which words are common in spam vs. ham.
- **Test set**: A separate portion that we use to **evaluate** the model — to see how well it performs on new, unseen messages.

*Why do we split the data?* If we evaluate the model on the same data it was trained on, we’re not really testing its ability to generalize.  To truly see how well it works, we need to test it on data it hasn’t seen — just like how it would be used in the real world.

To do this, we’ll use a function called `train_test_split` from a library called **scikit-learn** (or `sklearn` for short).  
Scikit-learn is one of the most widely used machine learning libraries in Python — it provides tools for building models, evaluating them, and managing data workflows.

```python
from sklearn.model_selection import train_test_split

# Split the dataset: 80% training, 20% testing
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Show the number of messages in each set
print("Training messages:", len(train_df))
print("Test messages:", len(test_df))
```

The `random_state` parameter is like a seed for randomness — it ensures that we get the same split every time we run this code.

Now we’re ready to start building and training our Naive Bayes model using the `train_df` data!

## Tokenizer

Just like we did last class, we’ll need a tokenizer. We’ll use the same function we defined earlier, which:

- Converts text to lowercase
- Uses a regular expression to extract word-like tokens
- Removes stop words (common words like “the”, “is”, “and” that don’t carry much meaning)

Here’s the function:

```python
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def tokenize(text):
    lowercase_text = text.lower()
    tokens = re.findall(r'\b\w+\b', lowercase_text)
    return [t for t in tokens if t not in ENGLISH_STOP_WORDS]
```

## Create Word Frequency Tables

Now that we can tokenize messages, we’ll count how often each word appears in:

- Spam messages
- Ham (non-spam) messages

These counts will help us estimate the probabilities **P(word | spam)** and **P(word | ham)** — which the Naive Bayes model needs to make predictions.

We’ll create two dictionaries:

- `spam_counts` — maps each word to how many times it appears in spam messages
- `ham_counts` — same idea, but for ham messages

```python
# Initialize word count dictionaries
spam_counts = {}
ham_counts = {}

# Loop through the training messages
for _, row in train_df.iterrows():
    tokens = tokenize(row["message"])
    label = row["label"]

    for token in tokens:
        if label == "spam":
            spam_counts[token] = spam_counts.get(token, 0) + 1
        else:
            ham_counts[token] = ham_counts.get(token, 0) + 1
```

At this point:

- `spam_counts["win"]` tells us how many times the word "win" appeared in spam messages
- `ham_counts["ok"]` tells us how many times the word "ok" appeared in ham messages

These raw counts are the building blocks of our model. Next, we’ll convert them into **probabilities** — and apply **Laplace smoothing** to avoid zero-probability issues.

## Estimate P(word | spam) and P(word | ham) with Laplace Smoothing

Now that we have word frequency tables, we want to calculate:

- **P(word | spam)** — how likely a word is to appear in spam messages
- **P(word | ham)** — how likely it is to appear in ham messages

But there's a problem:  If a word never appears in one category (e.g., `"pizza"` never shows up in spam), then its probability is 0 — which would cause division by zero errors. To avoid this, we use **Laplace smoothing**.

### Laplace Smoothing Formula

To calculate smoothed probabilities, we use this formula:

**P(word | class) = (count + 1) / (total_count + V)**

Where:
- `count` = how many times the word appears in that class (spam or ham)
- `total_count` = total number of words in that class
- `V` = size of the full vocabulary (number of unique words across both classes)

Adding `1` ensures that **every word has at least a tiny non-zero probability** in both classes.

```python
# Combine vocab from both classes
vocab = set(spam_counts.keys()) | set(ham_counts.keys())
V = len(vocab)

# Total number of words in each class
spam_total = sum(spam_counts.values())
ham_total = sum(ham_counts.values())

# Create smoothed probability tables
P_word_given_spam = {}
P_word_given_ham = {}

for word in vocab:
    # Smoothed spam probability
    spam_count = spam_counts.get(word, 0)
    P_word_given_spam[word] = (spam_count + 1) / (spam_total + V)

    # Smoothed ham probability
    ham_count = ham_counts.get(word, 0)
    P_word_given_ham[word] = (ham_count + 1) / (ham_total + V)
```
At this point:

- `P_word_given_spam["win"]` gives us the smoothed probability of the word "win" in spam
- `P_word_given_ham["ok"]` gives us the smoothed probability of "ok" in ham

These probabilities are now safe to use — even for rare or previously unseen words.

Next, we’ll convert these into **log probabilities** and build a function to classify new messages!

## Convert to Log-Probabilities

In Naive Bayes, we multiply together many small probabilities — one for each word in the message.  
But multiplying lots of small numbers can quickly underflow (i.e., become too small for the computer to handle).

To fix that, we take the **logarithm** of each probability — which turns multiplication into addition:

- `log(P(w1 | spam) × P(w2 | spam))` → `log(P(w1 | spam)) + log(P(w2 | spam))`

This also makes our model behave more like the **lexicon-based model** from earlier —  
Each word contributes a score, and we just **add them up**.

We’ll also precompute the **log-ratio** for each word:

```python
import math

log_ratios = {}

for word in vocab:
    pw_spam = P_word_given_spam[word]
    pw_ham = P_word_given_ham[word]
    log_ratios[word] = math.log(pw_spam / pw_ham)
```

This dictionary `log_ratios` is now our **"spam lexicon"**:

- Words with positive values are more spammy
- Words with negative values are more ham-like
- Words with scores near zero don’t strongly favor either class

Now we’re ready to build the final prediction function — using these word-level log scores!

## Classify a Message with Naive Bayes

To predict whether a message is spam or ham, we:

1. Tokenize the message
2. For each word, look up its **log-ratio**: `log(P(word | spam) / P(word | ham))`
3. Add those values together
4. Add the **log prior ratio**: `log(P(spam) / P(ham))`
5. Predict **spam if the total score > 0**, otherwise **ham**

Let’s calculate the class priors from our training data:

```python
# Count spam and ham messages
num_spam = (train_df["label"] == "spam").sum()
num_ham = (train_df["label"] == "ham").sum()

# Compute class prior probabilities
P_spam = num_spam / len(train_df)
P_ham = num_ham / len(train_df)

# Compute log prior (the constant)
log_prior = math.log(P_spam / P_ham)
```

Now we’ll write a prediction function that:

- Computes the total log-score of a message
- Returns `"spam"` if the score is greater than 0
- Returns `"ham"` otherwise

```python
def predict(message):
    tokens = tokenize(message)
    score = log_prior  # start with the class bias

    for token in tokens:
        if token in log_ratios:
            score += log_ratios[token]

    return "spam" if score > 0 else "ham"
```

Let’s try it out on a few test messages!

```python
print(predict("Congratulations! You've won a free ticket to Bahamas. Click here to claim."))
print(predict("Hey, are we still on for dinner tonight?"))
```

## Evaluate Model Accuracy on the Test Set

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

This tells us how well our Naive Bayes model generalizes to new data.

In most cases, you should see accuracy well above **90%**, even with this simple bag-of-words approach.

Next: we can analyze **which words had the strongest influence** on the model — or look at **false positives and false negatives** to better understand where it succeeds or fails.

### Confusion Matrix

A **confusion matrix** helps us see where the model gets things right and wrong:

- **True Positives (TP)**: Spam correctly labeled as spam  
- **True Negatives (TN)**: Ham correctly labeled as ham  
- **False Positives (FP)**: Ham incorrectly labeled as spam  
- **False Negatives (FN)**: Spam incorrectly labeled as ham

We’ll use `sklearn.metrics.confusion_matrix` to build one.

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

**Why Accuracy Isn’t Always Enough**

Suppose 99% of the messages in your dataset are **ham**.  
A model that simply **guesses "ham" every time** would be right 99% of the time — and get **99% accuracy** — but it would be **completely useless** at detecting spam.

This is why we need to look beyond accuracy and examine the **confusion matrix**:
- It shows **how many spam messages were missed** (false negatives)
- And **how many ham messages were mislabeled** as spam (false positives)

These errors matter a lot — especially in real-world systems like email filtering or fraud detection, where the "rare" class is often the most important to catch.

### Precision, Recall, and F1-Score

When evaluating a classifier — especially on **imbalanced datasets** — it's important to go beyond accuracy.

Let’s define three important metrics:

#### 🔹 Precision

> Of all the messages the model predicted as **spam**, how many were actually spam?

High precision means **few false positives**.

#### 🔹 Recall

> Of all the actual spam messages, how many did the model correctly identify?

High recall means **few false negatives** — you're catching most of the spam.

#### 🔹 F1-Score

> A balanced average of precision and recall.

F1 = 2 × (precision × recall) / (precision + recall)

Useful when you care about both avoiding **false alarms** and **missing real spam**.

We can compute all three using `sklearn.metrics.classification_report`:

```python
from sklearn.metrics import classification_report

print(classification_report(actual, predictions, target_names=["spam", "ham"]))
```

This gives you a full breakdown of:

- **Precision**: Of the messages predicted as spam/ham, how many were actually correct?
- **Recall**: Of the actual spam/ham messages, how many did the model correctly identify?
- **F1-score**: A balance between precision and recall — high only when both are high.
- **Support**: The number of true examples of each class in the test set.
- **Macro avg**: The unweighted average across classes — treats each class equally.
- **Weighted avg**: The average weighted by class frequency — reflects overall performance on imbalanced data.

These metrics help you understand where your model is strong — and where it needs improvement. They are especially useful when the dataset has class imbalance (e.g., more ham than spam).

### Most "Spammy" and "Hammy" Words

Let’s inspect which words had the strongest influence on our model — the highest and lowest log-ratios.

Words with very positive log-ratios are strong indicators of spam.

Words with very negative log-ratios are strong indicators of ham.

```python
# Sort words by log-ratio
sorted_words = sorted(log_ratios.items(), key=lambda x: x[1])

# Most hammy (very negative log-ratio)
print("💬 Most ham-like words:")
for word, score in sorted_words[:10]:
    print(f"{word:15} {score:.4f}")

# Most spammy (very positive log-ratio)
print("\n💬 Most spam-like words:")
for word, score in sorted_words[-10:]:
    print(f"{word:15} {score:.4f}")
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

## Summary


This week, you learned how to:

- **Compare rule-based and machine learning–based systems** for text analysis  
- Distinguish between **supervised** and **unsupervised** learning  
- Use **Bayes’ Rule** to build a Naive Bayes classifier for spam detection  
- Implement a **bag-of-words** model with **Laplace smoothing**  
- Convert word-level probabilities into **log-scores**  
- Classify new messages using a trained Naive Bayes model  
- Measure performance using **accuracy**, **confusion matrices**, and **precision/recall/F1**  
- Inspect **model behavior** by analyzing the most "spammy" and "hammy" words  
- Perform basic **error analysis** to see where the model makes mistakes  

Even though Naive Bayes is a relatively simple algorithm, it’s highly effective — and a great foundation for building intuition about how machine learning works with text.

## Exercises

1. **Manual Classification**  
   Pick 3 messages from the dataset (or your own inbox!).  
   Tokenize each one and estimate whether it’s more likely spam or ham using your intuition and word-level log scores.

2. **Evaluate with More Metrics**  
   Use `classification_report` to get the precision, recall, and F1-score for your model.  
   What do these scores tell you about your model’s strengths and weaknesses?

3. **Inspect the Lexicon**  
   Print the 20 most "spammy" and "hammy" words from the `log_ratios` dictionary.  
   Are any of them surprising? Do they make sense?

4. **Adjust the Prior**  
   What happens if you **manually increase or decrease** the prior (log(P(spam)/P(ham)))?  
   Try removing the prior entirely and observe how that affects performance on the test set.

5. **Compare with Lexicon Model**  
   Conceptually compare the Naive Bayes model to the lexicon-based sentiment model from Week 2.  
   What’s similar? What’s different?

6. **Error Inspection**  
   Use the error analysis loop to inspect at least 5 misclassified messages.  
   Try to identify *why* the model was confused.

7. **Explore Further**  
   Try applying your Naive Bayes classifier to a **different dataset** — for example, Yelp reviews labeled as positive or negative.  
   What would change? What parts of your code could be reused?

## Homework

- Read chapters 16 thru 20 of [Think Python](https://allendowney.github.io/ThinkPython/)
- Complete the Exercises above
- Get going on your Group Project