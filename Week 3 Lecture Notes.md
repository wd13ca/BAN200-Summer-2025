# Week 3 Lecture Notes

This week, we dive deeper into the **practical mechanics of text mining** by working with real-world app reviews from the Google Play Store. You’ll learn how to collect and process raw text data, transform it into structured numerical representations (using techniques like **TF**, **TF-IDF**, and **sparse vectors**), and use these representations to perform tasks such as **sentiment scoring** and **document clustering**. We’ll also explore cosine similarity and apply **K-means clustering** to uncover hidden patterns in user feedback — skills that form the foundation of many real-world text analytics applications.

## Downloading a Corpus of Google Play Store Reviews

Today, we are going to be analyzing some reviews downloaded from the Google Play Store.

### Google Play Scraper

`google-play-scraper` is a Python library that allows you to programmatically access data from the Google Play Store. You can use it to retrieve app information, ratings, and — most importantly for us — user reviews. This makes it a useful tool for collecting real-world text data that we can analyze using sentiment analysis techniques.

### PIP

`pip` is the standard package manager for Python. It allows you to install and manage additional libraries and tools that aren't included in the Python standard library. Think of it like an app store for Python packages — you can use it to quickly add functionality to your projects, like downloading datasets, building models, or scraping websites.

Normally, you would run `pip` commands in your computer's terminal or command prompt — for example, by typing `pip install package-name` into a command line window. However, since we're using **Google Colab**, we can run shell commands directly in notebook cells by adding an exclamation mark (`!`) at the beginning. This tells Colab to treat the line as a terminal command instead of Python code.

In the cell below, we're using `pip` to install the `google-play-scraper` package so we can fetch app reviews from the Google Play Store. This step ensures the library is available in our Colab environment.

```python
!pip install google-play-scraper
```

### Import Google Play Scraper

After installing a library with `pip`, we still need to **import** it into our Python environment to use its functions. The line `import google_play_scraper` makes the package available in our code so we can start using it.

```python
import google_play_scraper
```

### Download Reviews

The code below uses the `reviews_all()` function from the `google_play_scraper` library to download all available user reviews for a specific app. In this case, we're fetching reviews for the **TD Bank** mobile app, which has the app ID `'com.td'`.

We also specify:
- `lang='en'` to get reviews written in English (default is English)
- `country='ca'` to get reviews from the Canadian version of the Play Store (default is the U.S.)

The result is a list of dictionaries, where each dictionary contains details about a single review — including the review text, rating, date, and more.

```python
reviews = google_play_scraper.reviews_all(
    'com.td',
    lang='en', # defaults to 'en'
    country='ca', # defaults to 'us'
)
```

After downloading the reviews, we can use `len(reviews)` to see how many reviews were returned. This tells us the total number of reviews collected for the app.

```python
len(reviews)
```

To inspect the content, we can look at a few examples using slicing. For example, `reviews[0:3]` will display the first three reviews in the list. Each review is stored as a dictionary containing fields like the review text (`content`), rating (`score`), and timestamp (`at`).

```python
reviews[0:3]
```

## Creating a Tokenizer

To analyze these reviews we need to tokenize them. Here we're creating a simple **bag-of-words tokenizer** that splits a text into individual word tokens. We’ll **standardize** all tokens by converting them to lowercase — so that “Great” and “great” are treated as the same word. 

We’re using the regular expression `\b\w+\b` to extract tokens:

- `\w+` matches one or more alphanumeric characters (letters, numbers, and underscores)
- `\b` marks a **word boundary**, so this pattern finds full word-like chunks and skips punctuation, symbols, and whitespace

After tokenizing, we'll remove some common stop words. 

Our function `tokenize()` takes a string as input and returns a list of strings.

```python
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def tokenize(text):
    lowercase_text = text.lower()
    tokens = re.findall(r'\b\w+\b', lowercase_text)
    return [t for t in tokens if t not in ENGLISH_STOP_WORDS]
```

Let's try tokenizing a sample document.

```python
doc = "This app is incredibly frustrating. It keeps crashing every time I try to log in, and the new layout is confusing and slow. Terrible experience overall."
tokenize(doc)
```

## Sparse Vectors

As we discussed last class, when working with bag-of-words models, the next step after tokenization is to **vectorize** each document. You might think a list would be the most obvious way of storing a vector in Python.

But here’s the problem: most documents use only a small fraction of all the words in the full vocabulary. That means most of the values in these vectors are **zeros**.

This kind of structure is called a **sparse vector** — a vector where most of the entries are zero.

To save space and improve efficiency, we can represent sparse vectors using **dictionaries** in Python. Instead of storing every position, we only store the positions (words) that have non-zero values.

## Binary Vectorization

The `vectorize_binary` function creates a **binary (multi-hot) vector** from a list of tokens. This means it returns a dictionary where:

- Each **unique token** becomes a key
- The value is always `1`, indicating that the token is **present** in the document

This approach ignores how many times a word appears — it only cares **whether or not** the word is there. It's useful when we want to know which words are included but don’t need their exact frequency.

We use a Python `set` to remove duplicates, then build a dictionary where every token maps to the value `1`.

```python
def vectorize_binary(tokens):
    """
    Takes a list of tokens and returns a dictionary representing
    a binary (multi-hot) vector — 1 if the word appears, 0 if not (implied).
    """
    return {token: 1 for token in set(tokens)}
```

Let's try an example:

```python
doc = "This app is incredibly frustrating. It keeps crashing every time I try to log in, and the new layout is confusing and slow. Terrible experience overall."
tokens = tokenize(doc)
vectorize_binary(tokens)
```

## TF Vectorization

The `vectorize_tf` function creates a **term frequency (TF) vector** from a list of tokens. This means it returns a dictionary where:

- Each **unique token** becomes a key
- The value is the **number of times** that token appears in the document

Unlike binary (multi-hot) encoding, this approach **captures how often** each word occurs, which can give more nuanced insight into the text — especially when certain words are repeated for emphasis or importance.

We build this dictionary by looping through the tokens and counting how many times each one appears.

```python
def vectorize_tf(tokens):
    """
    Takes a list of tokens and returns a dictionary representing
    term frequency (TF) — how many times each token appears.
    """
    tf = {}
    for token in tokens:
        if token in tf:
            tf[token] += 1
        else:
            tf[token] = 1
    return tf
```

Let's try an example:

```python
doc = "This app is incredibly frustrating. It keeps crashing every time I try to log in, and the new layout is confusing and slow. Terrible experience overall."
tokens = tokenize(doc)
vectorize_tf(tokens)
```

## Calculating IDF

Before we can compute TF-IDF scores, we need to calculate **Inverse Document Frequency (IDF)** — a measure of how **rare or unique** a word is across all documents in the corpus.

Here’s what the code does step by step:

- `N = len(reviews)`  
  Counts the total number of documents (in this case, app reviews) in the corpus.

- We loop through each review and:
  - Use `tokenize()` to split the review into lowercase word tokens.
  - Convert the list of tokens into a `set` to get the **unique words** in that document.
  - For each unique word, we increase its **document frequency** — the number of documents in which the word appears.

- Finally, we calculate the **IDF score** for each token using the formula:  

  IDF = log(N / DF)
  
  Where:
  - `N` is the total number of documents
  - `DF(t)` is the number of documents containing token `t`

Words that appear in **fewer documents** get **higher IDF scores**, meaning they are considered more informative or unique. Common words that appear in many documents receive **lower scores**.


```python
from math import log

N = len(reviews) # total number of documents

doc_freq = {} # document frequency for each word
for review in reviews:
  if not review['content']:
    continue # some reviews do not have any content, skip these
  doc = tokenize(review['content'])
  unique_tokens = set(doc) # only count once per document
  for token in unique_tokens:
      doc_freq[token] = doc_freq.get(token, 0) + 1

idf_dict = {}
for token, df in doc_freq.items():
    idf_dict[token] = log(N / df)
```

Let's see what our `idf_dict` looks like:


```python
idf_dict
```

This code displays the tokens with the lowest and highest IDF:

```python
# Sort tokens by IDF score
sorted_idf = sorted(idf_dict.items(), key=lambda x: x[1])

# Lowest IDF scores (most common words)
print("🔽 5 Most Common Words (Lowest IDF):")
for token, score in sorted_idf[:5]:
    print(f"{token}: {score:.4f}")

# Highest IDF scores (most unique words)
print("\n🔼 5 Most Unique Words (Highest IDF):")
for token, score in sorted_idf[-5:]:
    print(f"{token}: {score:.4f}")

```

## TF-IDF Vectorization

The `vectorize_tf_idf` function creates a **TF-IDF vector** from a list of tokens. This means it returns a dictionary where:

- Each **unique token** becomes a key
- The value is the product of the token’s **term frequency (TF)** and its **inverse document frequency (IDF)**

TF-IDF scores reflect not just how often a word appears in a document, but also how **informative or distinctive** that word is across the entire corpus. Words that appear frequently in a document but rarely in others will have the **highest TF-IDF scores**.

We first count how many times each word appears (TF), then multiply that by its IDF value (from the `idf_dict`). Tokens that don’t appear in the IDF dictionary are ignored.

```python
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

Let's try an example:

```python
doc = "This app is incredibly frustrating. It keeps crashing every time I try to log in, and the new layout is confusing and slow. Terrible experience overall."
tokens = tokenize(doc)
vectorize_tf_idf(tokens)
```

## Downloading a Lexicon

We’re going to analyze the app reviews' sentiment using a **lexicon-based approach**. That means we'll score each review based on the words it contains, using a predefined list of words (a *lexicon*) where each word has an associated sentiment score.

To do this, we’ll first download a **lexicon file** that I’ve provided for the course.

The command below uses `wget` to download the file from GitHub:

```python
!wget https://raw.githubusercontent.com/wd13ca/BAN200-Summer-2025/refs/heads/main/lexicon.txt
```

Note that this is not a Python command — it’s a shell command that we’re running inside Colab by prefixing it with an exclamation mark (!). This tells Colab to treat the line as if it were typed into a terminal.

The file lexicon.txt is a tab-delimited text file with two columns:

- The first column is a word (e.g., great, terrible, fast)
- The second column is the word’s sentiment score — a number that represents how positive or negative that word is

We’ll load this lexicon into Python and use it to calculate sentiment scores for app reviews.

The code below reads the `lexicon.txt` file and stores its contents in a Python dictionary called `lexicon`.

Here’s what each part does:

- `with open("lexicon.txt", "r") as file:`  
  Opens the file in read mode. The `with` statement ensures the file is properly closed after reading.

- `for line in file:`  
  Loops through each line in the file. Each line contains one word and its sentiment score.

- `line.strip().split('\t')`  
  Removes any extra whitespace (like newlines) and splits the line into two parts using the **tab character** (`\t`) as the separator.

- `lexicon[word] = float(score)`  
  Adds the word to the dictionary, with its sentiment score stored as a floating-point number.

Once this code runs, you’ll have a dictionary where each word maps to a numeric sentiment score — ready to use for scoring reviews.


```python
lexicon = {}

with open("lexicon.txt", "r") as file:
    for line in file:
        word, score = line.strip().split('\t')
        lexicon[word] = float(score)
```

Let's take a look at our `lexicon`:

```python
lexicon
```

## Sparse Dot Product

The `sparse_dot_product` function calculates the **dot product** of two sparse vectors, which are represented as Python dictionaries.

The dot product multiplies matching values from both vectors and sums the results. In our case:

- The keys of the dictionaries are the tokens
- The values are numbers like term frequency, TF-IDF scores, or sentiment weights

Only **tokens that appear in both vectors** contribute to the result — which makes this efficient for sparse data. We are assuming that if a key doesn't appear in a vector, it has a value of zero.

```python
def sparse_dot_product(vec1, vec2):
    """
    Computes the dot product of two sparse vectors (dicts).
    Only keys that appear in both vectors contribute to the result.
    """
    return sum(vec1[token] * vec2[token] for token in vec1 if token in vec2)
```

Let's take a look at an example:

```python
vec1 = {'great': 2.0, 'battery': 1.0, 'screen': 0.5}
vec2 = {'great': 1.0, 'battery': 1.0, 'performance': 0.3}

dot = sparse_dot_product(vec1, vec2)
print(dot)
```

## Calculating Sentiment Scores

To compute the **sentiment score** of a document, we can take the **dot product** of two sparse vectors:

1. A **document vector**, which represents the words in the document (e.g., using term frequency or TF-IDF)
2. A **sentiment lexicon vector**, which assigns a sentiment score (positive, negative, or neutral) to each word

Each word in the vocabulary contributes to the sentiment score based on how often it appears in the document and how emotionally charged it is.

How It Works:

- For each word in the document, multiply its **frequency** (or weight) by its **sentiment score** from the lexicon.
- Add up all those products — that’s the total sentiment score.

This is exactly what a **dot product** does:

`sentiment_score = dot_product(document_vector, sentiment_lexicon)`

Only words that appear in **both** the document and the lexicon contribute to the result, making this efficient and interpretable.

Interpretation:
- A **positive score** indicates positive sentiment
- A **negative score** indicates negative sentiment
- A score of **zero** (or near zero) is neutral or mixed

Let's take a look at an example:

```python
doc = "This app is incredibly frustrating. It keeps crashing every time I try to log in, and the new layout is confusing and slow. Terrible experience overall."
doc_tokens = tokenize(doc)
doc_vector = vectorize_tf_idf(doc_tokens)

sparse_dot_product(lexicon, doc_vector)

```

## Evaluating Our Lexicon Model on TD App Reviews

Each review from the Google Play Store includes a **star rating** from 1 to 5, given by the user.

We can use our **lexicon-based sentiment model** to calculate a sentiment score for each review, then compare it to the original star rating to see how well our model captures real-world sentiment.

Steps:

1. **Tokenize** the review text.
2. **Vectorize** it using term frequency.
3. **Calculate sentiment** using the dot product of the document vector and the sentiment lexicon.
4. Store the **sentiment score** alongside the original **star rating**.

This lets us analyze how well our model's predictions align with user opinions. For example, we’d expect most 5-star reviews to have high positive sentiment scores, and most 1-star reviews to have negative ones.

### Calculating Corpus Sentiment Scores

Below, we use our lexicon-based sentiment model to compute a **predicted sentiment score** for each review, and pair it with the **actual star rating** (1 to 5) provided by the user.

What the Code Does:

- We create an empty list called `results` to store pairs of `(sentiment_score, star_rating)`.
- We loop through each review in the `reviews` list.
  - If a review has no text (`content` is empty), we skip it.
  - We tokenize the review text to prepare it for analysis.
  - We compute a **term frequency (TF)** vector for the tokens.
  - We calculate the **sentiment score** by taking the dot product between the document vector and the lexicon.
  - We extract the original star rating from the review (`review['score']`).
  - We store both the predicted sentiment and the actual star rating as a tuple.

This prepares the data for further analysis — for example, comparing how well our sentiment scores match the user ratings or visualizing the relationship between them.

```python
# For each review, calculate sentiment score and store it with the star rating
results = []

for review in reviews:
  if not review['content']:
    continue # some reviews do not have any content, skip these
  tokens = tokenize(review['content'])
  tf = vectorize_tf(tokens)
  sentiment_score = sparse_dot_product(tf, lexicon)  # you could also use TF-IDF
  star_rating = review['score']  # assuming 'score' is the star rating (1 to 5)
  results.append((sentiment_score, star_rating))
```

Let's see what results looks like:

```python
results
```

### Evaluation

To evaluate how well our sentiment model reflects user feedback, we compute the **average sentiment score** for each star rating (from 1 to 5) — but we only include reviews where the model detected a **non-zero** sentiment score.

This helps us focus on reviews that contain clearly positive or negative language, filtering out neutral or ambiguous ones.

How the code works:

- We create a regular dictionary called `scores_by_rating` to group sentiment scores by star rating.
- As we loop through each `(sentiment_score, star_rating)` pair:
  - If the sentiment score is exactly `0`, we skip that review.
  - If the star rating is not already in the dictionary, we initialize it with an empty list.
  - We append the sentiment score to the appropriate list for that star rating.
- After grouping, we calculate:
  - The **average sentiment score** for each rating
  - The **number of reviews** that contributed to the average (`n`)

This gives us a way to compare the model’s predictions to real user ratings, based only on reviews with detectable sentiment.

```python
# Store sentiment scores grouped by star rating (excluding zero scores)
scores_by_rating = {}

for sentiment_score, star_rating in results:
    if sentiment_score == 0:
        continue  # skip neutral (zero) sentiment scores
    if star_rating not in scores_by_rating:
        scores_by_rating[star_rating] = []
    scores_by_rating[star_rating].append(sentiment_score)

# Calculate and print average sentiment and review count per rating
print("⭐ Average Sentiment Score by Star Rating (non-zero only):\n")
for rating in sorted(scores_by_rating):
    scores = scores_by_rating[rating]
    avg_score = sum(scores) / len(scores)
    count = len(scores)
    print(f"{rating} stars: {avg_score:.2f} (n = {count} reviews)")

```

## Cosine Similarity

Cosine similarity is a common way to measure how **similar** two documents are, based on the **angle between their vector representations**.

It ranges from:

- `1` = very similar (point in the same direction)
- `0` = no similarity (orthogonal vectors)
- `-1` = opposite directions (only applicable if vectors contain negative values)

We already have a `sparse_dot_product` function, so we can reuse it to calculate both:

- The **dot product** between the two vectors
- The **magnitude** (length) of each vector (by taking the square root of its dot product with itself)

We then use the cosine similarity formula:

`cosine_similarity = dot_product(vec1, vec2) / (||vec1|| * ||vec2||)`

This function safely handles edge cases where one or both vectors have zero magnitude by returning `0.0`.

```python
from math import sqrt

def cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two sparse vectors (dicts).
    """
    dot = sparse_dot_product(vec1, vec2)
    mag1 = sqrt(sparse_dot_product(vec1, vec1))
    mag2 = sqrt(sparse_dot_product(vec2, vec2))
    if mag1 == 0 or mag2 == 0:
        return 0.0  # Avoid division by zero
    return dot / (mag1 * mag2)
```

Example:

```python
# Example TD app reviews
doc1 = "I can't log in at all — the app just freezes on the login screen. This has been going on for days and it's really frustrating."
doc2 = "Every time I try to log in, the app either crashes or gets stuck. Super annoying. Please fix the login issues."
doc3 = "Depositing cheques with the app is super easy and quick. Really impressed with how smoothly it works!"

# Step 1: Tokenize each review
tokens1 = tokenize(doc1)
tokens2 = tokenize(doc2)
tokens3 = tokenize(doc3)

# Step 2: Build a corpus and compute IDF
corpus = [tokens1, tokens2, tokens3]

# Step 3: Vectorize each document using TF-IDF
vec1 = vectorize_tf_idf(tokens1)
vec2 = vectorize_tf_idf(tokens2)
vec3 = vectorize_tf_idf(tokens3)

# Step 4: Compute cosine similarities
sim_1_2 = cosine_similarity(vec1, vec2)
sim_1_3 = cosine_similarity(vec1, vec3)
sim_2_3 = cosine_similarity(vec2, vec3)

# Step 5: Display results
print("Cosine Similarity Between Reviews:")
print(f"Doc1 vs Doc2: {sim_1_2:.3f}")
print(f"Doc1 vs Doc3: {sim_1_3:.3f}")
print(f"Doc2 vs Doc3: {sim_2_3:.3f}")

```

## K-Means

We will now use **K-Means clustering** to group together reviews that talk about similar things — without needing any labels.

Businesses often receive thousands of open-ended reviews and support messages. Reading them all manually is impossible. **Clustering** helps by automatically organizing this unstructured text into groups based on content.

With K-Means, we can:

- Identify major themes in customer feedback (e.g., login issues, mobile deposit problems, customer service complaints)
- Spot emerging issues without having to predefine categories
- Prioritize what to fix or investigate based on how many reviews fall into each cluster

This is especially powerful early in analysis, when you're just exploring the data and don't yet know what patterns exist.

How It Works
1. Randomly choose **k** cluster centroids
2. Assign each review to the cluster with nearest centroid based on **cosine similarity**
3. Update cluster centroids based on the contents of each group
4. Repeat 2 and 3 until the groups stabilize

Each cluster should end up containing reviews that are **similar in meaning or topic** — helping us quickly understand what customers are talking about, even in large datasets.

### Calculating Centroids

The centroid of a cluster is the average or mean of the vectors in the cluster. 

Averaging (or taking the mean of) a list of vectors means **combining them into a single "average" vector** that represents the typical values across all of them.

More specifically, we take the average **in each dimension** — which means:
- For every word (or feature) that appears in the vectors, we add up all its values across the documents
- Then we divide that total by the number of vectors

If the word "login" has a TF-IDF score of 0.4 in one review and 0.6 in another, the average value for "login" in the centroid would be (0.4 + 0.6) / 2 = 0.5.

This creates a new vector that reflects the **overall importance of each word** in that group of documents — and gives us a meaningful "center" for each cluster.

Before performing k-means clustering, we need a function to calculate the mean of a list of vectors:

```python
def mean_vector(vectors):
    """Averages a list of sparse vectors."""
    summed = {}
    for vec in vectors:
        for key, value in vec.items():
            summed[key] = summed.get(key, 0) + value
    count = len(vectors)
    return {k: v / count for k, v in summed.items()}
```

### Corpus Vectors

To make our code run more efficiently in class, we're going to randomly select a subset of reviews for k-means. 

```python
import random
valid_reviews = [r for r in reviews if r.get('content')]
sampled_reviews = random.sample(valid_reviews, 5000)
```

For each of `sampled_reviews` we're going to add a tf-idf vector: 

```python
for i, review in enumerate(sampled_reviews):
		tf_idf = vectorize_tf_idf(tokenize(review['content']))
		sampled_reviews[i]['tf-idf'] = tf_idf 
```

### K-Means

This code implements a basic version of the **K-means clustering algorithm** using **cosine similarity** to assign documents (e.g., text vectors) to clusters. 

Here's a breakdown of the steps:

1. We begin by defining the number of clusters `k` and randomly selecting `k` documents as the **initial centroids**

2. For each document in sampled_reviews, compute the cosine similarity to each centroid and assign the document to the cluster with the highest similarity.

3. After all documents have been assigned to clusters, calculate the new centroids by computing the mean vector of each cluster.

4. If the centroids have not changed from the previous iteration, the algorithm converges and stops

```python
# Step 1: Randomly initialize centroids
k = 4
centroids = random.sample([r['tf-idf'] for r in sampled_reviews], k)

for iteration in range(100):
    print(f"Iteration {iteration+1}")

    # Step 2: Assign documents to closest centroid
    clusters = [[] for _ in range(k)]
    for doc in sampled_reviews:
        similarities = [cosine_similarity(doc['tf-idf'], centroid) for centroid in centroids]
        best_cluster = similarities.index(max(similarities))
        clusters[best_cluster].append(doc)

    # Step 3: Update centroids by averaging each cluster
    new_centroids = [mean_vector([review['tf-idf'] for review in cluster]) if cluster else centroids[i] for i, cluster in enumerate(clusters)]

    # Step 4: Check for convergence (no change in centroids)
    if new_centroids == centroids:
        print("Converged.")
        break
    centroids = new_centroids
```

## Analyzing the Clustering Results

After running the K-means algorithm, we have grouped our reviews into `k = 4` clusters. Let’s explore each cluster to better understand the characteristics of the grouped reviews.

### Cluster Size

The first step is to see how many reviews were assigned to each cluster:

```python
for i, cluster in enumerate(clusters):
    print(f"Cluster {i}: {len(cluster)} reviews")
```

This helps identify whether the clustering was balanced or if one cluster dominated (which could suggest overlapping content or poor separation).

### Sample Reviews from Each Cluster

Print a few sample reviews from each cluster to get a sense of what kinds of comments ended up together:

```python
for i, cluster in enumerate(clusters):
    print(f"\n=== Cluster {i} ===")
    for review in cluster[:10]:
        print("-", review['content'][:200])  # print first 200 characters
```

### Average Review Score for Each Cluster

Calculate the average /5 score for each cluster.

```python
for i, cluster in enumerate(clusters):
    scores = [review['score'] for review in cluster]
    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"Cluster {i} average rating: {avg_score:.2f}")
```

### Word Cloud

This code generates a word cloud for each cluster.

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

for i, centroid in enumerate(centroids):
    wc = WordCloud(width=400, height=200)
    wc.generate_from_frequencies(centroid)
    plt.figure()
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Centroid {i} Word Cloud")
    plt.show()
```

## Summary

In this lecture, we explored the full journey from **raw text reviews** to **meaningful insights**:

- We used `google-play-scraper` to collect real app review data.
- We built custom tokenizers and vectorization functions for Binary, TF, and TF-IDF models.
- We learned how to represent text as sparse vectors and use them in sentiment scoring.
- We evaluated a **lexicon-based sentiment model** by comparing predicted sentiment scores to real user ratings.
- We introduced **cosine similarity** to measure document similarity.
- We implemented **K-means clustering** from scratch to discover themes in user reviews.
- Finally, we visualized and interpreted clusters using word clouds and average ratings.

This workflow demonstrates how to turn unstructured customer feedback into actionable insights — a critical capability in business analytics.

## Exercises

Try the following exercises to reinforce this week’s concepts:

1. Try re-running the k-means clustering. How stable are the clusters?

2. Try repeating the analysis with varying numbers of clusters. What number gives the best results?

3. Repeat this analysis using a different app or even another dataset.

4. Think about how you would present these results to a business audience. What advice would you give them?

## Homework

- Read chapters 10 thru 15 of [Think Python](https://allendowney.github.io/ThinkPython/)
- Complete the Exercises above
- Get going on your Group Project