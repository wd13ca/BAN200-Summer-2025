# Week 2 Lecture Notes

This week, we begin the process of turning raw text into something we can analyze. We’ll explore how to break documents into pieces, represent them numerically, and uncover patterns in large text collections. Along the way, we’ll cover foundational techniques like tokenization, vectorization, sentiment scoring, information retrieval, and clustering — all building toward more advanced modeling in future weeks.


## Documents and Corpora

In text mining, a **document** refers to any individual unit of text that we want to analyze. A document can be:

- A single tweet
- A product review
- An email
- A paragraph from a news article
- A transcript of a phone call

Essentially, a document is **the smallest unit of analysis** in our pipeline. We might analyze it for sentiment, extract information from it, or compare it to other documents.

> 📌 Example:  
> - One Amazon review = one document  
> - One customer support ticket = one document  

A **corpus** (plural: *corpora*) is a **collection of documents**. It forms the dataset we use for text mining tasks.

> 📌 Example:  
> - All Amazon reviews for a product = a corpus  
> - A month’s worth of customer support tickets = a corpus

We often analyze patterns **across** the corpus — such as discovering frequent terms, building sentiment models, or detecting clusters of related documents.

### Summary


| Term       | Definition                                   | Example                                         |
|------------|----------------------------------------------|-------------------------------------------------|
| Document   | An individual unit of text                   | A tweet, email, review, or article              |
| Corpus     | A collection of documents                    | All reviews for a product, all support tickets  |

## Tokenization and Text Preprocessing

Before we can analyze text, we need to break it down into smaller pieces that a computer can work with. This process — known as **tokenization** — is one of the most important foundational steps in text mining. It’s often followed by **preprocessing** steps to clean and normalize the text.

### What is Tokenization?

**Tokenization** is the process of splitting text into meaningful units called **tokens**. Tokens are typically words, but they can also be phrases, sentences, or even characters depending on the application.

> 📌 Example:
> "The battery life is great!"  
> → ["The", "battery", "life", "is", "great", "!"]

Tokenization transforms raw text into a form that we can analyze — for example, by counting word frequencies or converting text into numerical features.

### Types of Tokenization

- **Word Tokenization**: Splits text into words and punctuation marks.  
- **Sentence Tokenization**: Splits text into individual sentences.  
- **Character Tokenization**: Splits text into individual characters.  
- **Subword Tokenization**: Breaks rare or unknown words into meaningful components (used in deep learning).

### Challenges in Tokenization

Tokenization can be tricky due to:

- **Punctuation** (e.g., "great!" vs. "great")
- **Contractions** (e.g., "don't" → "do" and "n't"? Or leave it as "don't"?)
- **Emojis, hashtags, and symbols** (e.g., 😊 or #winning)
- **Language-specific rules** (e.g., compound words in German or no spaces in Chinese)

### Text Preprocessing

Once text is tokenized, we often apply **preprocessing steps** to clean and standardize it. These steps help simplify the data and reduce variation.

#### Lowercasing

Converts all text to lowercase to ensure consistency.

> "Battery" and "battery" are treated as the same word.

This helps reduce redundancy in the vocabulary.

#### Stop Words and Stop Tokens

**Stop words** are common words that carry little meaningful content in many contexts, such as:

- "the", "is", "in", "of", "and", "a"

Removing stop words can reduce noise in the data and improve model efficiency, especially in tasks like topic modeling or keyword extraction.

> 📌 Without stop word removal:  
> ["the", "battery", "is", "great"]

> 📌 With stop word removal:  
> ["battery", "great"]

More generally, we can remove **stop tokens**, which may include:

- **Punctuation** (e.g., "!", "...")
- **Digits** (e.g., "123", "2024")
- **Special symbols** (e.g., "@", "#", "$")

However, **stop word removal is task-dependent**. For example, in sentiment analysis, words like "not" or "never" may be crucial and should be retained.

#### Stemming

**Stemming** reduces words to their root form by removing common suffixes.

> Examples:
> - "running", "runner", "runs" → "run"
> - "relational", "relations" → "relat"

Stemming is fast and simple but can be imprecise. It may result in non-standard words.

#### Lemmatization

**Lemmatization** also reduces words to a base form, but it uses grammar rules and a dictionary to do so more accurately.

> Examples:
> - "am", "are", "is" → "be"
> - "better" → "good"

Lemmatization is generally more accurate than stemming but also more computationally intensive.

Next, we’ll look at how we break text down into smaller components — a process called tokenization — and prepare it for analysis.

### Why Tokenization and Preprocessing Matter

- They prepare raw text for structured analysis.
- They reduce noise and variability in the data.
- They improve the performance and interpretability of models.

> 🧠 **Key Insight**:  
> The right combination of tokenization and preprocessing depends on your task. For example, stop word removal may help topic modeling but hurt sentiment analysis. Always consider the context.

## Sequence vs. Bag of Tokens

After tokenization, we must decide how to represent a document for analysis. One key choice is whether to treat the document as a **sequence** of tokens (preserving word order) or as an **unordered collection** (a “bag”) of tokens. This decision shapes the kinds of models we can use and the kinds of patterns we can detect.

In a **sequence-based representation**, we preserve the **order** in which words appear. This mirrors how language works for humans — where word order affects meaning.

In an **unordered collection-based representation** model (also known as a bag-of-tokens), we treat a document as a **set of tokens**, disregarding order. We only keep track of which tokens appear and how often. When using word tokens, we call this approach **bag-of-words (BoW)**.

> Example:  
> "Great food, bad service!" vs. "Great service, bad food!"  
> These sentences contain the same words, but the order gives them opposite meanings. In a bag-of-words representation, these sentences are equivalent. 

This trade-off between preserving word order and simplifying representation leads to important differences in **accuracy**, **efficiency**, and **interpretability**.

### Pros and Cons

| Approach               | Pros                                                                 | Cons                                                                 |
|------------------------|----------------------------------------------------------------------|----------------------------------------------------------------------|
| **Sequence-based**     | - Captures meaning, grammar, and context<br>- Preserves word order | - More complex to implement<br>- Requires more data and computation<br>- Less interpretable |
| **Bag-of-Tokens (BoW)**| - Simple and fast<br>- Easy to interpret<br>- Effective for many tasks<br>- Scales well to large corpora | - Ignores word order<br>- Cannot capture context or negation<br>- May lose subtle meaning |


There’s no single "best" approach — the right choice depends on your task, data, and goals. 

## Vectorizing Text

Once we’ve tokenized and preprocessed our text, the next step is to **convert it into numbers** so that it can be used in machine learning models. This process is called **vectorization**.

A vector is simply a list of numbers — and each document can be represented as a vector based on the words (or tokens) it contains. This allows us to compare documents, build classifiers, cluster similar items, and much more.

For the remainder of this lecture, we will focus on **Bag-of-Words (BoW)** models. These models represent a document by **what words appear**, but not the order in which they appear.

The vectorization methods we’ll explore — Multi-Hot, Term Frequency, and TF-IDF — **only make sense when we ignore word order**. Later in the course, we’ll explore more advanced **sequence-based** models that take context and word position into account.

We will cover three common ways to represent bag-of-words documents as vectors:

- Multi-Hot
- Term Frequency
- TF-IDF

### Example Corpus and Vocabulary

Let’s use the following simple corpus as an example:

> Document A: `"The battery is great and the screen is great too."`<br>
> Document B: `"Battery performance is poor."`<br>
> Document C: `"The screen is bright and clear."`

After tokenization and preprocessing (lowercasing, removing punctuation, removing stop words "the", "is", "and", and "too"), we get:

> Document A Tokens: `["battery", "great", "screen", "great"]`<br>
> Document B Tokens: `["battery", "performance", "poor"]`<br>
> Document C Tokens: `["screen", "bright", "clear"]`

Now we build a **vocabulary** — the set of all unique tokens:

> Vocabulary: `["battery", "great", "screen", "performance", "poor", "bright", "clear"]`

We’ll now represent this corpus using three different vectorization methods.

### Multi-Hot (Binary) Encoding

This is the simplest method. We create a vocabulary of all the unique tokens in our corpus, and represent each document as a binary vector:

- 1 if the word is present in the document
- 0 if it is not

> Document A Vector: `[1,1,1,0,0,0,0]`<br>
> Document B Vector: `[1,0,0,1,1,0,0]`<br>
> Document C Vector: `[0,0,1,0,0,1,1]`

### Term Frequency (TF)

Instead of just indicating presence or absence, **term frequency** counts how many times each word appears in a document.

> Document A Vector: `[1,2,1,0,0,0,0]`<br>
> Document B Vector: `[1,0,0,1,1,0,0]`<br>
> Document C Vector: `[0,0,1,0,0,1,1]`

### Term Frequency–Inverse Document Frequency (TF-IDF)

**TF-IDF** adjusts term frequency by how common the word is across all documents in the corpus. Words that appear in many documents get **lower weights**, while rare but potentially important words get **higher weights**.

- **Term Frequency (TF)** = How often the word appears in the document  
- **Inverse Document Frequency (IDF)** = How rare the word is across the entire corpus

The idea: a word is important if it appears **frequently in one document** but **rarely in others**.

Below is the IDF for each word in the vocabulary, based on the number of documents in which the word appears:

| Word      | Document Frequency (DF)  | IDF = log(N / DF)         |
|-----------|-----|----------------------------|
| battery     | 2   | log(3 / 2) ≈ 0.4         |
| great       | 1   | log(3 / 1) ≈ 1.1         |
| screen      | 2   | log(3 / 2) ≈ 0.4         |
| performance | 1   | log(3 / 1) ≈ 1.1         |
| poor        | 1   | log(3 / 1) ≈ 1.1         |
| bright      | 1   | log(3 / 1) ≈ 1.1         |
| clear       | 1   | log(3 / 1) ≈ 1.1         |

> Document A Vector: `[0.4,2.2,0.4,0.0,0.0,0.0,0.0]`<br>
> Document B Vector: `[0.4,0.0,0.0,1.1,1.1,0.0,0.0]`<br>
> Document C Vector: `[0.0,0.0,0.4,0.0,0.0,1.1,1.1]`<br>

Choosing the right vectorization method depends on the **structure of your data**, the **task at hand**, and the **importance of context**.

## Lexicon-Based Sentiment Models

One of the simplest and most interpretable approaches to sentiment analysis is the **lexicon-based model**. Rather than training a machine learning algorithm, we use a **predefined dictionary of words** (called a **sentiment lexicon**) where each word is assigned a sentiment score.

The idea:  

- Words like "great", "amazing", or "happy" contribute **positive sentiment**  
- Words like "bad", "terrible", or "angry" contribute **negative sentiment**

To calculate the sentiment of a document, we:

1. Represent the document as a vector of term frequencies.
2. Represent the sentiment lexicon as a vector of scores for the same vocabulary.
3. Compute the **dot product** of the two vectors.

The dot product gives us an **overall sentiment score** for the document by combining word frequency and word polarity.

### Example

Let’s reuse the preprocessed version of **Document A**:

> Document A Tokens: `["battery", "great", "screen", "great"]`  
> Vocabulary: `["battery", "great", "screen", "performance", "poor", "bright", "clear"]`  
> Term Frequency Vector: `[1, 2, 1, 0, 0, 0, 0]`

Now suppose we have a simple **sentiment lexicon** that assigns the following scores:

| Word        | Sentiment Score |
|-------------|------------------|
| battery     | 0                |
| great       | +1               |
| screen      | 0                |
| performance | 0                |
| poor        | –1               |
| bright      | 0                |
| clear       | 0                |

We can represent the lexicon as a **sentiment score vector**:

> Sentiment Lexicon Vector: `[0, 1, 0, 0, -1, 0, 0]`

We compute the dot product of the term frequency vector and the sentiment score vector:

> **Sentiment Score** =  
> `[1, 2, 1, 0, 0, 0, 0] · [0, 1, 0, 0, -1, 0, 0] = (1×0) + (2×1) + (1×0) + (0×0) + (0×-1) + (0×0) + (0×0) = 2`

So, the overall sentiment score of Document A is **+2**, which suggests a **positive** tone.

### Polarity Classification

We can convert the sentiment score into a **polarity label**:

- If the score > 0 → **Positive**
- If the score < 0 → **Negative**
- If the score = 0 → **Neutral**

> In this example:  
> Score = 2 → **Positive sentiment**

This simple rule helps categorize documents even without training data, making lexicon-based sentiment models useful in many business contexts.

### Why This Works

This model assumes that:

- The sentiment of a document is the **sum of the sentiment of its words**
- Words contribute **independently** to sentiment
- The more frequently a sentiment-laden word appears, the stronger its effect

This makes it:

- **Simple and transparent**
- **Easy to implement**
- **Fast to compute**

### Limitations of Lexicon-Based Models

- **Context is ignored**: “not great” may still score as positive
- **No domain adaptation**: Words may have different meanings in different industries
- **All words are treated equally** unless weighted otherwise

> 🧠 **Key Insight**:  
> Lexicon-based models are a great baseline. They’re interpretable and require no training data — but they often miss nuance. Later in the course, we’ll compare them with machine learning–based sentiment models that learn patterns from labeled examples.

## Information Retrieval with TF-IDF and Cosine Similarity

So far, we’ve seen how to represent documents as vectors using term frequency and TF-IDF. Now we’ll use these vectors to perform a common text mining task: **information retrieval** — finding the most relevant documents for a given query.

The key idea is to measure how **similar** two pieces of text are using their vector representations. The most common method is **cosine similarity**.

### Cosine Similarity

To understand cosine similarity, it helps to think of each document vector as a **line** pointing outward from the origin in a high-dimensional space. Even though we can’t visualize more than 3 dimensions, the math still works the same way. 

Cosine similarity measures the **angle** between two vectors. If two documents use similar words in similar proportions, the angle between them is small — and the cosine of that angle is close to **1**.

If the documents are completely different, their vectors point in very different directions — and the cosine is closer to **0**.

**Formula:**
cosine_similarity(A, B) = (A · B) / (||A|| * ||B||)

Where:
- `A · B` is the **dot product** of the two vectors
- `||A||` and `||B||` are the **magnitudes** of the vectors

### Example: Querying "battery performance"

Let’s use the same corpus and vocabulary from before:

> Vocabulary: `["battery", "great", "screen", "performance", "poor", "bright", "clear"]`

TF-IDF Vectors:

- Document A: `[0.4, 2.2, 0.4, 0.0, 0.0, 0.0, 0.0]`
- Document B: `[0.4, 0.0, 0.0, 1.1, 1.1, 0.0, 0.0]`
- Document C: `[0.0, 0.0, 0.4, 0.0, 0.0, 1.1, 1.1]`

Now let’s create a query:  
> `"battery performance"`

After preprocessing and tokenization, we get:
> Query tokens: `["battery", "performance"]`

We create a **TF vector** for the query based on term frequency:

- Query TF: `["battery": 1, "performance": 1]`

Convert this into a **TF-IDF vector** using the same IDF values:

| Word        | IDF   |
|-------------|--------|
| battery     | 0.4    |
| performance | 1.1    |

Query TF-IDF vector:  
> `[0.4, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0]`

### Step-by-Step Similarity Calculations

**Dot Products:**
- Query · Document A = `(0.4×0.4) + (1.1×0.0) = 0.16`
- Query · Document B = `(0.4×0.4) + (1.1×1.1) = 0.16 + 1.21 = 1.37`
- Query · Document C = `(0.4×0.0) + (1.1×0.0) = 0.0`

**Vector Magnitudes:**
- ||Query|| = sqrt(0.4² + 1.1²) ≈ sqrt(0.16 + 1.21) ≈ sqrt(1.37) ≈ 1.17
- ||A|| = sqrt(0.4² + 2.2² + 0.4²) ≈ sqrt(5.16) ≈ 2.27
- ||B|| = sqrt(0.4² + 1.1² + 1.1²) ≈ sqrt(2.58) ≈ 1.61
- ||C|| = sqrt(0.4² + 1.1² + 1.1²) ≈ same as B ≈ 1.61

**Cosine Similarities:**
- Query vs. A = 0.16 / (1.17 × 2.27) ≈ 0.06
- Query vs. B = 1.37 / (1.17 × 1.61) ≈ 0.73
- Query vs. C = 0.0 / (1.17 × 1.61) = 0.00

### Ranking Results

Based on cosine similarity, the documents are ranked as:

1. **Document B** (score ≈ 0.73) – highly relevant
2. **Document A** (score ≈ 0.06) – weakly related
3. **Document C** (score = 0.00) – unrelated

> The query "battery performance" most closely matches Document B, which contains both terms. Document A includes “battery” but not “performance,” so it’s only weakly similar. Document C shares no terms and has no similarity.

### Why This Matters

Cosine similarity allows us to:
- Find documents similar to a query
- Build a **basic search engine**
- Recommend documents, products, or articles based on text similarity

> 🧠 **Key Insight**:  
> Using TF-IDF and cosine similarity, we can retrieve relevant documents based on word content — even without training a machine learning model.

## Clustering Documents Using TF-IDF Vectors

Once we’ve vectorized a set of documents, we can group them based on their **similarity**. This process is called **clustering** — an unsupervised machine learning technique used to discover natural groupings in data.

In text mining, clustering allows us to:

- Identify common themes in unstructured data
- Group similar documents without predefined labels
- Summarize or organize large collections of text

### What Is Clustering?

**Clustering** is the task of assigning each item (in our case, a document) to one of several **groups or clusters** such that:

- Documents in the same cluster are **more similar** to each other than to documents in other clusters
- The algorithm does not require any labeled examples

This is different from classification, which requires labeled training data.

### Common Text Clustering Algorithms

Several algorithms can cluster text documents. The most common are:

- **K-Means**: Assigns each document to one of *k* clusters by minimizing the distance to the cluster's center.
- **Hierarchical Clustering**: Builds a tree-like structure showing how documents can be grouped at different levels.
- **DBSCAN**: Groups dense regions of documents and identifies outliers as noise.

Today, we’ll focus on **K-Means**, which works well with vectorized text and is easy to interpret.

### How K-Means Works (with a 2D Example)

To understand K-Means clustering, it helps to start with a simple example in **2D space**, where each item is a point with just two features.

Let’s say we have the following five data points:

- A = (1, 2)
- B = (1, 4)
- C = (2, 3)
- D = (8, 8)
- E = (9, 10)

Our goal is to group these points into **k = 2** clusters.

#### Step 1: Choose the number of clusters (**k**)

We decide that we want to divide the data into **2 clusters**.

#### Step 2: Initialize random cluster centers (centroids)

Let’s start by randomly picking two initial centroids:

- Centroid 1 = A = (1, 2)
- Centroid 2 = D = (8, 8)

#### Step 3: Assign each point to the nearest centroid

We compute the **Euclidean distance** from each point to both centroids:

| Point | Distance to Centroid 1 (1,2) | Distance to Centroid 2 (8,8) | Assigned Cluster |
|-------|------------------------------|------------------------------|------------------|
| A     | 0                            | ≈ 9.2                        | Cluster 1        |
| B     | 2                            | ≈ 7.2                        | Cluster 1        |
| C     | ≈ 1.4                        | ≈ 7.2                        | Cluster 1        |
| D     | ≈ 9.2                        | 0                            | Cluster 2        |
| E     | ≈ 11.4                       | ≈ 2.2                        | Cluster 2        |

#### Step 4: Update the centroids

Now that the points are assigned, we compute the new centroids by taking the **average (mean)** of all points in each cluster.

- New Centroid 1 (Cluster 1: A, B, C):
  - x = (1 + 1 + 2)/3 = 1.33
  - y = (2 + 4 + 3)/3 = 3.0
  - New Centroid 1 = (1.33, 3.0)

- New Centroid 2 (Cluster 2: D, E):
  - x = (8 + 9)/2 = 8.5
  - y = (8 + 10)/2 = 9.0
  - New Centroid 2 = (8.5, 9.0)

#### Step 5: Repeat steps 3 and 4

With the new centroids, we repeat the assignment and update steps. After a few iterations, the clusters stop changing — this is when we say the algorithm has **converged**.

### Summary: What K-Means Does

- K-Means tries to find **k centroids** that minimize the distance between each point and its assigned cluster center.
- It works by **alternating between assignment and update steps**.
- It converges when points stop switching clusters (or changes become very small).

### Applying This to Documents

In text mining, each document is a **vector in high-dimensional space**, where each dimension corresponds to a word (as defined by the vocabulary). When we cluster using TF-IDF vectors:

- Each document is a point in that space
- Similar documents are **closer together**
- K-Means groups them based on **shared vocabulary patterns**

We use **cosine similarity** (not Euclidean distance) when working with TF-IDF, because direction matters more than raw magnitude in text data.

### Why Clustering Is Useful

- Helps uncover **hidden structure** in text data
- Useful for **exploratory analysis** (e.g., grouping customer reviews)
- Scales to large document collections
- Can be used to **summarize**, **tag**, or **prioritize** documents

### Things to Keep in Mind

- The number of clusters (`k`) must be chosen manually, and different values may produce different results
- Results can vary depending on initialization
- TF-IDF + cosine distance is usually a good starting point for clustering documents, but preprocessing quality has a big impact

> 🧠 **Key Insight**:  
> Clustering is a powerful way to organize large volumes of text without labels. It helps you discover patterns, themes, and structure — especially when you don’t know what you’re looking for.

## Summary

This week, we explored the key building blocks for text analysis:

- **Documents and corpora** define the structure of our data
- **Tokenization and preprocessing** help clean and prepare raw text
- **Sequence vs. Bag-of-Words models** define how we represent language
- **Vectorization** turns text into numbers using techniques like TF, TF-IDF
- **Lexicon-based sentiment models** use predefined word scores to estimate document tone
- **Cosine similarity** allows us to measure document similarity
- **Clustering** helps us find groups and patterns in unlabeled text

These techniques form the foundation of many real-world text mining applications — from search engines and customer support systems to product review analysis and market research.

## Exercises

1. **Vocabulary Practice**
   - Take the following sentence:  
     `"The battery is terrible but the screen is amazing!"`  
     - Tokenize it (remove stop words and punctuation)
     - Apply stemming or lemmatization
     - Create a term frequency vector using the vocabulary from this week's example

2. **Sentiment Scoring**
   - Using the lexicon `[battery: 0, terrible: -1, screen: 0, amazing: +1]`, calculate the sentiment score of the sentence above using dot product.

3. **Cosine Similarity**
   - Use the TF-IDF vectors from earlier to compute cosine similarity between Document A and Document C. Interpret the result.

4. **Clustering Scenario**
   - Suppose you have a corpus of product reviews. How might clustering help a product team? What would be a useful value for *k*, and why?

5. **Reflection**
   - What are some advantages and disadvantages of using a bag-of-words model? When might it fail to capture important meaning?

## Homework

- Read chapters 6 thru 10 of [Think Python](https://allendowney.github.io/ThinkPython/)
- Complete the Exercises above
- Get going on your Group Project
