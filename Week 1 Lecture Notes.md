# Week 1 Lecture Notes

Welcome to Week 1 of BAN200: Text Mining and Sentiment Analysis. This week, we’ll introduce the core ideas behind text mining, explore its real-world applications, and set up the tools we'll be using throughout the course. You’ll also begin learning Python through the book *Think Python* and start thinking about your group project.

## What is Text Mining?

Text mining is the process of extracting meaningful information and patterns from unstructured text data using techniques from natural language processing, machine learning, and statistics.

## What is Sentiment Analysis?

Sentiment analysis is a type of text mining that identifies and categorizes opinions or emotions expressed in text, typically to determine whether the sentiment is positive, negative, or neutral.

## Common Applications of Text Mining

Text mining can be applied in many ways depending on the business problem. Below are key types of applications, each with a real-world example.

### Sentiment Analysis
Identifies the emotional tone or opinion expressed in text.

**Example**: Analyzing Twitter posts to determine whether customers are happy or frustrated with a recent product launch.

### Text Classification
Assigns each document to one or more **predefined categories** based on its content.

**Example**: Automatically tagging incoming customer emails as "technical issue", "sales inquiry", or "account question" so they can be routed to the right team.

### Clustering
Groups documents together based on **content similarity**, without predefined labels.

**Example**: Grouping thousands of customer complaints into clusters to discover patterns like "login issues", "payment problems", or "shipping delays".

### Topic Modeling
Identifies common **themes or topics** across a collection of documents, where each document may be associated with **multiple topics**.

**Example**: Analyzing product reviews to find that one review may mention both "battery life" and "camera quality", while another mentions "price" and "design".

### Named Entity Recognition (NER)
Detects and categorizes names of people, organizations, locations, dates, and other entities.

**Example**: Extracting company names and monetary values from financial news articles.

### Information Extraction
Pulls specific facts or details from text to create structured data.

**Example**: Extracting a list of job titles and required skills from online job postings.

### Text Summarization
Produces a short summary that captures the key ideas in a longer document.

**Example**: Summarizing customer survey responses into a few bullet points for executive reports.

### Keyword Extraction
Identifies the most relevant or frequently mentioned words or phrases in a text.

**Example**: Finding the top keywords from thousands of open-ended survey comments.

### Relationship Extraction
Identifies connections or relationships between entities mentioned in text.

**Example**: Detecting that "Apple acquired Beats" indicates a corporate acquisition.

### Language Detection and Translation
Determines the language of a text and/or converts it to another language.

**Example**: Automatically translating customer reviews from Spanish to English for analysis by an English-speaking team.

## Types and Sources of Text Data

Text data comes in many forms and is generated across nearly every business function. Understanding where it comes from and how it behaves is essential before we can begin analyzing it.

### Structured vs. Unstructured Data

- **Structured data**: Data that fits neatly into tables or databases (e.g., sales numbers, dates, inventory counts).
- **Unstructured data**: Data that does not follow a fixed format or schema — includes most text data (e.g., product reviews, emails).
- Text mining focuses on **unstructured** (and sometimes semi-structured) data.

### Common Sources of Text Data in Business

- **Customer reviews**  
  Feedback from platforms like Amazon, Yelp, Google Reviews.

- **Social media posts**  
  Tweets, Facebook comments, LinkedIn posts, and other public sentiment streams.

- **Customer service communications**  
  Chat transcripts, support tickets, help desk emails, and call center notes.

- **Open-ended survey responses**  
  Free-text fields in customer or employee surveys that express opinions in the respondent's own words.

- **Internal documents**  
  Reports, meeting notes, policy manuals, or memos within an organization.

- **News and press releases**  
  Articles and corporate statements that reflect external views or company messaging.

- **Web content**  
  Blog posts, FAQs, product descriptions, and user forums.

### Characteristics of Text Data

- **High volume**  
  Text data is generated constantly and often in large quantities.

- **Informal and messy**  
  May include slang, typos, emojis, inconsistent punctuation, or abbreviations.

- **Context-dependent**  
  Words may carry different meanings depending on the domain or usage (e.g., “cold” in healthcare vs. weather).

- **Multilingual and diverse**  
  Businesses often deal with content in multiple languages and across different writing styles.

### Why Text Data Matters

- Contains **rich qualitative insights** that structured data cannot capture.
- Helps understand **customer motivations**, **employee concerns**, and **market trends**.
- Often underutilized — unlocking it can provide a **competitive advantage**.

## How Do We Analyze Text?

Text mining involves turning large volumes of unstructured language into structured insights. To do this, we rely on tools from the field of **artificial intelligence (AI)** — specifically, a branch called **natural language processing (NLP)**.

### NLP Models: AI for Language

**Natural Language Processing (NLP)** is a field of AI that focuses on enabling computers to understand and analyze human language.

In this course, we’ll use **NLP models** to help us process and extract meaning from text data.

These models can be built in different ways — primarily through:

- **Rule-based systems**
- **Machine learning–based systems**

### Rule-Based Systems

Rule-based systems use **hand-crafted rules** to analyze text.

Rules might include:

- If a sentence contains the word "refund", label it as a complaint.
- Count the number of times a positive or negative word appears.

These systems are easy to understand and quick to build, but:

- Can be rigid
- May not handle nuance or ambiguity well

They are useful for **simple or highly structured tasks**, especially when labeled data is limited.

### Machine Learning–Based Systems

Machine learning–based systems use **statistical methods** to learn patterns from data rather than relying on predefined rules.

Instead of telling the system *how* to analyze text, we show it examples, and it learns to make decisions based on patterns it detects in the data.

Machine learning approaches can handle **complex**, **ambiguous**, and **large-scale** text data that rule-based systems struggle with. They are built on **statistical foundations** — probabilities, correlations, and optimization — which allow them to generalize from examples. As businesses accumulate more text data, machine learning becomes increasingly essential for scalable and adaptable analysis.

There are two main types of machine learning approaches used in text mining:

- Supervised Learning
- Unsupervised Learning

### Supervised Learning

In supervised learning, the model is trained on labeled examples — that is, data where the correct answer is already known. The model learns the relationship between the text and the label so it can predict labels for new, unseen text.

**Example**: Training a model on customer reviews labeled as "positive" or "negative", so it can predict sentiment for new reviews.

- Relies on **statistical relationships** between features (like word frequencies) and outcomes (like categories or ratings)
- Commonly used for tasks like **classification**, **sentiment analysis**, and **spam detection**

### Unsupervised Learning

In unsupervised learning, the model looks for **patterns or structure** in text without using labeled examples. These models group or organize the text based on statistical similarities in content.

**Example**: Using topic modeling to uncover themes in a collection of open-ended survey responses, without knowing the topics ahead of time.

- Helps discover **natural groupings**, **hidden topics**, or **anomalies**
- Commonly used for tasks like **clustering**, **topic modeling**, and **keyword extraction**


## Python

We will be using Python in this course to **analyze text data, build sentiment models, and explore real-world applications of text mining**.

### What Is Python?

Python is a high-level programming language that is widely used in data science, artificial intelligence, and software development.

- Known for its **clear, readable syntax** — it looks closer to English than many other programming languages.
- General-purpose: can be used for everything from web development to automation to scientific computing.
- Open-source and supported by a large community, which means extensive documentation and free learning resources.

### Why Python for Text Mining and Sentiment Analysis?

Python has become the **standard language for data analytics and natural language processing (NLP)** for several reasons:

- **Rich ecosystem of libraries** for working with text:
- **Excellent support for machine learning and AI**, including both rule-based and statistical models
- **Easy to prototype and test ideas quickly**, which is ideal for iterative, exploratory work

Python allows us to go from raw text to insight **within a single environment** — from cleaning and transforming the text, to modeling, to visualizing results.

## Textbook: Think Python

Don’t worry if you’ve never written a line of code before — this course is designed to support beginners. To make sure everyone builds a solid foundation in Python, we’ll be reading [**Think Python**](https://allendowney.github.io/ThinkPython/) by Allen B. Downey during the first four weeks of the course.

This book is written specifically for people new to programming. It introduces key concepts gradually and emphasizes clear thinking, problem-solving, and practical examples — all essential skills for text mining.

### Reading Schedule

Each week, you’ll read a few short chapters. Try to follow this pace to stay in sync with the hands-on work we’ll be doing in class.

- **Week 1**: Chapters 1–5  
- **Week 2**: Chapters 6–10  
- **Week 3**: Chapters 11–15  
- **Week 4**: Chapters 16–19  

These chapters are short and readable — focus on understanding the main ideas and experimenting with small bits of code. We’ll reinforce the concepts in class and through applied exercises.


## Google Colab

We’ll be using [**Google Colab**](https://colab.research.google.com/) as our main coding environment in this course.

### What Is Google Colab?

Google Colab (short for *Colaboratory*) is a **free, cloud-based coding platform** that lets you write and run Python code directly in your web browser — no installation required.

- Developed by Google and based on the popular **Jupyter Notebook** interface
- Allows you to combine **code**, **text**, **outputs**, and **visualizations** in a single document
- Runs on Google’s servers, not your local machine — so it's fast, powerful, and requires no setup

### Why We're Using It

- **No installation or configuration needed** — just open a link and start coding
- Works on **any computer** with a browser (Windows, Mac, Chromebook, etc.)
- Comes with **Python and key libraries pre-installed**, including tools for:
- Files are saved to your **Google Drive**, so your work is always backed up and accessible

### How It's Similar to Jupyter Notebooks

If you've used Jupyter Notebooks before, Google Colab will feel very familiar:

- Code is organized into **cells**, which can be run independently
- You can mix code with **Markdown text** (for notes, explanations, and formatting)
- Visualizations and outputs appear **directly below the code** that generates them

### Getting Started

We'll walk through how to access and use Google Colab in class. All you need is:

- A **Google account**
- A modern **web browser**

## Assessments

The final grade for this course will be based on a combination of individual assessments and a team-based project. All components are designed to evaluate your understanding of course concepts and your ability to apply them in real-world text mining scenarios.

| Assessment             | Due Date   | Weight     |
|------------------------|------------|------------|
| Midterm Exam           | Week 8     | 17.5%      |
| Final Exam             | Week 14    | 17.5%      |
| Project Proposal       | Week 7     | 12.5%      |
| Project Milestone      | Week 12    | 17.5%      |
| Project Report         | Week 13    | 17.5%      |
| Project Presentation   | Week 13    | 17.5%      |

## Group Project

### Overview

The group project allows students to apply text mining and sentiment analysis techniques in a real-world business analytics context. Students will form their own groups, select a relevant text dataset, and perform in-depth analysis using methods learned in class. The goal is to generate insights that could inform business decisions, demonstrate technical competence, and effectively communicate findings.

### Code

You'll use Python throughout the term to complete your group project. You'll need to share your code with the Professor using a public GitHub repo.

### Group Formation

- Groups must consist of **1 to 6 students**.
- Groups are **self-formed**; you are responsible for organizing your team.
- Project scope and depth should **scale with group size**. Larger groups are expected to demonstrate more ambitious analysis and deliverables.

### Project Topic and Dataset

Each group must select a **text-based dataset** relevant to business or industry (e.g., customer reviews, social media data, product descriptions, news articles, earnings call transcripts, etc.).

*Places to find good datasets:*

- [NLP Datasets by niderhoff](https://github.com/niderhoff/nlp-datasets)
- [Awesome Public Datasets by awesomedata](https://github.com/awesomedata/awesome-public-datasets)
- [Stanford NLP Resources](https://nlp.stanford.edu/links/statnlp.html)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Google Dataset Search](https://datasetsearch.research.google.com/)
- [Hugging Face Datasets](https://huggingface.co/datasets)
- [data.world](https://data.world/)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
- [Registry of Open Data on AWS](https://registry.opendata.aws/)
- [Reddit Datasets Subreddit](https://www.reddit.com/r/datasets/)
- [DataHub.io](https://datahub.io/)
- [OpenML](https://www.openml.org/)
- [Zenodo](https://zenodo.org/)
- [Data.gov](https://data.gov/)
- [Microsoft Academic](https://www.microsoft.com/en-us/research/project/academic/)
- [Common Crawl](https://www.commoncrawl.org/)
- [NLTK Data](https://www.nltk.org/nltk_data/)
- [Corpus.tools](https://corpus.tools/)
- [Papers with Code Datasets](https://paperswithcode.com/datasets)

### Deliverables

**Project Proposal**

A 1-to-2-page document outlining:

  - Group members  
  - Project title  
  - Dataset description and source  
  - Business problem or question  
  - Planned methods and tools  

Each group must meet with the Professor to discuss their Proposal during Week 7. 

**Milestone**

A 2-to-3-page document outlining:

  - Group members  
  - Project title  
  - Dataset description and source  
  - Business problem or question  
  - Overview of work done so far  
  - Plan for remaining work  
  - Link to a public GitHub repo with your code  

Each group must meet with the Professor to discuss their Milestone during Week 12. 

**Report**

A 4-to-5-page report (plus appendix, if needed) containing:

  - Group members  
  - Project title  
  - Dataset description and source  
  - Business problem or question  
  - Methodology  
  - Results and evaluation  
  - Discussion of challenges, limitations, and potential improvements  
  - Link to a public GitHub repo with your code  

Reports are due in Week 13.

**Presentation**

A 10-minute presentation reviewing the content of your Report. Presentations can be made in-person, online, or via recording. Presentations are due in Week 13.

## Lectures and Class Time

This course is designed to be flexible, practical, and interactive.

- **Lectures will be kept short and focused**, typically covering core concepts or introducing new tools and techniques.
- The rest of the class time will be used for:
  - Hands-on exercises and coding practice
  - Asking questions and getting help
  - Working on your **group project**
  - Collaborating with peers

### Attendance and Access

The course is offered **in three modes**:

  - **In-person**
  - **Live online via Zoom**
  - **Recorded sessions**, available shortly after each class

**Attendance is not mandatory**, but students are responsible for keeping up with readings, exercises, and assessments.

You are encouraged to attend live (in-person or online) whenever possible so you can ask questions, get support, and engage with your classmates and instructor.

## Questions or Concerns

Feel free to email me at william.dick[at]senecapolytechnic.ca or speak to me before/during/after class.  

## Homework

- Read the first five chapters of [Think Python](https://allendowney.github.io/ThinkPython/)
- Start thinking about your Group Project
