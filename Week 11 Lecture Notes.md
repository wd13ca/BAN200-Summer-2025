# Week 11 Lecture Notes

This week, we’ll explore how principles like **honesty**, **fairness**, and **inclusion** impact the **data collection process** — and why these values are essential for making better decisions in text mining and machine learning. We'll focus on real-world implications, examine common sources of bias, and look at strategies to collect data in a more responsible and equitable way.

## Why Data Collection Matters

Before we ever build a model or analyze results, we make critical decisions about **what data to collect**, **how to collect it**, and **from whom**. These decisions can introduce **bias**, **distort outcomes**, or **exclude key perspectives** — often without us realizing it.

In earlier weeks, we saw that text mining techniques like sentiment analysis, topic modeling, and classification all depend on having access to relevant and accurate data. But just as a model can amplify patterns in data, it can also amplify **flaws** if the data is biased or incomplete.

> 🧠 **Key Insight**:
> The quality and fairness of your analysis is only as good as the data it’s built on.

## Dimensions of Ethical Data Collection

To make sure our data reflects the real world in a meaningful and inclusive way, we must pay attention to three core principles:

### 1. Honesty

**Honest data collection** means being transparent about:

* **What** data is being collected
* **Why** it’s being collected
* **How** it will be used

Lack of transparency can lead to **distrust**, **poor consent**, or **invalid results**. Participants, users, and customers may not understand what they are contributing or how their data will influence business decisions or product development.

**Example**: If users believe their feedback will improve product design but it’s actually used to target ads, that’s a breakdown in honest data practices.

Honesty also includes accurately **representing the data**. This means avoiding cherry-picking results, misleading visualizations, or misrepresenting how data was gathered. Models and insights are only useful if they reflect what really happened.

### 2. Fairness

**Fairness in data collection** is about avoiding **systematic bias** that favors one group over another. It’s important to ask:

* Who is **included** in the dataset?
* Who is **missing** or underrepresented?
* Are some voices being **amplified** while others are ignored?

**Example**: A sentiment model trained only on English-language tweets may work poorly for users who tweet in Spanish or use culturally specific slang.

Bias can also enter when using pre-labeled datasets. For example, crowd-sourced labels may reflect cultural assumptions that don’t generalize to all users. Without checks, these biases can embed themselves in models and go undetected until they cause harm.

Fairness also includes **balancing the cost of errors**. A spam detector that wrongly flags business emails might inconvenience someone. But a hate speech detector that fails to flag abusive content can endanger vulnerable users.

### 3. Inclusion

**Inclusive data collection** ensures that a diversity of **perspectives**, **languages**, **experiences**, and **demographics** are represented.

This doesn’t mean collecting everything from everyone — but it does mean being intentional about **whose stories** and **voices** are included in the dataset. Inclusion is not just about representation; it’s also about power and access. Who is allowed to speak? Who gets to be heard?

**Example**: If you're analyzing customer reviews, consider whether you're also capturing feedback from older adults, people with disabilities, or users on low-end devices.

Inclusion helps prevent models from being optimized for only the "average" or "majority" user. It also encourages systems that work for edge cases, minority language speakers, or users with different needs.

## Common Sources of Bias

Even well-intentioned data collection can go wrong. Here are some common pitfalls:

### Selection Bias

Only collecting data from a **subset of people** that doesn’t represent the whole population.

**Example**: Only including online reviews written by power users, while ignoring occasional or dissatisfied users.

Selection bias can happen unintentionally, especially when data comes from convenience samples like survey responses or social media scraping. The people easiest to reach are not always the most representative.

### Labeling Bias

Inconsistent or biased labeling by humans or algorithms.

**Example**: If annotators assume all negative posts from young people are sarcastic, the sentiment labels will be skewed.

Labeling bias is particularly dangerous in supervised learning tasks. The labels become the "ground truth" for the model, but they may reflect the prejudices or misunderstandings of the people who created them.

### Measurement Bias

Systematically measuring something in a way that **distorts** the outcome.

**Example**: Using star ratings without understanding cultural differences in rating behavior (e.g., some people rarely give 5 stars).

Measurement bias can come from flawed survey design, inconsistent units, platform differences, or user interface quirks. It may not show up until the data is analyzed and strange patterns emerge.

### Missing Data

Some groups or topics may have **less data** — not because they’re unimportant, but because they’re harder to reach or more private.

**Example**: Text data from rural users may be sparse, but still reflect valuable needs or insights.

Missing data isn’t always avoidable, but being aware of it lets us make better decisions. Techniques like imputation, weighting, or stratified sampling can help mitigate gaps.

## Better Practices for Ethical Data Collection

Here are some strategies to collect more honest, fair, and inclusive data:

* **Be Transparent**: Document what data is collected, how, and why. Make this visible to stakeholders or participants.
* **Diversify Sources**: Pull data from multiple platforms, languages, or user groups where possible.
* **Check Representation**: Look at the distribution of age, gender, location, device type, etc. across your dataset.
* **Engage Stakeholders**: Include perspectives from people who will be affected by the model or analysis.
* **Validate Labels**: Use multiple annotators and measure agreement; consider participatory labeling where appropriate.
* **Audit Regularly**: Analyze your data pipeline for gaps, imbalance, or hidden assumptions — especially before modeling.
* **Respect Consent and Privacy**: Ensure that individuals understand how their data will be used, and avoid collecting more than necessary.
* **Test Impact Across Groups**: Evaluate model performance for different user segments to detect disparities in outcome.

## Why This Matters for Decision Making

Ultimately, our goal is to use text mining and machine learning to make **better decisions**. But poor data can lead to:

* **Flawed insights**
* **Biased models**
* **Broken trust**
* **Harmful outcomes**

Honest, fair, and inclusive data supports models that are more reliable, more generalizable, and more aligned with human values. In fields like healthcare, hiring, law enforcement, or finance, the stakes are especially high — bad models can reinforce inequality or cause real harm.

On the other hand, collecting honest, fair, and inclusive data helps ensure that our tools:

* Work well across different user groups
* Reflect real-world conditions
* Empower decision-makers with **more complete**, **reliable**, and **ethical insights**

> 🧠 **Final Thought**:
> If you want to build a fair model, start with fair data. The decisions you make at the data collection stage shape everything that follows.

- **What** data is being collected  
- **Why** it’s being collected  
- **How** it will be used

Lack of transparency can lead to **distrust**, **poor consent**, or **invalid results**.

**Example**: If users believe their feedback will improve product design but it’s actually used to target ads, that’s a breakdown in honest data practices.

Honesty also means accurately **representing the data** — avoiding cherry-picking, mislabeling, or over-interpreting patterns that aren’t really there.

### 2. Fairness

**Fairness in data collection** is about avoiding **systematic bias** that favors one group over another. It’s important to ask:

- Who is **included** in the dataset?
- Who is **missing** or underrepresented?
- Are some voices being **amplified** while others are ignored?

**Example**: A sentiment model trained only on English-language tweets may work poorly for users who tweet in Spanish or use culturally specific slang.

Unfair data leads to **unfair outcomes** — models that misclassify, exclude, or reinforce stereotypes.

### 3. Inclusion

**Inclusive data collection** ensures that a diversity of **perspectives**, **languages**, **experiences**, and **demographics** are represented.

This doesn’t mean collecting everything from everyone — but it does mean being intentional about **whose stories** and **voices** are included in the dataset.

**Example**: If you're analyzing customer reviews, consider whether you're also capturing feedback from older adults, people with disabilities, or users on low-end devices.

Inclusion supports **better decisions** because it captures a **wider range of needs**, **pain points**, and **use cases** — especially for people who might otherwise be overlooked.

## Common Sources of Bias

Even well-intentioned data collection can go wrong. Here are some common pitfalls:

### Selection Bias

Only collecting data from a **subset of people** that doesn’t represent the whole population.

**Example**: Only including online reviews written by power users, while ignoring occasional or dissatisfied users.

### Labeling Bias

Inconsistent or biased labeling by humans or algorithms.

**Example**: If annotators assume all negative posts from young people are sarcastic, the sentiment labels will be skewed.

### Measurement Bias

Systematically measuring something in a way that **distorts** the outcome.

**Example**: Using star ratings without understanding cultural differences in rating behavior (e.g., some people rarely give 5 stars).

### Missing Data

Some groups or topics may have **less data** — not because they’re unimportant, but because they’re harder to reach or more private.

**Example**: Text data from rural users may be sparse, but still reflect valuable needs or insights.

## Better Practices for Ethical Data Collection

Here are some strategies to collect more honest, fair, and inclusive data:

- **Be Transparent**: Document what data is collected, how, and why. Make this visible to stakeholders or participants.
- **Diversify Sources**: Pull data from multiple platforms, languages, or user groups where possible.
- **Check Representation**: Look at the distribution of age, gender, location, device type, etc. across your dataset.
- **Engage Stakeholders**: Include perspectives from people who will be affected by the model or analysis.
- **Validate Labels**: Use multiple annotators and measure agreement; consider participatory labeling where appropriate.
- **Audit Regularly**: Analyze your data pipeline for gaps, imbalance, or hidden assumptions — especially before modeling.

## Why This Matters for Decision Making

Ultimately, our goal is to use text mining and machine learning to make **better decisions**. But poor data can lead to:

- **Flawed insights**
- **Biased models**
- **Broken trust**
- **Harmful outcomes**

On the other hand, collecting honest, fair, and inclusive data helps ensure that our tools:

- Work well across different user groups
- Reflect real-world conditions
- Empower decision-makers with **more complete**, **reliable**, and **ethical insights**

> 🧠 **Final Thought**:  
> If you want to build a fair model, start with fair data. The decisions you make at the data collection stage shape everything that follows.


## Group Project Reminder

Project Milestones are due next week. Class time will be used for groups to meet with the Professor to discuss their Project Milestones. 