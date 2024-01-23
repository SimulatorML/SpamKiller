# Design Doc (EN)

# **Introduction**

Spam is unwanted emails or mailing lists that can come to your address. They can contain advertising offers, computer viruses, or be phishing attempts. Spam messages in Telegram chats (hereinafter referred to as "chats") often contain advertising offers, less frequently viruses and phishing.
This document describes a project to create a Telegram bot (hereinafter referred to as "bot") that will automatically block users who leave spam messages (hereinafter referred to as "spam"). The document discusses the formulation of the project's tasks and objectives, defines indicators and describes methods of data collection, error analysis, and model integration stages.

<p align="center">
  <img src="https://github.com/maltsevd/SpamKiller/assets/54685997/7f57a7e0-33b2-463c-baf6-7aad0a1ebe50" width=20%>
</p>

# **Problem Statement and Project Objectives**

The problem of spam in large chats is common. Spam hinders communication between people, the search for necessary information, and ultimately can lead to people leaving the chat because it becomes impossible to be in it due to the abundance of spam.

By getting rid of spam in the chat, we can make communication between people more comfortable, because it is unpleasant to chat or search for information and often come across fraudulent information about discounts of up to 90%.

Overall, the goal of this project is to minimize the amount of spam, free up administrators from the routine task of checking the chat for spam, round-the-clock chat monitoring by the bot, timely decision-making about blocking the user, and removing spam.

# Preliminary Research

On average, a spam message in a chat appears 2-3 times a day, with a maximum recorded of 5 spam messages per day.

Two moderators handle the removal of spam messages.

# Metrics & Losses

### Online-metrics

- ***Reaction Time:** how quickly the bot removes spam (want < 1 sec);*
- ***Recall:** what proportion of spam is removed (want > 95%);*
- ***Specificity:** "true negative rate" (want > 99.9%).*

<p align="center">
<img src="https://github.com/maltsevd/SpamKiller/assets/54685997/01d0c470-4bcd-4274-b443-5b9247a77572" width=75%>
</p>

### Offline-metrics

- ***Recall @ Specificity > 99.9%:** the proportion of correctly detected spam given a cutoff that yields a false alarm rate of less than 1 in 1000.*
https://t.me/cryptovalerii/184
- ***ERR (Equal Error Rate):** how far the ROC curve point where FPR and FNR match.*
https://hal.science/hal-00674526/document
- ***ROC-AUC:** area under the ROC curve.*
https://en.wikipedia.org/wiki/Receiver_operating_characteristic
- ***PR-AUC:** area under the Precision-Recall curve.*
https://www.geeksforgeeks.org/precision-recall-curve-ml/

### Losses

- ***LogLoss:** a metric for assessing the predicted probability that a message is spam.*

### **Technical**

- ***Latency (0.99 quantile):** message processing speed in the worst-case scenario.*
- ***RAM (max):** maximum amount of RAM occupied by the ML service.*
- ***QPS:** queries-per-second, how many requests per second the bot processes.*

# **Data Collection**

Data for the training model was obtained by collecting spam messages from the target Telegram channel. This approach helped avoid data defocusing and focused on the type of spam more commonly encountered in the Telegram channel for which the bot is being created.

Data collection is done "manually," namely:

- Examples of spam messages are collected in a separate Telegram channel with subsequent extraction and processing for sending to the model. These messages will be assigned a label of 1 (positive). Example of a text message containing spam: "Due to the scandal, the World Bank has decided to apologize to clients and lower prices. Don't miss the chance to get any items for free - <channel link tg>"
- Examples of messages not containing spam are collected in a .json file. These messages were assigned a label of 0 (negative). Example of a text message not containing spam: "for such a <link> there is a cool Python library".
- There is automatic labeling already on the part of the working bot

The guarantee of the absence of spam messages in the data labeled 0 gives us confidence that the chat is moderated by an administrator who deletes spam, which highly likely guarantees that spam was removed and not uploaded. This applies to data labeled 1.
All messages with labels 0 and 1 are accepted for the same period – this ensures the same proportion in the data.

## Features

Currently, there are 13 rules as features:

1. **Checking for Telegram links presence**:
    - Only a Telegram link in the message: +0.3
    - Many Telegram links in the message: +0.3
    - One Telegram link in the message: +0.15
2. **Checking for stop words presence**:
    - The presence of each stop word: +0.30
3. **Checking for dangerous words presence**:
    - The presence of each dangerous word: +0.15
4. **Checking for spam words presence**:
    - The presence of each spam word: +0.5
5. **Checking for Cyrillic substitution**:
    - The presence of Cyrillic substitution: +0.1 for each word
6. **Checking for a photo presence**:
    - The presence of a photo in the message: +0.15
7. **Checking sender identifier for absence of spam**:
    - If **`from_id`** is not on the spam list: -0.5
8. **Checking for special characters presence**:
    - The presence of each prohibited character: +0.1
9. **Checking message length**:
    - Too short message: +0.1
10. **Checking for words not similar enough to others**:
    - The presence of each word from the list of words not similar enough to other words: +0.15
11. **Checking for uppercase letters presence**:
    - High concentration of uppercase letters: +0.15
12. **Checking for suspicious emojis presence**:
    - The presence of each suspicious emoji: +0.15
13. **Checking for links presence**:
    - Only a link in the message: +0.3
    - One link and text in the message: +0.15
    - Multiple links in the message: +0.3

All points are summed up and then normalized to a threshold of 1.0. If the sum of points equals or exceeds the threshold, the normalized score is set to 1.0. If the sum of points is less than zero, the normalized score is set to 0.0. In other cases, the normalized score equals the sum of points divided by the threshold value.

## Completeness and volume of data

Described in the **Error Analysis**

 section, under the **Sample Size Assessment** subsection.

# Data storage

## Unprocessed data

Text data, downloaded from two TG-chats (the first, where we collect spam, and the second, where the moderator regularly cleans our messages from spam), will be stored in .json format.

## Processed data

Processed (cleaned) data will be stored in csv tables.

## Cloud data storage

Data will be stored in .csv format tables, and version control will be done using cloud storage and DVC (Data Version Control). This will provide easy access to data and sharing it between team members, as well as register changes made to the data over time.

# **Validation**

Validation is an important step in model development, ensuring that the model performs well with new, unseen data. In our project, we will implement a validation scheme where the data set will be divided into two parts - a training set and a test set. On the first data set, we will train our model, and on the test set, we will evaluate our model's performance. These indicators will help us make decisions about model selection and parameter tuning, and also ensure that our model is reliable and performs well in real-world scenarios.

The training set will include 80%, and the test set will include 20% of the data. In the first iteration of the project, we will update the data once a week.

## Learning Curve

Assessing the sample size and studying the learning curve (Learning Curve) are important components in designing and evaluating machine learning models.

To optimally assess the size of the training sample, we need to build a Learning Curve.

The analysis algorithm is as follows:

1. Choose a metric to evaluate the quality of the model.
2. Then, split your data set into a training and test sample.
3. After that, start training the model on various subsets of the training sample. For example, you can first train the model on only 10% of the data, then on 20%, and so on, up to 100%.
4. At each stage, after training the model, you can check its quality on the test sample using the chosen metric.
5. As a result, we will have a set of pairs of values: the size of the training sample and the quality of the model. We can plot the dependence of these values, which will be the learning curve.

If the learning curve reaches a plateau, it means that further increasing the training data will not lead to a significant improvement in the quality of the model, and the sample size can be considered optimal.

# **Baselines Models**

## **List of models**

A three-tier implementation is proposed with gradual complication:

- a model based on heuristic rules
- Logistic regression/SVM with different kernels
- Isolated forest
- AutoEncoder

## **Model on heuristics**

As a basic version, for subsequent comparison with more complex options, a simple model on heuristics will be used.
The implementation is a simple series of binary questions. For example:

- does the message contain stop words?
- have there been messages from this user before?
- does the message contain a link in domain X?
- is there a picture in the message?
- the proportion of words in Latin and Cyrillic in the message is greater than X?
- the time it took for the user to leave the message is less than X?

Then these answers are summed up (with a certain weight) and divided by the total sum (for better interpretability). Upon passing the first threshold, they are sent to the administrator for further classification, upon passing the second threshold, they are automatically blocked. The thresholds are determined in accordance with the required results of the quality metric.
Implementation is considered possible due to the dominance of 3-4 classes of spam with their specificity and homogeneity within themselves.
Positive qualities of the model on heuristics:

- one of the fastest working speeds
- scalability
- interpretability of the result
- questions for classification can be learned from experts
- more complex models can be supplemented with answers from the basic model

Cons:

- have to rewrite the rules with the appearance of new classes, as well as with changes in existing ones
- does not detect rare manifestations of spam

## Logistic regression/SVM

Generalization of the model based on heuristic rules. Plus, if possible, attach a simple TF/IDF and calculate metrics.

## **Isolated forest**

Isolated forest can effectively detect abnormal or unusual patterns in data, including anomalies related to fraud. The algorithm builds decision trees, dividing data into different branches until each instance is completely isolated. Anomalous points usually require fewer splits to isolate them, so they have a shorter path length in the tree.

Features we will use:

- word embeddings. The neural network, which will process text messages (tokenize), will be retrained RuBert on our data. RuBert is best suited for our task, as it is important for us to preserve the context of the message, and not just the frequency and importance of the word, as it would be if we used TF–IDF
- answers from the rule-based model

The main plus of this implementation is the detection of rare spams by their anomalous dissimilarity. The downside is the complexity of implementation. It is also worth noting the expected overall quality improvement compared to previous models.

# **Error Analysis**

## **Sample Size Assessment**

For an approximate assessment of the sample size, we can plot the dependence of the error on our sample. Knowing the required level of quality, and having a small number of samples of different sizes, we can approximate the available data and find the necessary size with an acceptable level of quality.

<p align="center">
<img src="https://github.com/maltsevd/SpamKiller/assets/54685997/03c4dae4-9c25-4457-b972-a28ad120a5d1">
</p>

## **Checking the model for overfitting/underfitting**

Will be carried out using the classic graph of model complexity (number of iterations) from the result on test and train

<p align="center">
<img src="https://github.com/maltsevd/SpamKiller/assets/54685997/11247a97-c2f3-4544-bc59-b7279c471883">
</p>

## **Checking the model for convergence/divergence**

Used for iterative models. We build a graph of the decrease in error over time (number of iterations) while taking into account the possibility of overfitting the model. Another option is a graph of the size of the learning step over time (number of iterations).

<p align="center">
<img src="https://github.com/maltsevd/SpamKiller/assets/54685997/fd925038-e49b-4d22-aca6-80af3cd2ea9d">
</p>


## **Model Complexity and Depth Check**

In addition to checking for overfitting, this involves a simple graph showing the decrease in error as the model complexity increases. At some point, the graph is expected to plateau, indicating that further increasing the complexity of the model, and consequently its slower operation, will not be compensated by a reduction in error.

<p align="center">
<img src="https://github.com/maltsevd/SpamKiller/assets/54685997/19b0cb37-cb85-4f36-b0ac-3666e6062935">
</p>

## **Residuals**

For our binary classification task, the residuals will be the differences between the true probability of class 1 and the predicted probability.

## **Error Distribution Check for First and Second Class**

The distribution should tend towards normal. With a uniform distribution, we predict real data equally poorly. If there is a bias in the model, we can look towards increasing data or rebalancing it, as well as complicating the model. If the distribution is complex, there may be a pattern that our model cannot capture.

## **Error Pattern Analysis**

This will be done using analytical methods such as visualization, clustering, and correlation searches.

## **Incorporating Errors into the Training Pipeline**

Upon detecting frequent errors that are explicable, the model should be retrained with changes in the quantity and quality of the data.

# **Training Pipeline**

## **Heuristics-Based Model**

We collect a corpus of chat texts from the KC. Based on its analysis and using logic, we create rules that our model will follow. Experts in spam blocking can also assist in updating the rules.

We conduct validation on a held-out dataset. In the end, we save artifacts (metrics, loss curves, complex cases for classification).

<p align="center">
<img src="https://github.com/maltsevd/SpamKiller/assets/54685997/0a977dd5-25f5-42be-87d0-d47535373c3b">
</p>

## Logistic Regression/SVM

First, train the model on our heuristic rules and measure its quality.

Next:

For training, a word matrix will be created in the corpus of KC chat texts. We will remove unnecessary characters (emojis, etc.), convert all words to lowercase, and replace numbers in the text with the token <NUMER>.

To normalize the matrix, tf-idf transformation will be applied, allowing us to ignore less important words. To facilitate prediction, we will reduce the dimensionality of the existing matrix.

The linreg/SVM will be trained on the obtained matrix, followed by validation on held-out data.

In the end, collection of artifacts (metrics, loss curves, complex cases for classification) is implied.

<p align="center">
<img src="https://github.com/maltsevd/SpamKiller/assets/54685997/c415b1e6-fcf8-4e8a-ae9d-13b896812fcd">
</p>

## **Isolated Forest**

We upload the corpus of KC chat texts. We remove unnecessary characters (emojis, etc.), convert all words to lowercase, and replace numbers in the text with the token <NUMER>.

Using the RuBert neural network, we represent our texts in a compressed format that reflects our task well.

We also add responses from our rule-based model to the training dataset.

We train the isolated forest and conduct validation. In the end, collection of artifacts (metrics, loss curves, complex cases for classification) is implied.

<p align="center">
<img src="https://github.com/maltsevd/SpamKiller/assets/54685997/3e3e0ba5-fd01-4105-83b0-b656f022e876">
</p>

# **Inference Pipeline**

A written message is sent to the bot. Depending on the selected model, the bot either immediately sends the data to it (first-order model) or preprocesses the data (second and third-order models). Based on the prediction, the bot can take three actions:

- block the user
- send to experts (if in doubt about classification)
- ignore the message

<p align="center">
<img src="https://github.com/maltsevd/SpamKiller/assets/54685997/b090a69e-fe73-4549-bd22-81e370722b49">
</p>

# **Integration and Deployment**

An important stage in the bot's development is its subsequent use by end users – chat administrators.
The entire project will be packaged in Docker with the possibility of token replacement.

Next, the bot will be deployed on a VPS server.

# Protection from Unfair (Erroneous) Ban

This issue is extremely relevant because, in machine learning, we work with probabilities, and the probability of an unfair (erroneous) ban is not zero, so we must consider this and not give the bot absolute rights to ban users. If we do not consider this crucial factor, then, by releasing a bot with absolute power into production, we may face the situation where the bot may start banning unfairly, negatively affecting the reputation of the final product.

1. During the first n weeks, the ban is not introduced. If the bot sees spam, it sends this message to the administrator, and they decide whether to "delete/not delete".
2. If the chat administrator is satisfied with the bot's work (no or minimal number of unfair bans), a rule is introduced: a pause between issuing bans + if there are > 5 bans in 5 minutes, then the bot is turned off for an hour and notifies the administrator.

# What's Done?

As of 01.10.23, the following has been done:

- The bot operates in the chats [Karpov.courses](http://Karpov.courses)/Время Валеры/BOGDANISSIMO/Dimension. More than 35,000 users.
- Cleaning of the .json file with the history of messages from the KC chat and spam samples, and packaging into neat data frames.
- Monitoring set up: expert verification, division of messages into Spam/not SPAM.
- Rule-based model.
- Measurement of metrics.

To launch the bot, you need to create a token, insert it into .env, to start receiving spam notifications, you need to add your id to .env.
