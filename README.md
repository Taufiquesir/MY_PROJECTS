# SMS Spam Classifier

## Overview

This project is a machine learning-based SMS Spam Classifier developed using Python, Scikit-Learn, NLTK, and Streamlit. The objective of the project is to classify incoming SMS messages as either **Spam** or **Not Spam (Ham)**.

The application performs text preprocessing, feature extraction using TF-IDF Vectorization, and classification using the Multinomial Naive Bayes algorithm. A simple Streamlit-based web interface is provided to allow users to interact with the model and test messages in real time.

---

## Motivation

Spam messages are a common problem in digital communication. They often contain advertisements, fraudulent offers, phishing links, or misleading information. The goal of this project was to understand the complete machine learning workflow, starting from data preprocessing and feature engineering to model training and deployment.

This project was also built to gain practical experience with:

* Natural Language Processing (NLP)
* Text preprocessing techniques
* Feature extraction using TF-IDF
* Machine Learning model training and evaluation
* Model serialization using Pickle
* Web application deployment using Streamlit

---

## Dataset

The project uses the SMS Spam Collection Dataset containing labeled SMS messages.

Each message belongs to one of the following categories:

* Spam
* Ham (Not Spam)

Example:

| Message                                      | Label |
| -------------------------------------------- | ----- |
| Congratulations! You have won a free ticket. | Spam  |
| Are we meeting tomorrow at 10 AM?            | Ham   |

---

## Project Workflow

### 1. Data Cleaning

The dataset was cleaned before training the model.

Steps performed:

* Removed unnecessary columns
* Removed duplicate records
* Handled missing values
* Converted labels into numerical format

---

### 2. Exploratory Data Analysis

Several analyses were performed to understand the dataset:

* Distribution of spam and ham messages
* Number of characters in messages
* Number of words in messages
* Number of sentences in messages

This helped in understanding the characteristics of spam messages compared to legitimate messages.

---

### 3. Text Preprocessing

The preprocessing pipeline includes:

#### Lowercase Conversion

All text is converted to lowercase.

Example:

```text
HELLO WORLD
```

becomes

```text
hello world
```

#### Tokenization

Messages are split into individual words.

#### Removing Special Characters

Only meaningful alphanumeric words are retained.

#### Stopword Removal

Common words such as:

```text
the
is
and
of
```

are removed.

#### Stemming

Words are reduced to their root form.

Example:

```text
playing → play
played → play
plays → play
```

---

### 4. Feature Engineering

TF-IDF Vectorization is used to convert text into numerical features.

TF-IDF helps identify words that are important within a message while reducing the impact of frequently occurring words.

---

### 5. Model Training

Several machine learning algorithms were experimented with, and Multinomial Naive Bayes was selected due to its strong performance on text classification tasks.

Algorithm Used:

* Multinomial Naive Bayes

---

### 6. Model Serialization

The trained model and vectorizer were stored using Pickle.

Files generated:

```text
model.pkl
vectorizer.pkl
```

These files are loaded directly by the Streamlit application for prediction.

---

## Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* NLTK
* Scikit-Learn
* Streamlit
* Pickle

### Development Environment

* Ubuntu 24.04
* PyCharm
* Jupyter Notebook

---

## Project Structure

```text
SMS_SPAM_CLASSIFIER/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── spam.csv
├── requirements.txt
├── Procfile
├── setup.sh
├── sms_spam_detection.ipynb
└── README.md
```

---

## Running the Project Locally

### Clone Repository

```bash
git clone <repository-url>
cd SMS_SPAM_CLASSIFIER
```

### Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## Sample Predictions

### Example 1

Input:

```text
Congratulations! You have won ₹50000. Click here to claim your reward.
```

Output:

```text
Spam
```

### Example 2

Input:

```text
Hi, can we meet tomorrow regarding the project discussion?
```

Output:

```text
Not Spam
```

---

## Future Improvements

Some possible improvements for future versions include:

* Deep Learning based classification using LSTM or Transformers
* Email spam detection
* Multilingual spam detection
* Model monitoring and logging
* Deployment using Docker and Cloud Platforms

---

## Author

**MD Taufique Hussain**

M.Tech in Computer Science and Information Security (CSIS)
IIIT Hyderabad

Interests:

* Machine Learning
* Deep Learning
* Natural Language Processing
* Information Security
* Software Development

---

## License

This project is developed for educational and learning purposes.
