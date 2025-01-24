# EMAIL SPAM DETECTION WITH MACHINE LEARNING

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org)  [![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org) [![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) [![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org) [![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)  [![](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com) [![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io) [![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)



This project focuses on building a machine learning model to classify emails as spam or not spam. It leverages techniques such as Natural Language Processing (NLP) for feature extraction and uses algorithms like Naive Bayes or Logistic Regression to achieve accurate predictions. The solution includes exploratory data analysis, data preprocessing, and model evaluation for robust spam detection.

# Introduction:

Email spam detection is a critical task in cybersecurity, where the goal is to identify and filter out unsolicited and harmful messages, often referred to as "spam" emails. These emails are typically sent in bulk for purposes such as advertising, phishing, or spreading malware. The challenge lies in accurately distinguishing between legitimate emails (ham) and spam, especially as spam techniques evolve over time.

With the rise of email usage, the volume of spam messages has increased significantly, making it necessary to develop automated systems that can detect and filter these emails efficiently. Machine learning (ML) offers an effective solution to this problem by using algorithms that can learn patterns from historical email data and classify incoming messages as spam or non-spam (ham).

In this project, machine learning algorithms are employed to build a spam detection system. The dataset consists of labeled emails, where each email is marked as either spam or ham. Features such as the frequency of certain words, metadata (e.g., sender, subject), and email body content are extracted and used as input to train various ML models.

The goal of this project is to develop a robust, accurate, and efficient spam detection system that can automatically classify emails in real-time, providing users with a safer email experience and reducing the impact of spam on communication systems.

## About the Dataset

The provided dataset  **```spam.csv```** is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam.

This csv file contains one message per line. Each line is composed by two columns: v1 contains the label (ham or spam) and v2 contains the raw text. The corpus has been collected from free for research sources at the Internet.

Dataset Link:
```
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/data
```

## Problem Statement

Email is the most powerful tool in the world today, but it can also be a source of frustration due to endless spam emails. Businesses and organizations are facing a huge problem of abundance of spam Emails in their inboxes which are either promotional-based or fraudulent. But due to this the messages/Emails which are of much importance are squashed under the spam messages.

To tackle this issue, researchers have been developing techniques for spam mail detection, and one of the most promising approaches is using machine learning. In the field of natural language processing, there are numerous algorithms available for this kind of classification. Typically, spam emails contain a few recognisable terms that are pretty obvious indicators that the email is spam.

In this project, we will utilize Python to create a Spam Email Detection System which will help us to find whether an Email is a spam or not. We will follow the standard data science workflow, which includes data acquisition, exploring the data, preprocessing of the data using Natural Language Processing(NLP), text classification, feature engineering and applying the algorithms to find and compare the efficiency of several machine learning techniques such as KNN, Random Forest, Naive Bayes, SVM, and Logistic Regression, and more approaches.

## Objective

The primary goal of this project is to develop a predictive model that accurately recognises and categorizes incoming emails as either spam or ham by using machine learning algorithms to train the spam detector. By doing so, we aim to efficiently separate undesirable and potentially harmful emails from our inbox.

## Key Tasks

1. Data Collection: Gather data from the provided dataset.
2. Data Cleaning: Preparing the data for analysis by removing or modifying data that is incorrect, incomplete, irrelevant, duplicated, etc.
3. Exploratory Data Analysis (EDA): Analyzing and summarizing data to gain insights and understand its underlying patterns, relationships, and distributions.
4. Feature Engineering: Select and transform the most relevant variables from raw data.
5. Outlier Detection: Identifying and removing or handling outliers effectively to prevent them from biasing the model, reducing its performance, and hindering its interpretability.
6. Text Preprocessing: Cleaning and transforming unstructured text data to prepare it for analysis. It includes tokenization, stemming, lemmatization, stop-word removal, and vectorization.
7. Model Building: Create a machine learning model that accurately recognises and categorizes incoming emails as either spam or ham.
8. Model Evaluation: Assess the model's accuracy and performance using appropriate metrics.
9. Model Comparison: Compare the model's accuracy and choose the robust model for accurate Email Spam Detection model.

## Results
The following table represents the performance metrics for all the models trained. Performance metrics such as precision, Recall, F1-score, Trainset Accuracy and Testset Accuracy are evaluated on spam dataset to compare the model's accuracy and choose the robust model for accurate Email Spam Detection. I have selected Accuracy on Testset as the primary evaluation metric for the Email Spam Detection model. 

|          Model            | Precision |   Recall | F1 Score	| Accuracy on Trainset | Accuracy on Testset | Rank |
| :------------------------ | :-------: | :------: | :------: | :------------------: | :-----------------: | :--: |
| XGBClassifier             | 0.955752  | 0.892562 | 0.923077 | 0.988623	           | 0.982575            | 1    |
| SVM                       | 0.954128  | 0.859504 | 0.904348 | 0.986444             | 0.978703            | 2    |
| RandomForestClassifier    | 0.989583  | 0.785124 | 0.875576 | 0.999758	           | 0.973863            | 3    |
| BaggingClassifier         | 0.864000  | 0.892562 | 0.878049 | 0.999758             | 0.970958            | 4    |
| AdaBoostClassifier        | 0.945455  | 0.859504 | 0.900433 | 0.981118             | 0.977735            | 5    |
| NaiveBayes                | 0.988889  | 0.735537 | 0.843602 | 0.971435             | 0.968054            | 6    |
| LogisticRegression	      | 0.941748  | 0.801653 | 0.866071 | 0.966594             | 0.970958            | 7    |
| GradientBoostingClassifier| 0.946237  | 0.727273 | 0.822430 | 0.972162             | 0.963214            | 8    |
| KNeighborsClassifier      | 1.000000  | 0.314050 | 0.477987 | 0.922053             | 0.919652            | 9    |
| DecisionTreeClassifier    | 0.783019  | 0.685950 | 0.731278 | 0.957879             | 0.940949	           | 10   |

The **XGBoost classifier** accurately identified the email messages as spam or not spam with highest accuracy of 98.2% on test data and Precision of 95.5%, followed by **SVM** model with 97.8% accuracy on test data and Precision of 95.4%

## Conclusion

In this end-to-end project, we have learned how to approach a problem statement, and gather useful conclusions from the data using Data preprocessing, EDA with Data Visualisation which helped us to build a good Machine Learning Model.

This project provides an overview of using different techniques to classify an email as "spam" or "not". In order to solve this classification problem we used the several machine learning algorithmns and in particular, the XGBoost Classifier outperformed all other models as it was having the highest Rank and for the vectorization technique, we used TFIDF. TF-IDF is an information retrieval technique that weighs a term’s frequency (TF) and its inverse document frequency (IDF). Each word or term that occurs in the text has its respective TF and IDF score.

We propose the **XGBoost Classifier** for effective and efficient email spam filtering in this project. In addition, the algorithm’s efficacy and efficiency were evaluated using precision, Recall, F1-score, Trainset Accuracy and Testset Accuracy on spam dataset. We can conclude by saying that **XGBoost Classifier** and **SVM** Algorithm are the promising methods that may be used either at the mail server or at the mail client to reduce the number of spam messages in email user's inboxes.

With the right tools and techniques, it is possible to build highly effective spam mail detection systems using machine learning. The selection of the optimal model should consider factors beyond just accuracy, such as computational efficiency and the specific requirements of the application. It is advisable to perform further model fine-tuning and validation before making a final choice. By leveraging the power of these techniques, we can help protect individuals and organizations from the growing threat of spam and other email-based attacks.


# Feedback and Suggestions:

Thank you for visiting my repository! If you have any questions or feedback, feel free to reach out.

I’d love to hear your thoughts, feedback, and suggestions! Feel free to connect with me:

 LinkedIn: [Shivanand Nashi](https://www.linkedin.com/in/shivanand-s-nashi-79579821a)
 
 Email: shivanandnashi97@gmail.com


Looking forward to connecting and exchanging ideas!

# ✨ Support this project!
If you found this project helpful or interesting, please consider giving it a ⭐ on GitHub!
Your support helps keep the project active and encourages further development.

Thank you for your support! 💖





