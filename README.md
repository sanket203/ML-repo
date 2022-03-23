# Fake News Article Detection

## Setup:
  * There are two jupyter noteboos
    - clean_data.ipynb is for preprocessing and cleaning the train and test data. This file generates clean_train.csv and clean_test csv
    - train_models.ipynb is for training the model and predict on test data. This file uses previously generated clean_train.csv and clean_test.csv
  * train_models.ipynb generates output_submit.csv which contains required results for test data.

## Approach:
### Exploratory data analysis
  - Visualized the data inform of bar chart to check data imbalance with respect to categories. Data was balnced perfectly hence no need of oversampling
  - Checked if data contains any null values and replaced null values using pandas fillna() and dropna() functions with blank string as data was text
  - Data contains three features "Title", "Author" and "Text". As all three are strings it is better to concatenate as one string for better prediction.
### Clean the text data
  - First fix the encoding of the text as it might contain various characters such as quotes which are differently encoded. Hence converted the data to utf8 encoding and replaced such character with ascii characters.
  - Second, do the contractction expancsion. words such as "did't", "won't" need to be converted properly to "did not", "will not" etc.
  - Third, Lemmentization: it is important to lemantize the text so that variations of a single words will be considedre as one
  - Four, Remove stopwards such as "is","the" etc. which are not important and occure frequently in text. For this I have used NLTK stopward dictionary
  - Five, Clean any special character, html tags etc from the text
  - Six, convert the text to lowercase
### Vectorization
  - I have used "tfidf" vectorizer to convert the text into vector as it gives weightage to important words. Embeddings such as Word2Vec and glove can be used for better performance but due to time and resource limitations I have used tfidf.
  - Save the vetorized features to a file so that they can be used later for prediction on test data
### Train and Validation data split
  - Using sklearn's train_test_split() function split the data into training and validation sets
  - This is necessary to calculate metrics against unseen data which is validation data
  - I have split the data as 80% for training and 20% for validation as x_test, x_train, y_test, y_train
### Train the model
  - As it is a binary classification task, I have tried to train 3 m=different models "Logistic regression","Naive Bayes classifier", "Random forest classifier"
  - below are the metrics and ROC curve for all three models
  
| Model Name        | Accuracy           | Class 0 Precison  | class 1 Precision | class 0 recall | class 1 recall | class 0 F1 | class 1 F1 |
| ------------- |:-------------:| -----:| ----: |----: | ----: | -----: | -----: |
| Logistic regression      | 0.96 |  0.97 | 0.95 | 0.95 | 0.97 | 0.96| 0.96 |
| Naive Bayes     | 0.86      |   0.79 | 0.99 | 0.99 | 0.72 | 0.88 | 0.83 |
| Random Forest | 0.94      |   0.92 | 0.96 | 0.96 | 0.91 | 0.94 |0.93 |
| LSTM |  0.97  | 0.97  | 0.98  | 0.98  | 0.97  | 0.97  | 0.97 |

### ROC plot
  - Below is the ROC plot for all three models
  

![alt text](https://github.com/sanket203/ML-repo/blob/main/ROC.png "Logo Title Text 1")

### Model selection
* After analysing above results for all three models . Logistic Regression and LSTM model is selected as best model for prediction.
* Predictions are done on articles from clean_test.csv and results are saved to output_submit.csv

