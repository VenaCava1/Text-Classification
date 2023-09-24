#Task 6: Text_Classification

from google.colab import drive
drive.mount('/content/drive')
file_path = '/content/drive/My Drive/complaints.csv'
original = pd.read_csv(file_path)

#file_path = 'C:/Users/Aakash/Desktop/Kaiburr/Complaints/complaints.csv'
#original = pd.read_csv(file_path)

import pandas as pd

df = pd.DataFrame(original)

#Removing the columns that are not required and then removing rows containing null values
columns_to_remove = ["Sub-product", "Issue", "Sub-issue", "ZIP code", "Company public response", "Company", "Tags", "Consumer consent provided?", "Consumer disputed?", "Complaint ID", "Date sent to company", "Company response to consumer", "Timely response?"]
df = df.drop(columns=columns_to_remove)
df = df.dropna()

# Keep only the rows with the specified classes in the original DataFrame
selected_classes = [
    'Credit reporting, credit repair services, or other personal consumer reports',
    'Debt collection',
    'Mortgage',
    'Consumer Loan'
]
df = df[df['Product'].isin(selected_classes)]

df['Product'].replace(
    'Credit reporting, credit repair services, or other personal consumer reports',
    'Credit reporting, repair, or other',
    inplace=True
)

#Part 1: Exploratory Data Analysis (EDA)

import matplotlib.pyplot as plt
import seaborn as sns

#(i) Class Distribution
class_counts = df['Product'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title('Class Distribution')
plt.xlabel('Product')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()

#(ii) Text Length
df['Text Length'] = df['Consumer complaint narrative'].apply(lambda x: len(x.split()))
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Text Length', kde=True)
plt.title('Distribution of Text Length')
plt.xlabel('Text Length')
plt.ylabel('Count')
plt.show()

#(iii) Geographical Patterns
complaints_by_state = df['State'].value_counts()

plt.figure(figsize=(12, 6))
sns.barplot(x=complaints_by_state.index, y=complaints_by_state.values)
plt.title('Complaints by State')
plt.xlabel('State')
plt.ylabel('Number of Complaints')
plt.xticks(rotation=90)
plt.show()

#Part 2: Text Preprocessing

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    
    # Join the tokens back into a processed text
    processed_text = ' '.join(stemmed_tokens)
    
    return processed_text

# Apply the text preprocessing function to the 'Consumer Complaint Narrative' column
df['Processed Complaints'] = df['Consumer complaint narrative'].apply(preprocess_text)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust the number of features as needed
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Processed Complaints'])
# The 'tfidf_matrix' now contains the TF-IDF representation of the preprocessed text data

#Part 3: Selection and Evaluation of various Models

# Split your data into training and testing sets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

X = tfidf_matrix
y = df['Product']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_f1 = f1_score(y_test, rf_predictions, average='weighted')
rf_precision = precision_score(y_test, rf_predictions, average='weighted')
rf_recall = recall_score(y_test, rf_predictions, average='weighted')
tn, fp, fn, tp = confusion_matrix(y_test, rf_predictions).ravel()
rf_specificity = tn / (tn + fp)

print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest F1-Score:", rf_f1)
print("Random Forest Precision:", rf_precision)
print("Random Forest Recall:", rf_recall)
print("Random Forest Specificity:", rf_specificity)
plot_confusion_matrix(y_test, rf_predictions, classes=df['Product'].unique(), title="Random Forest Confusion Matrix")

# Support Vector Machine (SVM)
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, y_train)
svm_predictions = svm_classifier.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_f1 = f1_score(y_test, svm_predictions, average='weighted')
svm_precision = precision_score(y_test, svm_predictions, average='weighted')
svm_recall = recall_score(y_test, svm_predictions, average='weighted')
tn, fp, fn, tp = confusion_matrix(y_test, svm_predictions).ravel()
svm_specificity = tn / (tn + fp)

print("SVM Accuracy:", svm_accuracy)
print("SVM F1-Score:", svm_f1)
print("SVM Precision:", svm_precision)
print("SVM Recall:", svm_recall)
print("SVM Specificity:", svm_specificity)
plot_confusion_matrix(y_test, svm_predictions, classes=df['Product'].unique(), title="SVM Confusion Matrix")

# Logistic Regression
logistic_classifier = LogisticRegression(random_state=42)
logistic_classifier.fit(X_train, y_train)
logistic_predictions = logistic_classifier.predict(X_test)
logistic_accuracy = accuracy_score(y_test, logistic_predictions)
logistic_f1 = f1_score(y_test, logistic_predictions, average='weighted')
logistic_precision = precision_score(y_test, logistic_predictions, average='weighted')
logistic_recall = recall_score(y_test, logistic_predictions, average='weighted')
tn, fp, fn, tp = confusion_matrix(y_test, logistic_predictions).ravel()
logistic_specificity = tn / (tn + fp)

print("Logistic Regression Accuracy:", logistic_accuracy)
print("Logistic Regression F1-Score:", logistic_f1)
print("Logistic Regression Precision:", logistic_precision)
print("Logistic Regression Recall:", logistic_recall)
print("Logistic Regression Specificity:", logistic_specificity)
plot_confusion_matrix(y_test, logistic_predictions, classes=df['Product'].unique(), title="Logistic Regression Confusion Matrix")

#Step 4: Final comparison

metrics_df = pd.DataFrame({
    'Model': ['SVM', 'Random Forest', 'Logistic Regression'],
    'Accuracy': [svm_accuracy, rf_accuracy, logistic_accuracy],
    'F1-Score': [svm_f1, rf_f1, logistic_f1],
    'Precision': [svm_precision, rf_precision, logistic_precision],
    'Recall': [svm_recall, rf_recall, logistic_recall],
    'Specificity': [svm_specificity, rf_specificity, logistic_specificity]
})
print(metrics_df)
