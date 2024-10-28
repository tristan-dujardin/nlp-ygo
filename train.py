import pandas as pd
import pickle 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

from src.data_helper import prep_data

# Data Loading
full_data = pd.read_csv('yu-gi-oh/data/cards.csv')

# Data Processing
arch_df, ban_df = prep_data(full_data)

# Separation in Train and Test sets
train_df, test_df = train_test_split(arch_df, test_size=0.2, random_state=42)

# Vectorization relative to the train set
vectorizer = CountVectorizer()
train_vectors = vectorizer.fit_transform(train_df['desc'])

with open('models/archetype_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

inference_vectors = vectorizer.transform(test_df['desc'])

# Arch Classifier
arch_clf = LogisticRegression(max_iter=1000)
arch_clf.fit(train_vectors, train_df['archetype'])
with open('models/archetype_model.pkl', 'wb') as f:
    pickle.dump(arch_clf, f)

print("Saved models/archetype_model.pkl")


# Test set
inference_labels = arch_clf.predict(inference_vectors)
print("Trained Archetype Classifier Model, Accuracy: " + str(accuracy_score(test_df['archetype'], inference_labels)))

# Separation in Train and Test sets
train_df, test_df = train_test_split(ban_df, test_size=0.2, random_state=42)
#Vectorizer for banlist
vectorizer = CountVectorizer()
train_vectors = vectorizer.fit_transform(train_df['desc'])

with open('models/banlist_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f) 

inference_vectors = vectorizer.transform(test_df['desc'])

# Banlist Classifier
ban_clf = LogisticRegression(max_iter=300, random_state=4200)
ban_clf.fit(train_vectors, train_df['ban_tcg'])
with open('models/banlist_model.pkl', 'wb') as f:
    pickle.dump(ban_clf, f)

print("Saved models/banlist_model.pkl")

inference_labels = ban_clf.predict(inference_vectors)

print("Trained Banlist Classifier Model, Accuracy: " + str(accuracy_score(test_df['ban_tcg'], inference_labels)))


