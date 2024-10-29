import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.data.path.append('./nltk_data')

def remove_stopwords(text):
    """
    Removes stopwords from the given text.

    Args:
        text (str): The input text from which stopwords will be removed.

    Returns:
        str: The input text without stopwords.
    """
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)


def lemmatize_text(text):
    """
    Lemmatizes the given text.

    Args:
        text (str): The input text to be lemmatized.

    Returns:
        str: The lemmatized version of the input text.
    """
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized_tokens)