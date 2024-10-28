import string
import pandas as pd

from .nltk_helper import remove_stopwords, lemmatize_text


def nan_to_unlimited(text):
    """
    @brief Converts specific ban statuses to "Unlimited".

    This function checks the input text and returns "Banned", "Limited", 
    or "Semi-Limited" if it matches any of those statuses. Otherwise, 
    it returns "Unlimited".

    @param text The input ban status as a string.
    @return A string indicating the ban status, which is either 
            "Banned", "Limited", "Semi-Limited", or "Unlimited".
    """
    if text == "Banned" or text == "Limited" or text == "Semi-Limited":
        return text
    else:
        return "Unlimited"
    
def prep_data(df):
    """
    @brief Prepares the card data for analysis by filtering, transforming, and cleaning.

    This function filters the DataFrame to keep only non-Normal Monster cards, 
    drops entries with NaN values in the 'archetype' column, and further filters 
    archetypes based on their frequency. It also processes the 'desc' column 
    by converting it to lowercase, removing stopwords, punctuation, and lemmatizing 
    the text. Additionally, it processes the ban information.

    @param df The input DataFrame containing card data.
    @return A tuple of two DataFrames:
            - arch_df: A DataFrame of archetypes with cleaned descriptions.
            - ban_df: A DataFrame of ban information with transformed statuses.
    """
    arch_df = df[df['type'] != 'Normal Monster'].dropna(subset=['archetype'])
    archetype_counts = arch_df['archetype'].value_counts()
    archetypes_to_keep = archetype_counts[archetype_counts >= 10].index
    arch_df = arch_df[arch_df['archetype'].isin(archetypes_to_keep)]
    arch_df["desc"] = arch_df["desc"].map(lambda x: x.lower())
    arch_df["desc"] = arch_df["desc"].map(remove_stopwords)
    arch_df["desc"] = arch_df["desc"].map(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    arch_df["desc"] = arch_df["desc"].map(lemmatize_text)

    ban_df = df.copy()
    ban_df['ban_tcg'] = ban_df['ban_tcg'].apply(nan_to_unlimited)
    ban_df['tcg_date'] = pd.to_datetime(ban_df['tcg_date'])

    return arch_df, ban_df

def prep_data_predict(df):
    arch_df = df[df['type'] != 'Normal Monster'].dropna(subset=['archetype'])
    arch_df["desc"] = arch_df["desc"].map(lambda x: x.lower())
    arch_df["desc"] = arch_df["desc"].map(remove_stopwords)
    arch_df["desc"] = arch_df["desc"].map(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    arch_df["desc"] = arch_df["desc"].map(lemmatize_text)

    ban_df = df.copy()
    ban_df['ban_tcg'] = ban_df['ban_tcg'].apply(nan_to_unlimited)
    ban_df['tcg_date'] = pd.to_datetime(ban_df['tcg_date'])

    return arch_df, ban_df