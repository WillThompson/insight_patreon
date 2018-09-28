import spacy
import nltk
from spacy.lang.en import English
import data_cleaner
import string


# Create a Lemmatizer
from nltk.corpus import wordnet as wn
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

from nltk.stem.wordnet import WordNetLemmatizer
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

# Get the corpus of stop words from nltk and scikit learn

# Define a function which will first tokenize the text appropriately.
parser = English()
def tokenize(text):
    lda_tokens = []
    
    # Remove the html tags
    new_text = data_cleaner.clean_text(text)
    tokens = parser(new_text) 
    
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

# ... then a function which will filter appropriately.
# First get the corpus of stop words from nltk and scikit learn.
en_stop = set(nltk.corpus.stopwords.words('english'))

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
STOPLIST = set(list(en_stop) + ["n't", "'s", "'m", "ca"] + list(ENGLISH_STOP_WORDS))
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-----", "---", "...", "“", "”", "'ve"]

def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in STOPLIST]
    tokens = [get_lemma(token) for token in tokens]
    return tokens