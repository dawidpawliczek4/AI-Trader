import nltk
nltk.download('wordnet')

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

'''
    function form word lemmatization
    input: 
        text: str
    return
        str
'''

def stem(text: str) -> str:
    tokens = text.split()
    return ' '.join([stemmer.stem(token) for token in tokens])