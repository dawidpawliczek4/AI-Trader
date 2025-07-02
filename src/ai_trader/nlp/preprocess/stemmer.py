import nltk

try:
    nltk.data.find('wordnet')
except: 
    nltk.download('wordnet')

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()


def stem(text: str) -> str:
    '''
        function form word lemmatization
        input: 
            text: str
        return
            str
    '''
    
    tokens = text.split()
    return ' '.join([stemmer.stem(token) for token in tokens])