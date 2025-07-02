import pytest 
from utils.tokenizer import Tokenizer
from models import *

@pytest.fixture
def tokenizer():
    return Tokenizer()

@pytest.fixture
def sample_sequences():
    return [
        "I love reading books",
        "Python is great for data science",
        "Machine learning is fascinating",
        "Natural language processing with Python",
        "Deep learning and neural networks"
    ]
    
@pytest.fixture
def sample_text():
    text = '''Artificial intelligence is transforming industries worldwide.
                Data science and machine learning are key components of AI.
                Python is a popular programming language for AI development.
                Neural networks and deep learning are advancing rapidly.'''
    
    tokenizer = Tokenizer() 
    clenan_text = tokenizer.clean_text(text)
    return clenan_text

@pytest.fixture
def knn_regression():
    return Knn_regression()

@pytest.fixture
def linear_regression():
    return Linear_regression()  

@pytest.fixture
def decision_tree_regression():
    return Decision_tree_regression()   

@pytest.fixture
def nn_regression(encode: str):
    if encode == "ibe_text":
        return Nn_regression([20, 500, 100, 1], epochs=2000)
    elif encode == "bow_text":
        return Nn_regression([5000, 100, 50, 1], epochs=2000)
    elif encode == "word2vec_text":
        return Nn_regression([300, 100, 100, 1], epochs=2000)
    else:
        raise ValueError("Invalid text transformation method for neural network")
