import sys
import os
import sys
import time

sys.dont_write_bytecode = True
sys.path.append(os.path.abspath('..'))

from nlp import Sentiment_analyzer

word_transformations = ['ibe', 'bow', 'word2vec']
models = ['decision_tree_regression', 'knn_regression', 'linear_regression']

for wt in word_transformations:
    for model in models:
        start = time.time()
        nlp = Sentiment_analyzer(wt, model)
        nlp.fit()
        end = time.time()
        print(wt, model, nlp.evaluate(), end - start)