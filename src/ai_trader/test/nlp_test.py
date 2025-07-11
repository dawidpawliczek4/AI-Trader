import sys
import os
import sys
import time

sys.dont_write_bytecode = True
sys.path.append(os.path.abspath('..'))

from ai_trader.nlp.sentiment_analyzer import Sentiment_analyzer

word_transformations = ['ibe']

# models = ['linear_regression', 'decision_tree_regression', 'knn_regression', 'nn']
models = ['nn']

for wt in word_transformations:
    for model in models:
        print("doing ok")
        start = time.time()
        nlp = Sentiment_analyzer(wt, model)
        nlp.fit()
        end = time.time()
        print(wt, model, nlp.evaluate(), end - start)   