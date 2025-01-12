# NLP Module

---

This module provides tools for building and evaluating Natural Language Processing (NLP) models with a focus on sentiment analysis. It includes various methods for preprocessing text, transforming words into numerical data, and applying machine learning models for prediction.

### Steps
1. #### data preprocessing
    Text preprocessing is an essential step before applying machine learning models. This section covers common preprocessing techniques:
    * **stopwords removal**: removing `this`, `that`, `he`, `it`... from sentences <!-- TODO -->
    * **stemming**: groups different inflected forms of the same word, for example: `changing`, `changed` -> `change`
    * **lowercase conversion**

2. #### transformation word into data
    To train machine learning models, text needs to be transformed into numerical data. This section describes various encoding techniques:

    * **index based encoding**
    * **bag of words**
    * **TF-IDF**  <!-- TODO -->
    * **embedding** (Word2Vec, GloVe, FastText)
    * **BERT ENcoding**: Uses language models like BERT to generate contextual word embeddings. <!-- TODO -->

3. #### model training
    * **NN**
    * **knn regression**
    * **decition trees regression**
    * **linear regression**
    * **recurrent NN** <!-- TODO -->
    * **transformers** <!-- TODO -->

4. #### model evaluation
    * **MAE**: mean absolutive error
    * **MSE**: mean square error

5. #### using model for predictions 

---

### Methods in the main class Sentiment_analyzer():
* ####  __init__(word_transformation: str, model: str):
    class inicjalization, you have to pass chosen word_transformation and model,
    word_transformation: ["ibe", "bow", "word2Vec"]
    * ibe: index based encoding
    * bow: bag of words
    * word2vec: word2vec

    model: ["decision_tree_regression", "knn_regression", "linear_regression", "nn"]
    * decision_tree_regression: decision tree in regression version
    * knn_regression: k nearest neighbours in regresssion version
    * linear_regression: linear regression
    * nn: simple neural network

* #### fit():
    Before making predictions use this function. Don't pass any arguments, everything will be handle for you.
    input:
    * Nothing

    output:
    * Nothing

* #### predict(X: List[str]) -> List[float]:
    function for making predictions
    input:
    * X: List[str]: your list of strings for making predictions

    output:
    * List[float]: predictions [0, 1] 0: bad sentiment, 1: great sentiment

* #### evaluate() -> {mse, mae}:
    function for evaluating our model
    input:
    * Nothing
    output:
    * dictionary with mse and mae

---

### Performance

#### Mean Absolutive Error:

|     | ibe | bow | word2vec |
| --- | --- | --- | --- |
| decision tree| 0.31 | 0.20 | 0.29 |
| knn | 0.32 | 0.29 | 0.26 |
| linear regression | 0.31 | 0.23 |  0.27 |
| nn | 0.33 | | |