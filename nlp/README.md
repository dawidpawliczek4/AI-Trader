# NLP Module

---

This module provides tools for building and evaluating Natural Language Processing (NLP) models with a focus on sentiment analysis. It includes various methods for preprocessing text, transforming words into numerical data, and applying machine learning models for prediction.

### Steps
1. #### Data Preprocessing
    Text preprocessing is an essential step before applying machine learning models. This section covers common preprocessing techniques:
    * **Stopword Removal**: Removing `this`, `that`, `he`, `it`... from sentences <!-- TODO -->
    * **Stemming**: Groups different inflected forms of the same word, for example: `changing`, `changed` -> `change`
    * **Lowercase Conversion**

2. #### Transforming Words into Data
    To train machine learning models, text needs to be transformed into numerical data. This section describes various encoding techniques:

    * **Index-Based Encoding**
    * **Bag of Words**
    * **Embedding** (Word2Vec, GloVe, FastText)
    * **BERT Encoding**: Uses language models like BERT to generate contextual word embeddings. <!-- TODO -->

3. #### Models
    * **NN** (Neural Network)
    * **KNN Regression**
    * **Decision Tree Regression**
    * **Linear Regression**
    * **Recurrent NN** <!-- TODO -->
    * **Transformers** <!-- TODO -->

4. #### Model Evaluation
    * **MAE**: Mean Absolute Error
    * **MSE**: Mean Squared Error

5. #### Using the Model for Predictions 

---

### Methods in the Main Class `SentimentAnalyzer`
* #### `__init__(word_transformation: str, model: str)`
    Class initialization. You need to pass the chosen `word_transformation` and `model`.
    
    `word_transformation`: ["ibe", "bow", "word2vec"]
    * **ibe**: Index-Based Encoding
    * **bow**: Bag of Words
    * **word2vec**: Word2Vec

    `model`: ["decision_tree_regression", "knn_regression", "linear_regression", "nn"]
    * **decision_tree_regression**: Decision tree in regression mode
    * **knn_regression**: K-Nearest Neighbors in regression mode
    * **linear_regression**: Linear regression
    * **nn**: Simple neural network

* #### `fit()`
    Use this function before making predictions. No arguments are required; everything will be handled for you.
    
    **Input**:  
    * None
    
    **Output**:  
    * None

* #### `predict(X: List[str]) -> List[float]`
    Function for making predictions.
    
    **Input**:  
    * `X: List[str]`: Your list of strings for making predictions
    
    **Output**:  
    * `List[float]`: Predictions in the range [0, 1], where 0 indicates negative sentiment and 1 indicates positive sentiment.

* #### `evaluate() -> dict`
    Function for evaluating the model.
    
    **Input**:  
    * None
    
    **Output**:  
    * Dictionary containing `mse` and `mae`.

---

### Performance

#### Mean Absolute Error:

| Model               | ibe  | bow  | word2vec |
|---------------------|------|------|----------|
| Decision Tree       | 0.31 | 0.20 | 0.29     |
| KNN                 | 0.32 | 0.29 | 0.26     |
| Linear Regression   | 0.31 | 0.23 | 0.27     |
| Neural Network (NN) | 0.33 | 0.31 | 0.31     |