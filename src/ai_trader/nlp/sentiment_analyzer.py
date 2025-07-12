import pandas as pd
import numpy as np
import os
from typing import List
from sklearn.model_selection import train_test_split
from ai_trader.nlp.utils import Tokenizer, longest_sequence, mae, mse
from ai_trader.nlp.models import Decision_tree_regression, Knn_regression, Linear_regression, Nn_regression


class Sentiment_analyzer():
    '''
    input:
        word_transformation: string: ibe, bow, word2vec
        model: string: decision_tree_regression, knn_regression, linear_regression, nn        
        load_path: string | None: path to pretrained model
    '''

    def __init__(self, word_transformation: str, model: str, load_path: str = None):
        self.validate(word_transformation, model)
        self.tokenizer = Tokenizer()
        self.text_transformation = {
            "ibe": self.tokenizer.ibe_text,
            "bow": self.tokenizer.bow_text,
            "word2vec": self.tokenizer.word2vec_text
        }.get(word_transformation)

        self.modelClass = {
            "decision_tree_regression": Decision_tree_regression,
            "knn_regression": Knn_regression,
            "linear_regression": Linear_regression,
            "nn": Nn_regression
        }.get(model)
        
        self.load_path = load_path
        self.model = None

        
    def validate(self, word_transformation: str, model: str) -> None:
        if word_transformation not in ["ibe", "bow", "word2vec"]:
            raise ValueError("Invalid word transformation method. Choose from 'ibe', 'bow', or 'word2vec'.")
        if model not in ["decision_tree_regression", "knn_regression", "linear_regression", "nn"]:
            raise ValueError("Invalid model type. Choose from 'decision_tree_regression', 'knn_regression', 'linear_regression', or 'nn'.")
       
    def fit(self, custom_X: List[str] | None = None, custom_y: List[float] | None = None, save_path: str = None):
        # validate custom input
        if custom_X is not None and custom_y is None:
            raise ValueError("If custom_X is provided, custom_y must also be provided.")
        if custom_y is not None and custom_X is None:
            raise ValueError("If custom_y is provided, custom_X must also be provided.")
        if custom_X is not None and custom_y is not None:
            if len(custom_X) != len(custom_y):
                raise ValueError("custom_X and custom_y must have the same length.")
        
        # prepare data 
        if custom_X is None:
            # get data
            dir = os.path.dirname(os.path.abspath(__file__))
            df_path = os.path.join(dir, 'dataset.csv')

            df = pd.read_csv(df_path)
            X, y = df['Headline'].tolist(), df['Sentiment'].tolist()
        else: 
            X, y = custom_X, custom_y

        # clean data
        X = [self.tokenizer.clean_text(x) for x in X]

        # split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        
        self.y_train = np.array(self.y_train) 
        self.y_test = np.array(self.y_test).reshape(-1, 1)
        
        # encode data
        if self.text_transformation == self.tokenizer.ibe_text:
            self.MAX_TOKEN_NO = 5000
            self.SEQUENCE_LEN = longest_sequence(X)

            self.tokenizer.fit_on_sequences(self.X_train)
            self.X_train = np.array([self.tokenizer.ibe_text(
                text=x, max_token_no=self.MAX_TOKEN_NO, sequence_len=self.SEQUENCE_LEN) for x in self.X_train])
            self.X_test = np.array([self.tokenizer.ibe_text(
                text=x, max_token_no=self.MAX_TOKEN_NO, sequence_len=self.SEQUENCE_LEN) for x in self.X_test])

        elif self.text_transformation == self.tokenizer.bow_text:
            self.MAX_TOKEN_NO = 5000

            self.tokenizer.fit_on_sequences(self.X_train)
            self.X_train = np.array([self.tokenizer.bow_text(
                text=x, max_token_no=self.MAX_TOKEN_NO) for x in self.X_train])
            self.X_test = np.array([self.tokenizer.bow_text(
                text=x, max_token_no=self.MAX_TOKEN_NO) for x in self.X_test])

        elif self.text_transformation == self.tokenizer.word2vec_text:
            self.X_train = np.array([self.tokenizer.word2vec_text(
                text=x) for x in self.X_train])
            self.X_test = np.array([self.tokenizer.word2vec_text(
                text=x) for x in self.X_test])

            self.X_train = np.array(self.X_train, dtype=np.float32)
            self.X_test = np.array(self.X_test, dtype=np.float32)
        else:
            raise ValueError("Invalid text transformation method")
            
        # choose model
        if self.modelClass == Decision_tree_regression:
            self.model = Decision_tree_regression()

        elif self.modelClass == Knn_regression:
            self.model = Knn_regression()

        elif self.modelClass == Linear_regression:
            self.model = Linear_regression()

        elif self.modelClass == Nn_regression:
            if self.text_transformation == self.tokenizer.ibe_text:
                self.model = Nn_regression(
                    [self.SEQUENCE_LEN, 500, 100, 1], epochs=750)

            elif self.text_transformation == self.tokenizer.bow_text:
                self.model = Nn_regression([5000, 100, 50, 1], epochs=2000)

            elif self.text_transformation == self.tokenizer.word2vec_text:
                self.model = Nn_regression([300, 100, 100, 1], epochs=2000)
            else:
                raise ValueError("Invalid text transformation method for neural network")
        
        # Load pre-trained model if path is provided
        if self.load_path:            
            self.model.load(self.load_path)
            return
            
        # train model
        self.model.fit(self.X_train, self.y_train)
        if save_path:
            self.model.save(save_path)

    def predict(self, X: List[str]) -> List[float]:
        '''
            function for making prediction
            input:
                X: List[str]
            return:
                List[float]: our predictions [0, 1]

        '''
        if self.model is None and self.load_path:
            self.fit()
        elif self.model is None:
            raise ValueError("Model has not been trained or loaded. Please call fit() first.")
        
        # clean data 
        Xls = [self.tokenizer.clean_text(x) for x in X]
        
        # encode data 
        if self.text_transformation == self.tokenizer.ibe_text:
            Xna = np.array([self.tokenizer.ibe_text(text=x, max_token_no=self.MAX_TOKEN_NO, sequence_len=self.SEQUENCE_LEN) for x in Xls])
            
        elif self.text_transformation == self.tokenizer.bow_text:
            Xna = np.array([self.tokenizer.bow_text(text=x, max_token_no=self.MAX_TOKEN_NO) for x in Xls])
            
        elif self.text_transformation == self.tokenizer.word2vec_text:
            Xna = np.array([self.tokenizer.word2vec_text(text=x) for x in Xls])
            
        else:
            raise ValueError("Invalid text transformation method")
            
        return [self.model.predict(x.reshape(1, -1))[0] for x in Xna]
         
    def evaluate(self):
        '''
            function for model evaluation on metrices: [mse, mae]
            input:
                void 
            return: 
                dictionary {metrices: score}
        '''
        if self.X_test is None:
            raise ValueError("Model has not been trained yet. Please call fit() before evaluate().")
        y_pred = np.array([self.model.predict(x.reshape(1, -1)) for x in self.X_test])
        
        return {
            "mse": mse(y_pred, self.y_test),
            "mae": mae(y_pred, self.y_test)
        }
