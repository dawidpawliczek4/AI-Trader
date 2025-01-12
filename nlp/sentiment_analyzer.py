from typing import List
from utils import Tokenizer, longest_sequence, mae, mse
from models import Decision_tree_regression, Knn_regression, Linear_regression, Nn_regression
import pandas as pd 
from sklearn.model_selection import train_test_split
import torch

class sentiment_analyzer():
    '''
    input:
        word_transformation: string: ibe, bow, word2vec
        model: string: decision_tree_regression, knn_regression, linear_regression, nn
    '''
    def __init__(self, word_transformation: str, model: str):
        self.tokenizer = Tokenizer()
        self.text_transformation = {
            "ibe": self.tokenizer.ibe_text,
            "bow": self.tokenizer.bow_text,
            "word2vec": self.tokenizer.word2vec_text
        }.get(word_transformation)
        
        self.modelClass = {
            "decistion_tree_regression": Decision_tree_regression,
            "knn_regression": Knn_regression, 
            "linear_regression": Linear_regression,
            "nn": Nn_regression
        }.get(model)
        
    def fit(self):
        '''
            before making predictions use this function. Don't pass any arguments, 
            everything will be handle for you.
            input:
                Nothing
            return:
                Nothing
        '''
        
        # get data 
        df = pd.read_csv('dataset.csv')
        X, y = df['Headline'].tolist(), df['Sentiment'].tolist()
        
        # clean data
        X = [self.tokenizer.clean_text(x) for x in X]
        
        # split data 
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        
        # encode data
        if self.text_transformation == self.tokenizer.ibe_text:
            self.MAX_TOKEN_NO = 5000 
            self.SEQUENCE_LEN = longest_sequence(X)
            
            self.tokenizer.fit_on_sequences(self.X_train)
            self.X_train = [self.tokenizer.ibe_text(text=x, max_token_no=self.MAX_TOKEN_NO, sequence_len=self.SEQUENCE_LEN) for x in self.X_train]
            self.X_test = [self.tokenizer.ibe_text(text=x, max_token_no=self.MAX_TOKEN_NO, sequence_len=self.SEQUENCE_LEN) for x in self.X_test]
            
        elif self.text_transformation == self.tokenizer.bow_text:
            self.MAX_TOKEN_NO = 5000 
            
            self.tokenizer.fit_on_sequences(self.X_train) 
            self.X_train = [self.tokenizer.bow_text(text=x, max_token_no=self.MAX_TOKEN_NO) for x in self.X_train]
            self.X_test = [self.tokenizer.bow_text(text=x, max_token_no=self.MAX_TOKEN_NO) for x in self.X_test]
            
        elif self.text_transformation == self.tokenizer.word2vec_text:
            self.X_train = [self.tokenizer.word2vec_text(text=x) for x in self.X_train]
            self.X_test = [self.tokenizer.word2vec_text(text=x) for x in self.X_test]
            
        # choose model
        if self.modelClass == Decision_tree_regression:
            self.model = Decision_tree_regression()
            
        elif self.modelClass == Knn_regression: 
            self.model = Knn_regression()
            
        elif self.modelClass == Linear_regression:
            self.model = Linear_regression()
            
        elif self.modelClass == Nn_regression:
            self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
            self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
            self.y_train = torch.tensor(self.y_train, dtype=torch.float32).view(-1, 1)
            self.y_test = torch.tensor(self.y_test, dtype=torch.float32).view(-1, 1)
            
            if self.text_transformation == self.tokenizer.ibe_text:
                self.model = Nn_regression([5000, 1000, 100, 1])
                
            elif self.text_transformation == self.tokenizer.bow_text:
                self.model = Nn_regression([5000, 1000, 100, 1])
                
            elif self.text_transformation == self.tokenizer.word2vec_text:
                self.model = Nn_regression([300, 100, 100, 1])
            
        # train model
        self.model.fit(self.X_train, self.y_train)
        
    def predict(self, X):
        # clean data 
        X = [self.tokenizer.clean_text(x) for x in X]
        
        # encode data 
        if self.text_transformation == self.tokenizer.ibe_text:
            X = [self.tokenizer.ibe_text(text=x, max_token_no=self.MAX_TOKEN_NO, sequence_len=self.SEQUENCE_LEN) for x in X]
            
        elif self.text_transformation == self.tokenizer.bow_text:
            
            X = [self.tokenizer.bow_text(text=x, max_token_no=self.MAX_TOKEN_NO) for x in X]
            
        elif self.text_transformation == self.tokenizer.word2vec_text:
            X = [self.tokenizer.word2vec_text(text=x) for x in X]
            
        # check if we use tf
        if self.modelClass == Nn_regression:
            X = torch.tensor(self.X_train, dtype=torch.float32)
            
        # predict 
        return self.model.predict(X)
        
    
    def evaluate(self):
        '''
            function for model evaluation on metrices: [mse, mae]
            input:
                void 
            return: 
                dictionary {metrices: score}
        '''
        y_pred = self.model.predict(self.X_test)
        
        return {
            "mse": mse(y_pred, self.y_test),
            "mae": mae(y_pred, self.y_test)
        }
        
            
            
        
        
        
        
        
        
    