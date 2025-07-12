from sklearn.linear_model import LinearRegression
from .base_model import BaseModel
from numpy import ndarray
import joblib

class Linear_regression(BaseModel):
    def __init__(self, **kwargs):
       self.model = LinearRegression(**kwargs) 
    
    def fit(self, X_train: ndarray, y_train: ndarray) -> None:
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test: ndarray) -> ndarray:
        return self.model.predict(X_test)
    
    def load(self, path: str):
        self.model = joblib.load(path)

    def save(self, path: str):
        joblib.dump(self.model, path)