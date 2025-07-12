from sklearn.tree import DecisionTreeRegressor
from .base_model import BaseModel
from numpy import ndarray
import joblib

class Decision_tree_regression(BaseModel):
    def __init__(self, **kwargs):
        self.model = DecisionTreeRegressor(**kwargs)
        
    def fit(self, X_train: ndarray, y_train: ndarray) -> None:
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test: ndarray) -> ndarray:
        X_test = X_test.reshape(1, -1)
        return self.model.predict(X_test)
    
    def load(self, path: str):
        self.model = joblib.load(path)

    def save(self, path: str):
        joblib.dump(self.model, path)