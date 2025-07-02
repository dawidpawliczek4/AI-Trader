from abc import ABC, abstractmethod
from numpy import ndarray

class BaseModel(ABC):
    def __init__(self):
        pass
   
    @abstractmethod 
    def fit(self, X_train: ndarray, y_train: ndarray) -> None:
        pass
    
    @abstractmethod 
    def predict(self, X_test: ndarray) -> ndarray:
        pass