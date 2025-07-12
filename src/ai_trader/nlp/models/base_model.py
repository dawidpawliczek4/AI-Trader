from abc import ABC, abstractmethod
from numpy import ndarray

class BaseModel(ABC):
    def __init__(self):
        pass
   
    @abstractmethod 
    def fit(self, X_train: ndarray, y_train: ndarray) -> None:
        """
        Fit the model to the training data.
        """
        pass
    
    @abstractmethod 
    def predict(self, X_test: ndarray) -> ndarray:
        """
        Predict the output for the given input data.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the model from the specified path.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the model to the specified path.
        """
        pass