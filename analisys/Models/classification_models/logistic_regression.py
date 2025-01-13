from sklearn.linear_model import LogisticRegression

class Logistic_regression:
    def __init__(self, **kwargs):
        self.model = Logistic_regression(**kwargs)
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)