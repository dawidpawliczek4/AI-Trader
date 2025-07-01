from sklearn.linear_model import LinearRegression

class Linear_regression:
    def __init__(self, **kwargs):
       self.model = LinearRegression(**kwargs) 
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)