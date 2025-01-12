from sklearn.tree import DecisionTreeRegressor

class Decision_tree_regression:
    def __init__(self, **kwargs):
        self.model = DecisionTreeRegressor(**kwargs)
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)