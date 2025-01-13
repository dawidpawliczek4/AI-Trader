from sklearn.ensemble import RandomForestClassifier

class Random_forest_classifier:
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
