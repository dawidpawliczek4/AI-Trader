import xgboost as xgb

class XGBoost_classifier:
    def __init__(self, **kwargs):
        self.model = xgb.XGBClassifier(**kwargs)
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
