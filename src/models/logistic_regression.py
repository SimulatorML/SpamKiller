from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from dataclasses import dataclass
from loguru import logger


logger.info("Init LogisticRegression")

class SpamLogisticRegression:
    def __init__(self):
        self.model = LogisticRegression(class_weight='balanced',solver='lbfgs', max_iter=10_000, n_jobs=-1)
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        predictions = self.model.predict_proba(X)
        return predictions[:, 1]
