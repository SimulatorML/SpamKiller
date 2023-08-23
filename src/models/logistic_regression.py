from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from dataclasses import dataclass
from loguru import logger


logger.info("Init LogisticRegression")
class SpamLogisticRegression:
    def __init__(self):
        class_weights = {0 : 1, 1 : 837}
        self.model = LogisticRegression(penalty='l1',
                                        class_weight=class_weights,
                                          solver='liblinear', n_jobs=-1)
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        predictions = self.model.predict_proba(X)
        return predictions[:, 1]
