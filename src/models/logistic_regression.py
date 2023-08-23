from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from dataclasses import dataclass
from loguru import logger


logger.info("Init LogisticRegression")
class SpamLogisticRegression:
    def __init__(self):
        self.model = LogisticRegression(penalty='l1',
                                        class_weight='balanced',
                                        solver='liblinear', n_jobs=-1)
    
    def preproc_data(self, X):
        scale = StandardScaler()
        normalize_data = scale.fit_transform(X)
        return normalize_data

    def fit(self, X, y):
        X = self.preproc_data(X)
        self.model.fit(X, y)
    
    def predict(self, X):
        X = self.preproc_data(X)
        predictions = self.model.predict_proba(X)
        return predictions[:, 1]
