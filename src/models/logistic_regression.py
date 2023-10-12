from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from loguru import logger


logger.info("Init LogisticRegression")


class SpamLogisticRegression:
    def __init__(self):
        self.logreg = LogisticRegression(penalty='l1',
                                         C=0.001,
                                         solver='saga',
                                         class_weight={0: 1, 1: 1000},
                                         random_state=1)
        self._percentage_not_spam_id = 0.3
        self._not_spam_id = []
        self._scaler = None

    def fit(self, X, y):
        X = X.copy()

        # Filling not_spam_id
        not_spam_id_list = X[y == 0].from_id.unique()
        self._not_spam_id = not_spam_id_list[:int(len(not_spam_id_list) * self._percentage_not_spam_id)]

        # Creating 'contains_not_spam_id' feature based on train data
        X['contains_not_spam_id'] = X.from_id.apply(lambda from_id: int(from_id in self._not_spam_id))
        X = X.drop('from_id', axis=1)

        # Scaling data
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Fitting data into LogReg model
        self.logreg.fit(X_scaled, y)

        return self

    def predict(self, X):
        X = X.copy()

        # Creating 'contains_not_spam_id' feature based on train data
        X['contains_not_spam_id'] = X.from_id.apply(lambda from_id: int(from_id in self._not_spam_id))
        X = X.drop('from_id', axis=1)

        # Scaling data
        X_scaled = self._scaler.transform(X)

        # Prediction
        scores = self.logreg.predict_proba(X_scaled)[:, 1]

        return scores
