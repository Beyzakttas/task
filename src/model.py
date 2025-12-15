# src/model.py - GÜNCELLENMİŞ (Random Forest eklendi)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

class SpamClassifier:
    def __init__(self, model_type='logistic'):
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(max_features=1500, stop_words='english')
        self.model = None
    
    def create_model(self):
        models = {
            'naive_bayes': MultinomialNB(alpha=0.1),
            'logistic': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                C=1.0,
                penalty='l2'
            ),
            'svm': SVC(
                probability=True, 
                random_state=42,
                kernel='linear',
                C=1.0
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            )
        }
        self.model = models.get(self.model_type, LogisticRegression())
    
    def train(self, X_train, y_train):
        if self.model is None:
            self.create_model()
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def evaluate(self, y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }