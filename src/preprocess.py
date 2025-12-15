import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd

class TextPreprocessor:
    '''Metin ön işleme sınıfı'''
    
    def __init__(self):
        # NLTK verilerini indir
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        '''Metni temizle'''
        if not isinstance(text, str):
            return ''
        
        # Küçük harf
        text = text.lower()
        
        # Noktalama ve rakamları temizle
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Stopwords kaldır
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        
        # Lemmatization
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df, text_column='text'):
        '''DataFrame'i temizle'''
        print('📝 Metinler temizleniyor...')
        df_clean = df.copy()
        df_clean['cleaned_text'] = df_clean[text_column].apply(self.clean_text)
        return df_clean