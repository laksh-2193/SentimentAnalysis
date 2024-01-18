import nltk
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import re,string,unicodedata
from keras.preprocessing import text, sequence
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk import pos_tag
from nltk.corpus import wordnet
import keras
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,Dropout
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
import numpy as np

nltk.download('stopwords')
stop=set(stopwords.words('english'))
punctuation=list(string.punctuation)
stop.update(punctuation)

def   clean_text(raw_text):
    text_without_html = BeautifulSoup(raw_text, "html.parser").get_text()
    cleaned_text = re.sub(r"[^a-zA-Z0-9\s]", "", text_without_html)

    return cleaned_text
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+|\S+\.\S+')
    return re.sub(url_pattern, '', text)
def remove_stopwords(text):
    final_text=[]
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    return " ".join(final_text)
def denoise_text(text) :
    text = text.lower()
    text = clean_text(text)
    text = remove_urls(text)
    text = remove_between_square_brackets(text)
    text = remove_stopwords(text)
    return text
