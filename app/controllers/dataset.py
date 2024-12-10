from flask import render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
from app.models.Dataset import *
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from collections import Counter
from nltk.tokenize import word_tokenize
import os
import re
import csv
import nltk
import json
import pandas as pd
import numpy as np
nltk.download('punkt')
nltk.download('stopwords')

# konfigurasi file
conf = {
    'UPLOAD_EXTENSIONS': ['.xlsx']
}

def index():
    data      = Dataset.get().serialize()
    _all      = Dataset.count()
    negatif   = Dataset.where('label', '=', 'Negatif').count()
    netral    = Dataset.where('label', '=', 'Netral').count()
    positif   = Dataset.where('label', '=', 'Positif').count()

    return render_template('pages/dataset/index.html', data=data, all=_all, negatif=negatif, 
    netral=netral, positif=positif, segment='dataset')

def store(request):
    post = request.form # Berisi data dari form HTML
    
    uploaded_file = request.files['file']
    filename      = secure_filename(uploaded_file.filename)

    file_ext = os.path.splitext(filename)[1]
    if file_ext not in conf['UPLOAD_EXTENSIONS']:
        flash('Tipe file tidak sesuai.', 'danger')
        return redirect(url_for('dataset_index'))

    # Upload file to static with new name
    uploaded_file.save("static/import_data" + file_ext)

    # Read uploaded file
    df = pd.read_excel("static/import_data.xlsx")
    df = df.replace(np.nan, 'EMPTY')

    # -------------------- PREPROCESSING ------------------------
    for index, r in df.iterrows():
        print('Proccessing Index '+str(index)+'... ', end = '')

        text = r['review']
        
        if text != 'EMPTY':
            _casefolding     = case_folding(text)
            _cleaning        = cleaning(_casefolding)
            _tokenizing      = tokenizing(_cleaning)
            _normalisasi     = normalization(_tokenizing)
            _stopwordRemoval = stopwordRemoval(_normalisasi)
            _stemming        = stemming(_stopwordRemoval)
            _labelling       = label_text(_stemming)
            
            input_data = {
                "username"        : r['profile_name'],
                "review"          : text,
                "casefolding"     : _casefolding,
                "cleaning"        : _cleaning,
                "tokenizing"      : json.dumps(_tokenizing),
                "normalization"   : _normalisasi,
                "stopwordRemoval" : json.dumps(_stopwordRemoval),
                "stemming"        : _stemming,
                "label"           : _labelling
            }
            Dataset.insert(input_data)
            print('Done.!')

    data = Dataset.get().serialize()   
    datasets = pd.DataFrame(data)
    datasets = datasets.drop(columns=['id', 'created_at', 'updated_at', 'deleted_at'])
    datasets = datasets.dropna()
    
    word_counts = Counter()

    for i, r in datasets.iterrows():
        stem = r['stemming'].split()
        word_counts.update(stem)

    word_counts_df = pd.DataFrame.from_dict(word_counts, orient='index', columns=['count'])
    word_counts_df = word_counts_df.sort_values(by='count', ascending=False)
    word_counts_df.to_csv('static/words.csv')

    # word_counts_index = word_counts_df.sort_index()
    # word_counts_index.to_csv('static/words_index.csv')

    # -------------------- END PREPROCESSING ------------------------
    flash('Data berhasil di simpan.!', 'success')
    return redirect(url_for('dataset_index'))

def dataset_reset():
    Dataset.truncate()
    flash('Data berhasil direset.!', 'success')
    return redirect(url_for('dataset_index'))

# https://www.w3schools.com/python/ref_string_casefold.asp#:~:text=Definition%20and%20Usage.%20The%20casefold()%20method%20returns%20a%20string%20where
def case_folding(text):
    return text.casefold()

# https://medium.com/@yashj302/text-cleaning-using-regex-python-f1dded1ac5bd#:~:text=We%E2%80%99ll%20use%20re.sub-%3E%20%E2%80%9CReturn%20the%20string%20obtained%20by%20replacing%20the
def cleaning(text):
    # Menghapus URL
    text = re.sub(r'http[s]?\://\S+', '', text)

    # Menghapus hashtag
    text = re.sub(r'#\S+', '', text)

    # Menghapus mentions (@)
    text = re.sub(r"@\S+", "", text)

    # Menghapus teks dalam tanda kurung
    text = re.sub(r"(\(.*\))|(\[.*\])", "", text)

    # Menghapus karakter baris atau tab (\n, \r, \t..)
    text = re.sub(r"\n|\r|\t", " ", text)

    # Menghapus tanda baca
    text = re.sub(r'[^\w\s]', '', text)

    # Menghapus semua karakter yang bukan huruf alfabet
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Menghapus spasi yang berlebihan
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# https://medium.com/@utkarsh.kant/tokenization-a-complete-guide-3f2dd56c0682#:~:text=While%20NLTK%20offers%20multiple%20different%20tokenizers%20for%20the%20same%20task,
def tokenizing(text):
    min_len_token = 2 # Minimal huruf pada kata
    text = nltk.tokenize.word_tokenize(text)
    text = [t for t in text if len(t) > min_len_token]
    return text

def normalization(token):
    data_normalisasi  = pd.read_excel('static/corpus/Normalisasi.xlsx') # load kamus kata normalisasi
    kata_normalisasi  = data_normalisasi['Kata'].tolist() # kata sebelum normalisasi
    hasil_normalisasi = data_normalisasi['Normalisasi'].tolist()# kata sesudah normalisasi
    # token = kata.split()
    for t in range(len(token)):
        if token[t] in kata_normalisasi:
            index_kata = kata_normalisasi.index(token[t])
            #print('Kata ', token[t], ' telah di normalisasi menjadi ', hasil_normalisasi[index_kata])
            token[t] = hasil_normalisasi[index_kata].lower()
    return " ".join(token)

# https://medium.com/analytics-vidhya/removing-stop-words-with-nltk-library-in-python-f33f53556cc1#:~:text=How%20to%20remove%20stop%20words%20with%20NLTK%20library%20in%20Python.
def stopwordRemoval(text):
    token = text.split()
    # ----- START Setup Stopword and Additional Stopword
    stopword             = set(stopwords.words('indonesian'))
    # additional_stopwords = pd.read_excel('static/corpus/Additional_Stopwords.xlsx')['Stopwords'].tolist() #load kamus stopwords
    # new_stopwords        = stopword.union(additional_stopwords)
    # ----- END
    filter_ = [t for t in token if t not in stopword]
    return " ".join(filter_)

# https://medium.com/@93Kryptonian/stemming-with-sastrawi-877cc40a37ad#:~:text=Sastrawi%20Python%20is%20a%20simple%20python%20library%20which%20allows%20you
def stemming(filtered_token):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    words = filtered_token.split()
    backstr = []
    for w in words:
        stemmed = stemmer.stem(w)
        backstr.append(stemmed)

    return ' '.join(backstr)

def load_lexicon_with_weights(file_path):
    lexicon = {}
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')  # Membaca file .tsv
        for row in reader:
            word = row['word'].strip().lower()
            weight = int(row['weight'])
            lexicon[word] = weight
    return lexicon

# Memuat positive.tsv dan negative.tsv
positive_words = load_lexicon_with_weights('static/corpus/positive.tsv')
negative_words = load_lexicon_with_weights('static/corpus/negative.tsv')

def label_text(text):
    words = text.split()
    
    # Hitung total bobot kata positif dan negatif
    positive_score = sum(positive_words.get(word, 0) for word in words)
    negative_score = sum(negative_words.get(word, 0) for word in words)
    score = positive_score + negative_score
    
    # Tetapkan label berdasarkan total bobot kata positif dan negatif
    if score > 0:
        return 'Positif'
    elif score < 0:
        return 'Negatif'
    else:
        return 'Netral'

