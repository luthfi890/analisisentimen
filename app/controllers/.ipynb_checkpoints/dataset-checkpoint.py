from flask import render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
from app.models.Dataset import *
from app.models.Detail import *
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from collections import Counter
import os
import re
import nltk
import json
import pandas as pd
import numpy as np
nltk.download('punkt')
nltk.download('stopwords')

# konfigurasi file
conf = {
    'UPLOAD_EXTENSIONS': ['.csv', '.xlsx']
}

def index():
    data = Dataset.get().serialize()
    return render_template('pages/dataset/index.html', data=data, segment='dataset')

def store(request):
    post = request.form # Berisi data dari form HTML
    # Menyimpan nama_data kedalam database
    dataset = Dataset()
    dataset.nama_data = post['nama_data']
    dataset.save()
    
    uploaded_file = request.files['file']
    filename      = secure_filename(uploaded_file.filename)

    file_ext = os.path.splitext(filename)[1]
    if file_ext not in conf['UPLOAD_EXTENSIONS']:
        flash('Tipe file tidak sesuai!', 'danger')
        return redirect(url_for('dataset_index'))

    # Upload file to static with new name
    uploaded_file.save("static/import_data" + file_ext)

    if file_ext == '.csv':
        # Read uploaded file
        df = pd.read_csv("static/import_data.csv")
        df = df.replace(np.nan, 'EMPTY')
    elif file_ext == '.xlsx':
        # Read uploaded file
        df = pd.read_excel("static/import_data.xlsx")
        df = df.replace(np.nan, 'EMPTY')

    # -------------------- PREPROCESSING ------------------------
    stem_data  = []
    disable_duplicate = True
    for index, r in df.iterrows():
        print('Proccessing Index '+str(index)+'... ', end = '')

        text = r['content']
        
        if text != 'EMPTY':
            _casefolding     = caseFolding(text)
            _normalisasi     = normalisasi_kata(_casefolding)
            _tokenizing      = tokenizing(_normalisasi)
            _stopwordRemoval = stopwordRemoval(_tokenizing)
            _stemming        = stemming(_stopwordRemoval)

            rating = r['score']
            if rating < 3:
                _label = 'Negatif'
            elif rating == 3:
                _label = 'Netral'
            elif rating > 3:
                _label = 'Positif'
            
            
            _process = False
            if disable_duplicate == True:
                if _stemming not in stem_data: # If stem text not in list
                    _process = True
                    stem_data.append(_stemming)
                else:
                    print('Skip.! (Duplicate data)', end=' ')
            else:
                _process = True
            
            if _process == True:
                input_data = {
                    'dataset_id'      : dataset.serialize()['id'],
                    "username"        : r['userName'],
                    "review"          : text,
                    "casefolding"     : _casefolding,
                    "normalization"   : _normalisasi,
                    "tokenizing"      : json.dumps(_tokenizing),
                    "stopwordRemoval" : _stopwordRemoval,
                    "stemming"        : _stemming,
                    "score"           : rating,
                    "label"           : _label
                }
                Detail.insert(input_data)
            print('Done.!')

    data = Detail.get().serialize()   
    datasets = pd.DataFrame(data)
    datasets = datasets.drop(columns=['id', 'created_at', 'updated_at', 'deleted_at'])
    datasets = datasets.dropna()
    # datasets.to_csv('static/hasil_preprocessing.csv', index=False)

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

def detail_data(id):
    nama_data = Dataset.where('id', id).select('nama_data').first().serialize()
    data      = Detail.where('dataset_id', id).get().serialize()
    _all      = Detail.where('dataset_id', id).count()
    negatif   = Detail.where('dataset_id', id).where('label', '=', 'Negatif').count()
    netral    = Detail.where('dataset_id', id).where('label', '=', 'Netral').count()
    positif   = Detail.where('dataset_id', id).where('label', '=', 'Positif').count()
    return render_template('pages/dataset/detail.html', data=data, nama_data=nama_data, all=_all, negatif=negatif, netral=netral, positif=positif, segment='dataset')

def delete(id):
    try:
        delete = Dataset.find(id).delete()
        del_detail = Detail.where('dataset_id', id).delete()
        flash('Data berhasil di hapus.!', 'success')
        return redirect(url_for("dataset_index"))
    except Exception as e:
        return 'Something went wrong ' + str(e)

def dataset_reset():
    Dataset.truncate()
    Detail.truncate()
    flash('Data berhasil direset.!', 'success')
    return redirect(url_for('dataset_index'))

def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                    "]+", re.UNICODE)
    return re.sub(emoj, '', data)

def caseFolding(text):
    text = str(text).lower()
    # remove URLs
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', text)
    # remove usernames
    text = re.sub('@[^\s]+', '', text)
    # remove the # in #hashtag
    text = re.sub('#([^\s]+)', '', text)
    # Remove digits number
    text = re.sub("\d+", "", text)
    # Remove multiple dots and replace with a single space
    text = re.sub(r'\.{2,}', ' ', text)
    # Remove punctuations except for single dots and replace them with a space
    text = re.sub(r'[^\w\s\.]', ' ', text)
    # Remove single dots and replace them with a space
    text = re.sub(r'\.', ' ', text)
    # Remove Emojis
    text = remove_emojis(text)
    # Remove extra spaces
    text = ' '.join(text.split())
    return text

def normalisasi_kata(kata):
    data_normalisasi  = pd.read_excel('static/corpus/Normalisasi.xlsx') # load kamus kata normalisasi
    kata_normalisasi  = data_normalisasi['Kata'].tolist() # kata sebelum normalisasi
    hasil_normalisasi = data_normalisasi['Normalisasi'].tolist()# kata sesudah normalisasi
    token = kata.split()
    for t in range(len(token)):
        if token[t] in kata_normalisasi:
            index_kata = kata_normalisasi.index(token[t])
            #print('Kata ', token[t], ' telah di normalisasi menjadi ', hasil_normalisasi[index_kata])
            token[t] = hasil_normalisasi[index_kata].lower()
    return " ".join(token)

def tokenizing(text):
    min_len_token = 2 # Minimal huruf pada kata
    text = nltk.tokenize.word_tokenize(text)
    text = [t for t in text if len(t) > min_len_token]
    return text

def stopwordRemoval(token):
    # ----- START Setup Stopword and Additional Stopword
    stopword             = set(stopwords.words('indonesian'))
    additional_stopwords = pd.read_excel('static/corpus/Additional_Stopwords.xlsx')['Stopwords'].tolist() #load kamus stopwords
    new_stopwords        = stopword.union(additional_stopwords)
    # ----- END
    filter_ = [t for t in token if t not in additional_stopwords]
    return " ".join(filter_)

def stemming(filtered_token):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    words = filtered_token.split()
    backstr = []
    for w in words:
        stemmed = stemmer.stem(w)
        backstr.append(stemmed)

    return ' '.join(backstr)
