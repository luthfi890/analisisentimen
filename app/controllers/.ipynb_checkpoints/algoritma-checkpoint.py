from flask import render_template, flash, redirect, url_for, request
from app.models.Dataset import *
from app.models.Detail import *
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

def index():
    list_data = Dataset.get().serialize()
    show_data = 0
    nama_data = ''
    path_hasil = ''
    evaluasi = ''
    train_data = ''
    test_data = ''
    if len(request.args) > 0:
        show_data = 1
        nama_data = request.args['namaData']

        dataset = Dataset.where('nama_data', nama_data).first().serialize()
        detailData = Detail.where('dataset_id', dataset['id']).get().serialize()

        df = pd.DataFrame(detailData)
        df = df.drop(columns=['id', 'dataset_id', 'created_at', 'updated_at', 'deleted_at'])
        df = df.dropna()
        print(df)

        # -------------------- Setup X & y --------------------------------
        X = list()
        y = list()
        for index, row in df.iterrows():
            X.append(row['stemming'])
            y.append(row['label'])

        # -------------------- LabelEncoder ---------------------------------
        # Membuat instance LabelEncoder
        le = LabelEncoder()
        # Fit dan transform label
        y_encoded = le.fit_transform(y)
        # Mengubah nilai 0, 1, 2 menjadi -1, 0, 1
        # y_encoded = np.where(y_encoded == 0, -1, y_encoded - 1)
        print(le.classes_)

        # ---------------------- Split Data ----------------------------------
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        # Path direktori untuk membuat folder hasil dan folder sesuai nama data yang dipilih didalam folder static
        path_hasil = os.path.join('static', 'hasil', nama_data)
        # Membuat direktori jika belum ada
        if not os.path.exists(path_hasil):
            os.makedirs(path_hasil)

        # Folder split data untuk menyimpan data train dan test 
        path_split = os.path.join(path_hasil, 'split data')
        # Membuat direktori jika belum ada
        if not os.path.exists(path_split):
            os.makedirs(path_split)

        # Visualisasi Train Data
        train_data = {'stemming':list(), 'label':list()}
        for xt in range(len(X_train)):
            train_data['stemming'].append(X_train[xt])
            train_data['label'].append(y_train[xt])
        train_data = pd.DataFrame(train_data)
        train_data.to_csv(path_split+'/Train data.csv', index=False)

        # Visualisasi Test Data
        test_data = {'stemming':list(), 'label':list()}
        for xt in range(len(X_test)):
            test_data['stemming'].append(X_test[xt])
            test_data['label'].append(y_test[xt])
        test_data = pd.DataFrame(test_data)
        test_data.to_csv(path_split+'/Test data.csv', index=False)

        # ------------------ TF-IDF -----------------------------
        # TF-IDF Vectorizer
        vectorizer    = TfidfVectorizer(norm='l2') # Membuat TF ID-F Vectorizer
        X_train_vec   = vectorizer.fit_transform(X_train).toarray() # Mencari vocabulary
        X_test_vec    = vectorizer.transform(X_test).toarray()

        # untuk memperoleh feature name kata
        feature_names = vectorizer.get_feature_names_out() # Mendapatkan Feature Names matrix

        #menghitung frekuensi
        count_vector  = CountVectorizer()
        X_train_count = count_vector.fit_transform(X_train).toarray()
        X_test_count  = count_vector.transform(X_test).toarray()

        # Folder tfidf untuk menyimpan hasil tfidf dari data train dan test
        path_tfidf = os.path.join(path_hasil, 'tfidf')
        # Membuat direktori jika belum ada
        if not os.path.exists(path_tfidf):
            os.makedirs(path_tfidf)

        # Menyimpan hasil TFIDF sebelum seleksi fitur chi square
        tf_train = pd.DataFrame(X_train_count,index = [f'D{i+1}' for i in range(len(X_train))],columns=feature_names)
        tf_train.to_csv(path_tfidf+'/frekuensi data train.csv', index=False)

        tfidf_train = pd.DataFrame(X_train_vec,index = [f'D{i+1}' for i in range(len(X_train))],columns=feature_names)
        tfidf_train['label'] = y_train
        tfidf_train.to_csv(path_tfidf+'/TF-IDF data train.csv', index=False)

        tf_test = pd.DataFrame(X_test_count,index = [f'D{i+1}' for i in range(len(X_test))],columns=feature_names)
        tf_test.to_csv(path_tfidf+'/frekuensi data test.csv', index=False)

        tfidf_test = pd.DataFrame(X_test_vec,index = [f'D{i+1}' for i in range(len(X_test))],columns=feature_names)
        tfidf_test['label'] = y_test
        tfidf_test.to_csv(path_tfidf+'/TF-IDF data test.csv', index=False)

        # --------------- Feature Selection Chi-square -----------------
        # `percentile` adalah variable untuk menentukan berapa persen selected feature yang ingin di ambil
        chi2_selector = SelectPercentile(chi2, percentile=90)

        # Apply fit_transform ke data train
        X_train_chi2 = chi2_selector.fit_transform(X_train_count, y_train)

        # Apply transform ke data test
        X_test_chi2 = chi2_selector.transform(X_test_count)

        print('Original number of features:', X_train_count.shape[1])
        print('Reduced number of features:', X_train_chi2.shape[1])

        # Mengambil feature names yang dipilih
        selected_feature_names = chi2_selector.get_feature_names_out(feature_names)

        # --------------- Mengembalikan ke bentuk teks -----------------
        def vector_to_text(vector, feature_names):
            text = []
            for idx, value in enumerate(vector):
                if value > 0:
                    text.append(feature_names[idx])
            return ' '.join(text)

        # Mengonversi data train kembali ke bentuk teks
        selected_texts_train = []
        for vec in X_train_chi2:
            selected_texts_train.append(vector_to_text(vec, selected_feature_names))

        # Mengonversi data test kembali ke bentuk teks
        selected_texts_test = []
        for vec in X_test_chi2:
            selected_texts_test.append(vector_to_text(vec, selected_feature_names))

        # Menyimpan hasil teks yang telah dipilih fiturnya ke file CSV
        train_data_text = pd.DataFrame({'text': selected_texts_train, 'label': y_train})
        train_data_text.to_csv(path_split+'/Train data-chi square.csv', index=False)

        test_data_text = pd.DataFrame({'text': selected_texts_test, 'label': y_test})
        test_data_text.to_csv(path_split+'/Test data-chi square.csv', index=False)

        # TF-IDF Vectorizer setelah seleksi fitur dengan Chi-Square
        vectorizer_chi2 = TfidfVectorizer(vocabulary=selected_feature_names, norm='l2')

        X_train_vec_chi2 = vectorizer_chi2.fit_transform(X_train).toarray()
        X_test_vec_chi2 = vectorizer_chi2.transform(X_test).toarray()

        # Menyimpan hasil TFIDF sebelum seleksi fitur chi square
        tfidf_train_chi2 = pd.DataFrame(X_train_vec_chi2, index=[f'D{i+1}' for i in range(len(X_train))], columns=selected_feature_names)
        tfidf_train_chi2['label'] = y_train
        tfidf_train_chi2.to_csv(path_tfidf+'/TF-IDF data train-chisquare.csv', index=False)

        tfidf_test_chi2 = pd.DataFrame(X_test_vec_chi2, index=[f'D{i+1}' for i in range(len(X_test))], columns=selected_feature_names)
        tfidf_test_chi2['label'] = y_test
        tfidf_test_chi2.to_csv(path_tfidf+'/TF-IDF data test-chisquare.csv', index=False)

        selected_features_train = pd.DataFrame(X_train_chi2, index=[f'D{i+1}' for i in range(len(X_train))], columns=selected_feature_names)
        selected_features_train['label'] = y_train
        selected_features_train.to_csv(path_tfidf+'/frekuensi data train-chisquare.csv', index=False)

        selected_features_test = pd.DataFrame(X_test_chi2, index=[f'D{i+1}' for i in range(len(X_test))], columns=selected_feature_names)
        selected_features_test['label'] = y_test
        selected_features_test.to_csv(path_tfidf+'/frekuensi data test-chisquare.csv', index=False)

        # ---------------------- NAIVE BAYES ---------------------------
        # Menghitung class weights
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

        # Menentukan bobot sampel berdasarkan class weights
        sample_weights = np.array([class_weights_dict[label] for label in y_train])

        clf = MultinomialNB()
        clf.fit(X_train_vec_chi2, y_train, sample_weight=sample_weights)
        y_predict = clf.predict(X_test_vec_chi2)

        test_data['pred_label'] = y_predict

        Confusion_matrix = confusion_matrix(y_test, y_predict)
        class_label      = le.classes_
        df_confusion     = pd.DataFrame(Confusion_matrix, index = class_label, columns = class_label)

        sns.heatmap(df_confusion, annot=True, fmt = "d", cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.xlabel('Prediction Label')
        plt.ylabel('Actual Label')
        plt.savefig(path_hasil+'/confusion_matrix.jpg')
        plt.close()

        _accuracy   = round(accuracy_score(y_test, y_predict)*100, 2)
        _precission = round(precision_score(y_test, y_predict, average='weighted')*100, 2)
        _recall     = round(recall_score(y_test, y_predict, average='weighted')*100, 2)
        _fscore     = round(f1_score(y_test, y_predict, average='weighted')*100, 2)

        print('Accuracy Score   :', _accuracy,'%')
        print('Precission Score :', _precission,'%')
        print('Recall Score     :', _recall,'%')
        print('F-Score Score    :', _fscore,'%')

        evaluasi = {
		'_accuracy': _accuracy,
		'_precission': _precission,
		'_recall': _recall,
		'_fscore': _fscore
        }

    return render_template('pages/algoritma.html', list_data=list_data, nama_data=nama_data, segment='algoritma', train_data=train_data,
    test_data=test_data, show_data=show_data, path_hasil=path_hasil, evaluasi=evaluasi)