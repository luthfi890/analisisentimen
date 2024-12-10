from flask import render_template, flash, redirect, url_for, request
from app.models.Dataset import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.decomposition import PCA

def index():
    # List kernel yang akan digunakan
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    kernel  = ''
    balancingData = ''
    train_df = ''
    test_df = ''
    accuracy_= ''
    precision_ = ''
    recall_ = ''
    f1_score_ = '' 
    show_data = 0

    Negatif_train = ''
    Netral_train  = ''
    Positif_train = ''

    Negatif_test = ''
    Netral_test  = ''
    Positif_test = ''

    Negatif_pred = ''
    Netral_pred  = ''
    Positif_pred = ''

    if len(request.args) > 0:
        show_data = 1
        kernel    = request.args['kernel']
        balancingData = request.args['balancingData']

        data = Dataset.get().serialize()

        df = pd.DataFrame(data)
        # df = df.head(100)
        df = df.drop(columns=['id', 'created_at', 'updated_at', 'deleted_at'])
        df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(n=min(len(x), 140), random_state=42))
        df = df.dropna()
        print(df)

        # -------------------- Setup X & y --------------------------------
        #mengubah kolom 'review' dan 'target' dari DataFrame df menjadi numpy array
        review = df['stemming'].values
        label = df['label'].values

        # -------------------- LabelEncoder ---------------------------------
        # Membuat instance LabelEncoder
        le = LabelEncoder()
        # Fit dan transform label
        y_encoded = le.fit_transform(label)
        # Mengubah nilai 0, 1, 2 menjadi -1, 0, 1
        y_encoded = np.where(y_encoded == 0, -1, y_encoded - 1)
        print(y_encoded)
        print(le.classes_)

        # ------------------ TF-IDF -----------------------------
        # TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(norm='l2') # Membuat TF ID-F Vectorizer
        X_vector   = vectorizer.fit_transform(review).toarray()

        # untuk memperoleh feature name kata
        feature_names = vectorizer.get_feature_names_out()

        # ---------------------- Split Data ----------------------------------
        X_train, X_test, y_train, y_test = train_test_split(X_vector, y_encoded, test_size=0.2, random_state=42)

        # ---------------------- SMOTE --------------------------------------
        if balancingData == '1':
            # Inisialisasi SMOTE
            smote = SMOTE(random_state=42)

            # Terapkan SMOTE hanya pada training data
            X_train, y_train = smote.fit_resample(X_train, y_train)


        # Membuat train_df dan test_df
        train_texts = inverse_tfidf(X_train, feature_names)
        test_texts = inverse_tfidf(X_test, feature_names)

        train_df = pd.DataFrame({
            'stemming': train_texts,
            'label': y_train
        })

        test_df = pd.DataFrame({
            'stemming': test_texts,
            'label': y_test
        })

        # Menampilkan hasil
        print("Train DataFrame:")
        print(train_df)
        print("\nTest DataFrame:")
        print(test_df)

        Negatif_train = train_df.label[train_df.label == -1].count()
        Netral_train  = train_df.label[train_df.label == 0].count()
        Positif_train = train_df.label[train_df.label == 1].count()

        path_hasil = 'static/hasil/'
        # Membuat direktori jika belum ada
        if not os.path.exists(path_hasil):
            os.makedirs(path_hasil)

        # Inisialisasi model SVM dengan kernel yang dipilih
        svm_model = SVC(kernel=kernel)
        
        # Melatih model dengan data training
        svm_model.fit(X_train, y_train)

        # Reduce dimensions with PCA
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # Train SVM on reduced data with best parameters
        svm_best = SVC(kernel=kernel)
        svm_best.fit(X_train_pca, y_train)

        # Visualize the hyperplane
        coordinates = plot_decision_boundary(X_train_pca, y_train, svm_best, 'Hyperlane')
        
        # Prediksi hasil untuk data testing
        y_pred = svm_model.predict(X_test)

        # Menambahkan hasil prediksi ke kolom baru di test_df
        test_df['Predictions'] = y_pred

        Confusion_matrix = confusion_matrix(y_test, y_pred, labels=[-1, 0, 1])
        class_label      = le.classes_
        df_confusion     = pd.DataFrame(Confusion_matrix, index = class_label, columns = class_label)

        sns.heatmap(df_confusion, annot=True, fmt = "d", cmap=plt.cm.Blues)
        plt.title(f'Confusion matrix kernel {kernel}')
        plt.xlabel('Prediction Label')
        plt.ylabel('Actual Label')
        plt.savefig('static/hasil/confusion_matrix.jpg')
        plt.close()
        
        # Menghitung metrik evaluasi
        accuracy_  = round(accuracy_score(y_test, y_pred) * 100, 2)
        precision_ = round(precision_score(y_test, y_pred, average='weighted') * 100, 2)
        recall_    = round(recall_score(y_test, y_pred, average='weighted') * 100, 2)
        f1_score_  = round(f1_score(y_test, y_pred, average='weighted') * 100, 2)

        print(test_df)
        Negatif_test = test_df.label[test_df.label == -1].count()
        Netral_test  = test_df.label[test_df.label == 0].count()
        Positif_test = test_df.label[test_df.label == 1].count()

        Negatif_pred = test_df.Predictions[test_df.Predictions == -1].count()
        Netral_pred  = test_df.Predictions[test_df.Predictions == 0].count()
        Positif_pred = test_df.Predictions[test_df.Predictions == 1].count()

        test_df.to_csv(f'{path_hasil}data test {kernel} {balancingData}.csv')

    return render_template('pages/algoritma.html', segment='algoritma', train_df=train_df,
    test_df=test_df, kernels=kernels, show_data=show_data, kernel=kernel, balancingData=balancingData,
    accuracy_=accuracy_, precision_=precision_, recall_=recall_, f1_score_=f1_score_,
    Negatif_train=Negatif_train, Netral_train=Netral_train, Positif_train=Positif_train,
    Negatif_test=Negatif_test, Netral_test=Netral_test, Positif_test=Positif_test,
    Negatif_pred=Negatif_pred, Netral_pred=Netral_pred, Positif_pred=Positif_pred)

# Fungsi untuk mengembalikan vektor TF-IDF ke teks
def inverse_tfidf(X_vector, feature_names):
    texts = []
    for vector in X_vector:
        # Ambil indeks kata dengan TF-IDF > 0
        indices = np.where(vector > 0)[0]
        # Ambil kata-kata asli berdasarkan indeks
        words = [feature_names[index] for index in indices]
        # Gabungkan kata menjadi satu kalimat
        reconstructed_text = " ".join(words)
        texts.append(reconstructed_text)
    return texts

def plot_decision_boundary(X, y, model, title):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=plt.cm.Paired)
    
    # Handle the legend
    handles, labels = scatter.legend_elements()
    unique_labels = ['Negatif', 'Netral', 'Positif']
    plt.legend(handles, unique_labels)
    
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    # plt.title(title)
    plt.savefig(f'static/{title}.jpg')
    plt.close()

    # Mengumpulkan koordinat setiap titik
    coordinates = [(round(X[i, 0], 2), round(X[i, 1], 2)) for i in range(X.shape[0])]
    return coordinates