import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as w
w.filterwarnings('ignore')
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

book_pivot = pd.read_csv('book_pivot.csv')

pipeline = Pipeline([('knn', NearestNeighbors(n_neighbors=6))])

pipeline.fit(book_pivot.values)


def get_recommendations(book_title, data, model_pipeline):
    book_index = data.index.get_loc(book_title)

    distances, indices = model_pipeline.named_steps['knn'].kneighbors(book_pivot.values)

    similar_books = []
    for i in range(1, 6):  # 1'den başlayarak 5'e kadar döngüyü dolaşacak
        similar_books.append({
            'Book Title': book_pivot.index[indices.flatten()[i]],
            'Similarity Score': 1 - distances.flatten()[i]  # Cosine similarity ölçüsü kullanıldığından 1 - mesafe
        })

    return similar_books

input_book = "Voyager"
similar_books = get_recommendations(input_book,book_pivot,pipeline)

# Sonuçları gösterin
print(f"Kullanıcı için '{input_book}' kitabına en benzer 5 kitap:")

for i, book in enumerate(similar_books, start=1):
    print(f"{i}. Kitap: {book['Book Title']}, Benzerlik Skoru: {book['Similarity Score']}")
"""
tavsiye_isim = []
tavsiye_yazar = []
tavsiye_url = []
for i, book in enumerate(similar_books, start=1):
    tavsiye_isim.append(book['Book Title'])
    tavsiye_yazar.append(books.iloc[books['Book Title'] == book['Book Title']]['Book-Author'])
    tavsiye_url.append(books.iloc[books['Book Title'] == book['Book Title']]['Image-URL-M'])

books.head()
"""
tavsiye_isim = []
tavsiye_yazar = []
tavsiye_url = []

for i, book in enumerate(similar_books, start=1):
    tavsiye_isim.append(book['Book Title'])
    # books DataFrame'inde 'Book Title' sütununu filtreleyin ve 'Book-Author' ve 'Image-URL-M' değerlerini alın
    filtered_book = books.loc[books['Book-Title'] == book['Book Title']]
    tavsiye_yazar.append(filtered_book['Book-Author'].iloc[0])  # İlk yazar değerini almak için .iloc[0] kullanın
    tavsiye_url.append(filtered_book['Image-URL-M'].iloc[0])  # İlk URL değerini almak için .iloc[0] kullanın

tavsiye_isim
tavsiye_yazar
tavsiye_url

joblib.dump(pipeline, 'knn_book_recommender_pipeline.pkl')

