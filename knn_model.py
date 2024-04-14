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

##Read Data sets

books = pd.read_csv("Proje/Books.csv")
users = pd.read_csv("Proje/Users.csv")
ratings = pd.read_csv("Proje/Ratings.csv")


books.dropna(inplace=True)

valid_years = books['Year-Of-Publication'].astype(str).str.isnumeric()
books = books[valid_years]
books['Year-Of-Publication'] = books['Year-Of-Publication'].astype(int)
books['Publication_Date'] = pd.to_datetime(books['Year-Of-Publication'], format='%Y', errors='coerce')
books.drop(columns=['Year-Of-Publication'], inplace=True)
books = pd.DataFrame(books)
books['Year-Of-Publication'] = books['Publication_Date'].dt.year
books['Year-Of-Publication'].value_counts().index.values

books = books[~books['Year-Of-Publication'].isin([2037, 2026, 2030, 2050, 2038])]


mean_age = users['Age'].mean()
users['Age'].fillna(mean_age, inplace=True)



def outlier_thresholds(dataframe, col_name, q1=0.5, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(users, 'Age', q1=0.5, q3=0.95)

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(users, "Age")



rating_books_name = ratings.merge(books,on='ISBN')

numer_rating = rating_books_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
numer_rating.rename(columns={'Book-Rating':'Totle_number_rating'},inplace=True)

avg_rating = rating_books_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
avg_rating.rename(columns={'Book-Rating':'Totle_avg_rating'},inplace=True)

popular_df = numer_rating.merge(avg_rating,on='Book-Title')

popular_df = popular_df[popular_df['Totle_number_rating'] >= 150].sort_values('Totle_avg_rating',ascending=False)

popular_df = popular_df.merge(books,on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author', 'Image-URL-M', 'Totle_number_rating','Totle_avg_rating']]


x = ratings['User-ID'].value_counts() > 150
y = x[x].index

ratings = ratings[ratings['User-ID'].isin(y)]

rating_with_books = ratings.merge(books, on='ISBN')

number_rating = rating_with_books.groupby('Book-Title')['Book-Rating'].count().reset_index()
number_rating.rename(columns= {'Book-Rating':'number_of_ratings'}, inplace=True)

final_rating = rating_with_books.merge(number_rating, on='Book-Title')
final_rating = final_rating[final_rating['number_of_ratings'] >= 25]
final_rating.drop_duplicates(['User-ID','Book-Title'], inplace=True)

merged_data = final_rating.merge(users, on='User-ID', how='inner')

book_pivot = final_rating.pivot_table(columns='User-ID', index='Book-Title', values="Book-Rating")
book_pivot.fillna(0, inplace=True)

book_pivot.head()


pipeline = Pipeline([('knn', NearestNeighbors(n_neighbors=6))])

pipeline.fit(book_pivot.values)

""" buradaki ilk denemeydi
def get_recommendations(book_title, data, model_pipeline):
    book_index = data.index.get_loc(book_title)

    distances, indices = model_pipeline.named_steps['knn'].kneighbors(book_pivot.values)

    similar_books = []
    for i in range(1, len(indices.flatten())):
        similar_books.append({
            'Book Title': book_pivot.index[indices.flatten()[i]],
            'Similarity Score': 1 - distances.flatten()[i]  # Cosine similarity ölçüsü kullanıldığından 1 - mesafe
        })

    return similar_books
"""

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

