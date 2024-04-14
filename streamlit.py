import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors
#pip install streamlit

st.set_page_config(layout = "wide", page_title="Book Recommender", page_icon=(":book:"))


@st.cache_data
def get_data():
    dataframe = pd.read_csv('Books.csv')
    return dataframe

def get_data2():
    dataframe = pd.read_csv('kitap_isimleri.csv')
    return dataframe

def get_data3():
    dataframe = pd.read_csv('book_index_tablosu')
    return dataframe
@st.cache_data
def get_pipeline():
    pipeline = joblib.load('knn_book_recommender_pipeline.pkl')
    return pipeline



st.title("📖 :blue[Kitap Tavsiye Sistemi] 📖")

random_tab, recommendation_tab = st.tabs(["Rastgele Kitap","Kitap Önerisi"])

df = get_data()
df2 = get_data2()
df3 = get_data3()

common_titles = df2["Book-Title"].tolist()

col1, col2, col3 = random_tab.columns(3, gap="small")
columns = [col1, col2, col3]
empty_col1, empty_col2, empty_col3 = random_tab.columns([4,3,2])

if empty_col2.button("Rastgele Kitap Önerisi"):

    random_books = df[~df["Book-Author"].isna()]
    random_books = df[df["Book-Title"].isin(common_titles)].sample(3)

    for i, col in enumerate(columns):

        col.image(random_books.iloc[i]['Image-URL-M'])
        col.write(f"**{random_books.iloc[i]['Book-Title']}**")
        col.write(f"*{random_books.iloc[i]['Book-Author']}*")


# recommendation_tab
kolon1, kolon2= recommendation_tab.columns(2, gap="small")
columns2 = [kolon1, kolon2]
#empty_coll1, empty_coll2= random_tab.columns([1,8])



pipeline = get_pipeline()

selected_book = kolon1.selectbox('Lütfen bir kitap seçin:', df2['Book-Title'].values)


book_index = df3[df3['Book-Title'] == selected_book].index[0]
features = df3.iloc[book_index]
features = features[1:]
features = np.array([features]).reshape(1,-1)

if kolon1.button("Öneri Getir!"):
    distances, indices = pipeline.named_steps['knn'].kneighbors(features)

    recommended_indices = indices[0]
    recommended_books = df2.iloc[recommended_indices]
    #recommendation_tab.write(recommended_books)

    recommended_books_titles = recommended_books.iloc[1:6]["Book-Title"]
    #recommendation_tab.write(recommended_books_titles)

    recommended_books2 = df[df["Book-Title"].isin(recommended_books_titles)]
    #recommendation_tab.write(recommended_books2)

    count = 0
    for i, book in enumerate(recommended_books2.iterrows(), start=1):
        kolon2.image(book[1]['Image-URL-M'])
        kolon2.write(f"**{i}. Kitap: {book[1]['Book-Title']}**")
        kolon2.write(f"*{book[1]['Book-Author']}*")
        count += 1
        if count == 5:
            break



