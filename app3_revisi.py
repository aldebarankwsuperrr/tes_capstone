####
#### Can read data from live spreadsheet
#### 20-11-2023

import urllib.request, json 
import pandas as pd
import streamlit as st
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='dark')

def get_weather(time_now, number=1):
    with urllib.request.urlopen("https://ibnux.github.io/BMKG-importer/cuaca/501193.json") as url:
        weather = json.load(url)    
    weather_dataframe = pd.DataFrame.from_dict(weather, orient='columns')
    weather_dataframe["jamCuaca"] = pd.to_datetime(weather_dataframe["jamCuaca"])
    weather_dataframe = weather_dataframe[(weather_dataframe['jamCuaca'] >= time_now)].head(number)    
    return weather_dataframe

def get_weather_forcast():
    with urllib.request.urlopen("https://ibnux.github.io/BMKG-importer/cuaca/501193.json") as url:
        weather = json.load(url)
    weather = pd.DataFrame(weather)
    weather_forcast = weather.loc[4:6, ["cuaca", "kodeCuaca"]].mode()
    return weather_forcast

def read_data(file_path):
    df = pd.read_csv(file_path)
    return df

def get_time_now():
    return datetime.datetime.now()

def get_weather_icon(weather_dataframe):
    weather_dataframe = weather_dataframe.head(1)
    return "https://ibnux.github.io/BMKG-importer/icon/{id_cuaca}.png".format(id_cuaca=int(weather_dataframe["kodeCuaca"]))

def cleaned_jajan_items():
    jajan_items = read_data("https://docs.google.com/spreadsheets/d/197Z0h89qd2R3Z3m5aWvG4LMRBHokA6DdhjqdUHZdEzk/export?gid=311357912&format=csv")
    jajan_items = jajan_items.drop(['image', 'update_at', 'created_at', 'deleted_at'], axis = 1)
    
    return jajan_items
    
def cleaned_categories():
    categories =read_data("https://docs.google.com/spreadsheets/d/197Z0h89qd2R3Z3m5aWvG4LMRBHokA6DdhjqdUHZdEzk/export?gid=1524768059&format=csv")
    categories = categories.drop(['icon', 'update_at', 'created_at', 'deleted_at', 'Unnamed: 6'], axis = 1)
    categories.dropna(inplace=True)
    categories['tags'] = [
        'cerah, cerah berawan, berawan, berawan tebal', 
        'cerah, cerah berawan, berawan, berawan tebal, hujan, hujan ringan',
        'cerah, cerah berawan, berawan, berawan tebal, hujan, hujan ringan',
        'hujan ringan, hujan sedang, hujan lebat, hujan lokal',
        'cerah, cerah berawan, berawan tebal',
        'cerah, cerah berawan, berawan'
    ]
    
    return categories

def cleaned_transactions():
    transactions = pd.read_csv('https://docs.google.com/spreadsheets/d/197Z0h89qd2R3Z3m5aWvG4LMRBHokA6DdhjqdUHZdEzk/export?gid=1989279773&format=csv')

    transactions.dropna(inplace=True)
    
    return transactions

def Transaction_all_categories(transactions):
    Transaksi_total = transactions.groupby('created_at')['total_transaksi'].sum().reset_index(name='total')
    
    return Transaksi_total

def Transaction_per_categories(transactions):
    Transaksi_kategori = transactions.groupby(['created_at','category_id'])['total_transaksi'].sum().reset_index(name='total')
    Transaksi_kategori = Transaksi_kategori.pivot(index='created_at', columns='category_id', values='total').reset_index()
    Transaksi_kategori.fillna(0, inplace=True)

    return Transaksi_kategori


def measure_total_transaction_vendor(transactions, jajan_items):
    total_transaction_vendor = transactions.groupby(by="vendor_id").agg({'total_transaksi' : 'sum'}).sort_values(by='total_transaksi')
    return total_transaction_vendor

def get_high_transaction_vendor(total_transaction_vendor):
    high_vendor_transaction_quartile =  total_transaction_vendor.quantile(.75, axis = 0)
    high_boundary = high_vendor_transaction_quartile['total_transaksi']
    high_transaction_vendor = total_transaction_vendor[total_transaction_vendor['total_transaksi'] >= high_boundary]
    high_transaction_vendor = high_transaction_vendor.sort_values(by='total_transaksi', ascending=False).head(5)
    
    return high_transaction_vendor

def get_low_transaction_vendor(total_transaction_vendor):
    low_vendor_transaction_quartile =  total_transaction_vendor.quantile(.25, axis = 0)
    low_boundary = low_vendor_transaction_quartile['total_transaksi']
    low_transaction_vendor = total_transaction_vendor[total_transaction_vendor['total_transaksi'] <= low_boundary]
    
    low_transaction_vendor = low_transaction_vendor.sort_values(by='total_transaksi', ascending=True).head(5)
    
    return low_transaction_vendor

def vendor_segmen(total_transaction_vendor):
    total_transaction_vendor_temp = total_transaction_vendor
    high_vendor_transaction_quartile =  total_transaction_vendor.quantile(.75, axis = 0)
    high_boundary = high_vendor_transaction_quartile['total_transaksi']
    
    low_vendor_transaction_quartile =  total_transaction_vendor.quantile(.25, axis = 0)
    low_boundary = low_vendor_transaction_quartile['total_transaksi']
    
    total_transaction_vendor_temp.loc[total_transaction_vendor['total_transaksi'] <= low_boundary,"segmen"] = "Sedikit Laku"
    total_transaction_vendor_temp.loc[total_transaction_vendor['total_transaksi'] >= high_boundary,"segmen"] = "Laku Banyak"
    total_transaction_vendor_temp.loc[(total_transaction_vendor['total_transaksi'] < high_boundary) & (total_transaction_vendor["total_transaksi"] > low_boundary),"segmen"] = "Cukup Laku"
    
    vendor_percentage_segmen = total_transaction_vendor_temp.groupby(['segmen']).size().reset_index(name='Count')
    
    return vendor_percentage_segmen

def user_segmen(transactions):
    
    total_transaction_user = transactions.groupby(by='user_id').agg({'total_transaksi' : 'sum'}).sort_values(by='total_transaksi')
    
    total_transaction_user.sort_values(by=['total_transaksi'],ascending=[False]).head()

    high_user_transaction_quartile = total_transaction_user.quantile(.75, axis=0)
    high_boundary = high_user_transaction_quartile['total_transaksi']
    
    low_user_transaction_quartile = total_transaction_user.quantile(.25, axis=0)
    low_boundary = low_user_transaction_quartile['total_transaksi']
    
    total_transaction_user.loc[total_transaction_user['total_transaksi'] <= low_boundary,"segmen"] = "Uang Jajan Pas Pas an"
    total_transaction_user.loc[(total_transaction_user['total_transaksi'] < high_boundary) &(total_transaction_user['total_transaksi'] > low_boundary),"segmen"] = "Uang Jajan Lumayan"
    total_transaction_user.loc[total_transaction_user['total_transaksi'] >= high_boundary,"segmen"] = "Banyak Uang jajan"
    
    user_percentage_segmen = total_transaction_user.groupby(['segmen']).size().reset_index(name='Count')
    
    return user_percentage_segmen


def get_frequencies(transactions):
    frequencies = transactions.groupby(['jajan_item_id', 'Weather']).size().reset_index(name='count')
    
    return frequencies

def merge_jajan_items_categories(jajan_items, categories):
    merged = jajan_items.merge(categories, left_on='category_id', right_on='category_id')
    
    return merged
    
def create_similarity_vector(df):
    corpus = [
        'cerah',
        'cerah berawan',
        'berawan',
        'berawan tebal',
        'hujan',
        'hujan ringan',
        'hujan sedang',
        'hujan lebat',
        'hujan lokal'
    ]
    
    vector = TfidfVectorizer(max_features=40, ngram_range=(1, 2))
    vector.fit(corpus)
    tfidf_matrix = vector.fit_transform(df['tags'])
    result = pd.DataFrame(
        tfidf_matrix.todense(),
        columns=vector.get_feature_names_out(),
        index=df['jajan_item_name']
    )
    
    column_result = corpus
    result = result[column_result]
    result.reset_index(inplace=True)
    result['jajan_item_id'] = df['jajan_item_id']
    result['category_id'] = df['category_id']
    return result

def bobot_kategori_cuaca(similarity_vector):
    bobot_cuaca_kategori = similarity_vector
    bobot_cuaca_kategori=bobot_cuaca_kategori.groupby('category_id').agg({
        'cerah': 'sum',
        'cerah berawan': 'sum',
        'berawan': 'sum',
        'berawan tebal': 'sum',
        'hujan': 'sum',
        'hujan ringan': 'sum',
        'hujan sedang': 'sum',
        'hujan lebat': 'sum',
        'hujan lokal' : 'sum'
        }).reset_index()
    return bobot_cuaca_kategori

def bobot(cuaca,bobot_cuaca_kategori):
    hasil = bobot_cuaca_kategori.sort_values(by=cuaca,ascending =False)
    return hasil

def weighting(recommendation, transactions_frequencies):
  weather_list =  list(recommendation.columns.values)
  for i in range(len(transactions_frequencies)):
    data = transactions_frequencies.iloc[i]
    cuaca = data['Weather']
    if cuaca in weather_list:
      condition = recommendation['jajan_item_id'] == data['jajan_item_id']
      hasil = recommendation[condition][cuaca]
      recommendation.loc[condition, cuaca] = hasil * (data['count'] + 1)
  return recommendation

def get_recommended (weather, similarity_vector, transactions_frequencies):
  column_list = ['jajan_item_name', 'jajan_item_id']
  for column in list(similarity_vector.columns.values):
    if weather in column:
      column_list.append(column)
  recommended = similarity_vector[column_list]
  recommended = weighting(recommended, transactions_frequencies)
  recommended = recommended.sort_values(by=column_list[2:], ascending=False)
  return recommended.head()

def get_recommendation(recommended, jajan_items):
    recommendation = recommended.merge(jajan_items, left_on=['jajan_item_id', 'jajan_item_name'], right_on=['jajan_item_id', 'jajan_item_name'])
    recommendation = recommendation[['jajan_item_name', 'price']]
    return recommendation

def calculate_category_sum(user_id, transactions):                                                 #1 nov 2023
    user_data = transactions[transactions['user_id'] == user_id]                                  #1 nov 2023
    category_sum = user_data.groupby(['Weather', 'category_id'])['amount'].sum().reset_index()            #1 nov 2023

    return category_sum                                                                             #1 nov 2023

def vendor_from_forcast(similarity_vector, jajan_items, weather, ascending):
    similarity_forcast = similarity_vector[["jajan_item_id", weather]]
    similarity_forcast = similarity_forcast[similarity_forcast[weather] > 0]
    similarity_forcast_merged = similarity_forcast.merge(jajan_items, left_on='jajan_item_id', right_on='jajan_item_id')
    similarity_forcast_merged_cleaned = similarity_forcast_merged.dropna()
    vendor = similarity_forcast_merged_cleaned['vendor_id'].sort_values(ascending=ascending).unique()
    
    
    return vendor
    


### gathering the main files
jajan_items = cleaned_jajan_items()
categories = cleaned_categories()
transactions = cleaned_transactions()

Transaksi_total = Transaction_all_categories(transactions)

Transaksi_kategori= Transaction_per_categories(transactions)

transaksi_vendor = measure_total_transaction_vendor(transactions, jajan_items)

high_transaction_vendor = get_high_transaction_vendor(transaksi_vendor)
low_transaction_vendor = get_low_transaction_vendor(transaksi_vendor)

vendor_percentage_segmen = vendor_segmen(transaksi_vendor)
user_percentage_segmen = user_segmen(transactions)

jajan_items_categories_merged = merge_jajan_items_categories(jajan_items, categories)
transactions_frequencies = get_frequencies(transactions)
time_now = get_time_now()

weather = get_weather(time_now)
weather_icon = get_weather_icon(weather)
weather_forcast = get_weather_forcast()
weather_forcast_icon = get_weather_icon(weather_forcast)
similarity_vector = create_similarity_vector(jajan_items_categories_merged)
vendor_for_forcast_increase = vendor_from_forcast(similarity_vector, jajan_items, weather_forcast["cuaca"].iloc[0].lower(), True)
vendor_for_forcast_decrease= vendor_from_forcast(similarity_vector, jajan_items, weather_forcast["cuaca"].iloc[0].lower(), False)

recommended = get_recommended(weather["cuaca"].iloc[0].lower(),similarity_vector, transactions_frequencies)
recommendation = get_recommendation(recommended, jajan_items)
bobot_cuaca_kategori=bobot_kategori_cuaca(similarity_vector)

header=st.container()
with header:
    st.title('DA-1-B Project Dashboard')
    st.text("""This dashboard shows the Jajan Mania's Data and Plot that hopefully helps
management solve their problems""")
    
Dataset=st.container()
with Dataset:
    st.header("Thoose are the datasets that we're using")
    st.markdown("* **transaction_histories**    : consist of 11 columns (_transaction_id, user_id, jajan_item_id, amount, payment_type, last_latitude, last_longitude, update_at, created_at, deleted_at, and weather_) which tell the transaction data each transaction_id")
    st.markdown("* **jajan_items**              : consist of 9 columns (_jajan_item_id, vendor_id, category_id, jajan_item_name, image, update_at, created_at, and deleted_at_) which tell the information of each jajan")
    st.markdown("* **Categories**               : consist of 6 columns (_category_id, category_name, icon, update_at, created_at, and deleted_at_) which tell about the information of jajan category")
  
Time_series=st.container()
with Time_series:
    st.header("Let's See Our Sales First")
    col_disp, col_explain = st.columns(2)                                                                              #5 nov 2023
    with col_disp: 
        Pembelian_harian=Transaction_all_categories(transactions)                                                       #15 nov 2023                                                   #1 nov 2023
        fig, ax = plt.subplots(figsize=(6, 6))                                                              #15 nov 2023
        sns.lineplot(data=Pembelian_harian, x="created_at", y="total")                                      #15 nov 2023
        ax.set_title("Penjualan Total Harian", loc="center", fontsize=17)                                   #15 nov 2023
        ax.tick_params(axis='y', labelsize=15)                                                              #15 nov 2023
        ax.tick_params(axis='x', labelsize=16)                                                              #15 nov 2023
        ax.set_ylabel(None)                                                                                 #15 nov 2023
        ax.set_xlabel(None)                                                                                 #15 nov 2023
        st.pyplot(fig)  
    with col_explain:
        st.markdown("                   ")
        st.markdown("                   ")
        st.markdown("""**Explanation** :
                    this chart tells us about total sales in **every categories** every day,
                    we can see the chart to track our sales, is it increasing, decreasing, or fluctuating""")
    col_disp, col_explain = st.columns(2)
    with col_disp:
        Pembelian_per_kategori=Transaksi_kategori= Transaction_per_categories(transactions)
        fig, ax = plt.subplots(figsize=(8, 8))                                                             #15 nov 2023
        sns.lineplot(data=Pembelian_per_kategori, x="created_at", y="C001")
        sns.lineplot(data=Pembelian_per_kategori, x="created_at", y="C002")
        sns.lineplot(data=Pembelian_per_kategori, x="created_at", y="C003")
        sns.lineplot(data=Pembelian_per_kategori, x="created_at", y="C004")
        sns.lineplot(data=Pembelian_per_kategori, x="created_at", y="C005")
        sns.lineplot(data=Pembelian_per_kategori, x="created_at", y="C006")                                 #15 nov 2023
        ax.set_title("Penjualan per Kategori", loc="center", fontsize=17)                                   #15 nov 2023
        ax.tick_params(axis='y', labelsize=15)                                                              #15 nov 2023
        ax.tick_params(axis='x', labelsize=16)                                                              #15 nov 2023
        ax.set_ylabel(None)                                                                                 #15 nov 2023
        ax.set_xlabel(None)                                                                                 #15 nov 2023
        st.pyplot(fig)
    with col_explain:
        st.markdown("                   ")
        st.markdown("                   ")
        st.markdown("""**Explanation** :
                    this chart tells us about total sales in **each categories** every day,
                    we can see the chart to track our sales, is it increasing, decreasing, or fluctuating""")

vendor_performance = st.container()
with vendor_performance:
    st.header("How's our lovely vendor doin ?")
col_disp, col_explain = st.columns(2)                                                                              #15 nov 2023
with col_disp: 
    vendor_tinggi=high_transaction_vendor                                                                   #15 nov 2023                                                   #1 nov 2023
    fig, ax = plt.subplots(figsize=(6, 6))                                                              #15 nov 2023
    sns.barplot(data=vendor_tinggi, x="vendor_id", y="total_transaksi", color ='green')                           #15 nov 2023
    ax.set_title("Vendor Paling Laku", loc="center", fontsize=17)                                       #15 nov 2023
    ax.tick_params(axis='y', labelsize=15)                                                              #15 nov 2023
    ax.tick_params(axis='x', labelsize=16)                                                              #15 nov 2023
    ax.set_ylabel(None)                                                                                 #15 nov 2023
    ax.set_xlabel(None)                                                                                 #15 nov 2023
    st.pyplot(fig) 
with col_explain: 
        st.markdown("                   ")
        st.markdown("                   ")
        st.markdown("""**YAAYYY !!!** thoose are TOP 5 Vendors that have the most
                    sales, clap for their hardwork and don't forget to give'em compliment :D""")    
col_explain, col_disp = st.columns(2) 
with col_explain:
        st.markdown("                   ")
        st.markdown("                   ")
        st.markdown("""**OH NO :(** thoose are TOP 5 Vendors that have the least
                    sales, cheer them up and help them with effective marketing strategy !!!""")
with col_disp : 
    vendor_rendah=low_transaction_vendor                                                                #15 nov 2023                                                   #1 nov 2023
    fig, ax = plt.subplots(figsize=(6, 6))                                                              #15 nov 2023
    sns.barplot(data=vendor_rendah, x="vendor_id", y="total_transaksi", color ='red')                             #15 nov 2023
    ax.set_title("Vendor Laku Paling Sedikit", loc="center", fontsize=17)                               #15 nov 2023
    ax.tick_params(axis='y', labelsize=15)                                                              #15 nov 2023
    ax.tick_params(axis='x', labelsize=16)                                                              #15 nov 2023
    ax.set_ylabel(None)                                                                                 #15 nov 2023
    ax.set_xlabel(None)                                                                                 #15 nov 2023
    st.pyplot(fig) 

col_disp, col_explain = st.columns(2)                                                                              #15 nov 2023
with col_disp: 
    vendor_segmen=vendor_percentage_segmen                                                                       #15 nov 2023
    fig, ax = plt.subplots(figsize=(6, 6))                                                              #15 nov 2023
    plt.pie(x='Count', data=vendor_segmen, labels='segmen', autopct='%1.1f%%', startangle=90)           #15 nov 2023
    ax.set_title("Persentase Segmen Vendor ", loc="center", fontsize=17)                                #15 nov 2023                                                                                #7 nov 2023
    st.pyplot(fig) 
with col_explain: 
        st.markdown("                   ")
        st.markdown("                   ")
        st.markdown("""That is our vendor **segmentation chart** based on their sales.
                    With that, you can see the percentage of each segment and give you the insight of your vendor performance """) 
user_info = st.container()
with user_info:
    st.header("Our King, Customer !!!")
    col_disp, col_explain = st.columns(2)
    with col_disp: 
        user_segmen=user_percentage_segmen                                                                        #15 nov 2023                                                   #1 nov 2023
        fig, ax = plt.subplots(figsize=(6, 6))                                                              #15 nov 2023
        plt.pie(x='Count', data=user_segmen, labels='segmen', autopct='%1.1f%%', startangle=90)             #15 nov 2023
        ax.set_title("Persentase Segmen Customer ", loc="center", fontsize=17)                              #15 nov 2023                                                                                #7 nov 2023
        st.pyplot(fig) 
    with col_explain: 
        st.markdown("                   ")
        st.markdown("                   ")
        st.markdown("""That is our customer **segmentation chart** based on their purchase.
                    With that, you can see the percentage of each segment and give you the insight of your customer
                    and give you consideration to choose the best _marketing strategy_""")

    st.markdown("""You can also **pick specific user id to see their purchase history** with this dropdown below, Like people said,
            "Understanding customer means increasing profit", right ?!?""")
    
col5, col6 = st.columns(2)                                                                             #7 nov 2023
with col5:                           #1 nov 2023                                             #1 nov 2023
    user_id = st.selectbox('Select a User ID', transactions['user_id'].unique())
    if user_id:                                                                                             #1 nov 2023
        recommendation2 = calculate_category_sum(user_id, transactions)
        st.write(recommendation2)
with col6:
    pembelian_user = calculate_category_sum(user_id, transactions)                                     #7 nov 2023
    fig, ax = plt.subplots(figsize=(7, 6))                                                              #7 nov 2023
    plt.pie(data=pembelian_user,x='amount', labels='category_id',autopct='%.0f%%')                            #7 nov 2023
    ax.set_title("transaksi historis customer", loc="center", fontsize=17)                                #7 nov 2023 
    st.pyplot(fig)                                                                         #1 nov 2023

recommendation2 = calculate_category_sum(user_id, transactions) 
                                  
header_cuaca=st.container()
with header_cuaca:
    st.header("AH it's raining, what category customer more likely to purchase ?")
    st.markdown("""From the transaction history data, we can see how our customer behaviour in each weather
                condition, what they purchased when it rain, or suny, or cloudy, so we can give
                **weight in each categories** in every weather condition """)
    cuaca = st.selectbox('Select a Weather', bobot_cuaca_kategori.columns[1:10])                           #16 nov 2023         
    if cuaca:                                                                                             #16 nov 2023
        tabel_per_cuaca = bobot(cuaca,bobot_cuaca_kategori)                                                            #16 nov 2023
        st.write(tabel_per_cuaca)                                                                           #16 nov 2023

tabel_per_cuaca = bobot(cuaca,bobot_cuaca_kategori)                                                             #16 nov 2023
                                                                         
forecast = st.container()
forcast_col1, forcast_col2, forcast_col3 = st.columns(3)
with forcast_col2:
    st.image(weather_forcast_icon)
    st.write(weather_forcast["cuaca"].iloc[0])
with forecast:
    st.header("AH tomorrow's {weather_forcast}, let ours vendors know !!!".format(weather_forcast=weather_forcast["cuaca"].iloc[0]))
    st.markdown("""we knew the **best and the worst category in each weather**, and we can
              see the **tomorrow forecast**, so we can inform our vendor to **prepare their stock**.
              vendor who sell the least profitable category based on the forecast need to
              reduce their stock, and so on""")

col_tinggi, col_rendah = st.columns(2)
with col_tinggi :
    st.markdown("here are the vendors that will have their **sales increase** based on the forecast above!")
    st.table(data=vendor_for_forcast_increase)
with col_rendah:
    st.markdown("here are the vendors that will have their **sales decrease** based on the forecast above!")
    st.table(data=vendor_for_forcast_decrease)
with st.sidebar:
    st.image(weather_icon)
    st.write(weather["cuaca"].iloc[0])



