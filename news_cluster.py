from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import random
import re, string
from konlpy.tag import Komoran, Hannanum
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from keybert import KeyBERT

### 데이터 로드
data = pd.read_excel('/app/data/news_1000_results.xlsx')
data_raw = data.copy()
#print(data.head())

### 1. 전처리
komoran = Komoran()
hannanum = Hannanum()

# 불용어 파일 열기
with open('korean_stopwords.txt', 'r', encoding='utf-8') as f:
    list_file = f.readlines() 
stopwords = list_file[0].split(",")

# 정규화, 특수기호 제거
def preprocess(text):
    text=text.strip()  
    text=re.compile('<.*?>').sub(' ', text) 
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
    text = re.sub('\s+', ' ', text)  
    text = re.sub(r'\[[0-9]*\]',' ',text)
    text=re.sub(r'[^\w\s]', ' ', str(text).strip())
    text = re.sub(r'\d',' ',text) 
    text = re.sub(r'\s+',' ',text) 
    return text


# 명사/영단어 추출, 한글자 제외, 불용어 제거
def final(text):
    n = []
    word = komoran.nouns(text)
    p = komoran.pos(text)
    for pos in p:
      if pos[1] in ['SL']:
        word.append(pos[0])
    for w in word:
      if len(w)>1 and w not in stopwords:
        n.append(w)
    return " ".join(n)


def finalpreprocess(text):
  return final(preprocess(text))

data['noun'] = data['title'].apply(lambda x: finalpreprocess(x))
# data.to_excel(excel_writer='news_with_noun.xlsx')
# print(data.head())


### 2. 클러스터링
text = data['noun']

#1 tf-idf 임베딩(+Normalize)
def tfidf_vectorizer(text, min_df=3, ngram_range=(1,5)):
    tfidf_vectorizer = TfidfVectorizer(min_df = 3, ngram_range=(1,5)) #min_df :최소 빈도값 (단어 등장하는 문서 수)
    tfidf_vectorizer.fit(text)
    vector = tfidf_vectorizer.transform(text).toarray()
    vector = np.array(vector) # Normalizer를 이용해 이미 변환된 벡터
    return vector

vector_1st = tfidf_vectorizer(text, min_df=1)

#2 DBSCAN Clustering

model = DBSCAN(eps=0.1,min_samples=1, metric = "cosine") 
#     거리 계산 식으로는 Cosine distance를 이용
#     eps이 낮을수록, min_samples 값이 높을수록 군집으로 판단하는 기준이 까다로움.
result_1st = model.fit_predict(vector_1st)
data['cluster1st'] = result_1st
# print(data['cluster1st'])
# print('군집개수 :', result_1st.max())
# data



##########################################################################################################

def print_cluster_result(train, result, col_cluster):
    # train : 데이터, result : 군집화 결과, col_cluster : 클러스터넘버 컬럼명
    clusters = []
    counts = []
    titles = []
    urls = []
    thumbnails = []
    nouns = []
    for cluster_num in set(result):
            print("cluster num : {}".format(cluster_num))
            temp_df = train[train[col_cluster] == cluster_num] # cluster num 별로 조회
            clusters.append(cluster_num)
            counts.append(len(temp_df))
            titles.append(temp_df.reset_index()['title'][0])
            urls.append(temp_df.reset_index()['url'][0])
            thumbnails.append(temp_df.reset_index()['thumbnail'][0])
            nouns.append(temp_df.reset_index()['noun'][0]) 

            for title in temp_df['title']:
                print(title) # 제목으로 살펴보자
            print()

    cluster_result = pd.DataFrame({'cluster_num':clusters, 'count':counts, 'title': titles, 'url':urls, 'thumbnail':thumbnails, 'noun': nouns})
    return cluster_result

cluster1_result = print_cluster_result( train=data, 
                                       result=result_1st, col_cluster="cluster1st")
print(cluster1_result)
# exit()

text2 = cluster1_result['noun']
vector_2nd = tfidf_vectorizer(text2, min_df=1)
     

# Silhouette Score - 최적 k
def visualize_silhouette_layer(data, param_init='random', param_n_init=10, param_max_iter=300):
    clusters_range = range(2,30)
    results = []

    for i in clusters_range:
        clusterer = KMeans(n_clusters=i, init=param_init, n_init=param_n_init, max_iter=param_max_iter, random_state=0)
        cluster_labels = clusterer.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        results.append([i, silhouette_avg])

    result = pd.DataFrame(results, columns=["n_clusters", "silhouette_score"])
    pivot_km = pd.pivot_table(result, index="n_clusters", values="silhouette_score")

    # plt.figure()
    # sns.heatmap(pivot_km, annot=True, linewidths=.5, fmt='.3f', cmap=sns.cm._rocket_lut)
    # plt.tight_layout()
    # plt.show()

visualize_silhouette_layer(vector_2nd) # 가장 높은 실루엣 계수와 매핑되는 k

# kmeans 군집 결과 확인
from sklearn.cluster import KMeans

result_2nd = KMeans(n_clusters=29).fit_predict(vector_2nd)
cluster1_result['cluster2nd'] = result_2nd

cluster2_result = print_cluster_result( train=cluster1_result, 
                                       result=result_2nd, col_cluster="cluster2nd")




###### 키워드 추출

key_model = KeyBERT('paraphrase-multilingual-MiniLM-L12-v2')  #distilbert-base-nli-mean-tokens / paraphrase-multilingual-MiniLM-L12-v2

def keyword(data, col_cluster):  #data = cluster_result (데이터프레임) #1분 30초 소요됨
    result = []
    for i in range(len(data)):
        key_text = cluster1_result[cluster1_result[col_cluster]==i]['noun']
        key_text = ' '.join(key_text)
        keyword = key_model.extract_keywords(key_text, keyphrase_ngram_range=(1,2), top_n=1)
        result.append(keyword[0][0])
    return result

def merge_keyword(data, col_cluster): #새 열로 추가.
    data_temp = data.copy()
    data_temp['keyword'] = keyword(data, col_cluster)
    return data_temp

keyword_result = merge_keyword(cluster2_result, col_cluster='cluster2nd')

keyword_df = keyword_result[['cluster_num', 'count', 'keyword']]
keyword_df.sort_values(by='count', ascending=False, inplace=True, ignore_index=True)
keyword_df.drop(index=[0], inplace=True)
keyword_df = keyword_df[keyword_df['count']>5]
lst = []
for i in keyword_df['keyword']:
  lst.append(i.upper())
keyword_df['keyword'] = lst
keyword_df

# keyword_df.to_excel(/xcel_writer='news_keyword.xlsx')


################################################################################################################


# 1. 클러스터별 대표 제목 dict 만들기
cluster_to_representative_title = {
    cluster_num: df.iloc[0]['title']
    for cluster_num, df in data.groupby('cluster1st') if cluster_num != -1
}

# 2. 대표 제목 맵핑 (노이즈는 그대로 제목 사용)
def assign_cluster_title(row):
    if row['cluster1st'] == -1:
        return row['title'] 
    else:
        return cluster_to_representative_title.get(row['cluster1st'], row['title'])  # 혹시 없을 경우 대비

# 3. clusters 열 생성
data['clusters'] = data.apply(assign_cluster_title, axis=1)

# 4. 저장
# data.to_excel("/app/data/news_preproc.xlsx", index=False)
# print("매핑 완료")


# news 단위로 확장하려면 cluster1_result에서 원래 뉴스들과 매핑되어야 함
# cluster1_result: 클러스터별 대표 뉴스
# data: 전체 뉴스

# cluster1_result의 cluster1st와 대표 제목 clusters 기반으로 매핑
title_to_cluster2nd = cluster1_result.set_index('title')['cluster2nd'].to_dict()
data['cluster2nd'] = data['clusters'].map(title_to_cluster2nd)

# cluster2nd별 keyword 딕셔너리 생성
cluster2nd_to_keyword = keyword_result.set_index('cluster_num')['keyword'].to_dict()
# keyword 열 생성
data['keyword'] = data['cluster2nd'].map(cluster2nd_to_keyword)

# 1. cluster2nd별 뉴스 개수(count)를 dict로 저장
cluster2nd_to_count = keyword_result.set_index('cluster_num')['count'].to_dict()
# 2. 각 뉴스 기사에 counts 열 추가
data['counts'] = data['cluster2nd'].map(cluster2nd_to_count)
# 노이즈에는 counts = 1
data['counts'] = data['counts'].fillna(1)


# 기존 인덱스를 'index'라는 이름의 열로 추가하고, 새로운 정수 인덱스를 부여
data = data.reset_index()

# 열 순서 재정렬: 'index'를 가장 앞으로
cols = ['index'] + [col for col in data.columns if col != 'index']
data = data[cols]


data.to_excel("/app/data/news_preproc.xlsx", index=False)
print("뉴스 기사에 keyword 열 추가 완료!")



#   id integer [primary key]                   ==> index    (0부터 시작)
#   title varchar // 뉴스 제목                  ==> title
#   link varchar // 뉴스 원본 링크               ==> url
#   upload_date timestamp // 뉴스 업데이트 기사   ==> upload_date
#   photo_link varchar // 뉴스 사진 링크         ==> thumbnail
#   press_id integer // 언론사                  ==> company
#   category_id integer // 카테고리 ID           ==> subject
#   topic_id integer // 요약 ID                 ==> cluster2nd



########################## DB #################################

# import mysql.connector

# # MySQL 연결
# if os.environ.get("KUBERNETES_SERVICE_HOST") is None:
#     # .env 파일의 경로를 명시적으로 지정합니다.
#     # 프로젝트 구조에 따라 'backend/.env' 대신 '.env'만 필요할 수도 있습니다.
#     # 예: load_dotenv(dotenv_path=".env")
#     load_dotenv(dotenv_path=".env")

# # MySQL 연결 설정
# # 환경 변수가 설정되어 있지 않을 경우를 대비하여 .get() 메서드를 사용하고 기본값을 제공하는 것이 좋습니다.
# try:
#     conn = mysql.connector.connect(
#         host=os.environ.get("DB_HOST", "localhost"), # 환경 변수 없으면 'localhost' 기본값
#         user=os.environ.get("DB_USER", "root"),       # 환경 변수 없으면 'root' 기본값
#         password=os.environ.get("DB_PASSWORD", ""),   # 환경 변수 없으면 빈 문자열 기본값
#         database=os.environ.get("DB_NAME", "test_db"), # 환경 변수 없으면 'test_db' 기본값
#     )
#     cursor = conn.cursor()
#     print("✅ 데이터베이스 연결 성공!")
# except mysql.connector.Error as err:
#     print(f"❌ 데이터베이스 연결 오류: {err}")
#     # 오류 발생 시 프로그램 종료 또는 적절한 예외 처리
#     exit()


# # 1. 우선 필요한 열이 없다면 테이블에 열 추가
# alter_sqls = [
#     "ALTER TABLE news_raw ADD cluster2nd INT",
#     "ALTER TABLE news_raw ADD keyword VARCHAR(255)",
#     "ALTER TABLE news_raw ADD counts INT"
# ]

# for sql in alter_sqls:
#     try:
#         cursor.execute(sql)
#     except mysql.connector.errors.ProgrammingError as e:
#         print(f"[무시됨] ALTER 오류 (이미 존재할 가능성): {e}")

# conn.commit()

# if 'data' not in locals():
#     print("경고: 'data' DataFrame이 정의되지 않았습니다. DB에 데이터를 넣을수 없습니다.")

# update_sql = """
# UPDATE news_raw
# SET cluster2nd = %s,
#     keyword = %s,
#     counts = %s
# WHERE url = %s
# """

# for _, row in data.iterrows():
#     try:
#         cursor.execute(update_sql, (
#             int(row['cluster2nd']) if not pd.isnull(row['cluster2nd']) else None,
#             row['keyword'],
#             int(row['counts']) if not pd.isnull(row['counts']) else 1,
#             row['url']  # 기준 URL
#         ))
#     except Exception as e:
#         print(f"❌ UPDATE 실패 (URL: {row['url']}) → {e}")

# conn.commit()
# cursor.close()
# conn.close()
# print("✅ 클러스터링 정보(키워드 포함)를 기존 테이블 news_raw에 반영 완료!")
