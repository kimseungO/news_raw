import os
from dotenv import load_dotenv
import requests
import urllib.parse
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import mysql.connector

#   id integer [primary key]                   ==> index    (0부터 시작)
#   title varchar // 뉴스 제목                  ==> title
#   link varchar // 뉴스 원본 링크               ==> url
#   upload_date timestamp // 뉴스 업데이트 기사   ==> upload_date
#   photo_link varchar // 뉴스 사진 링크         ==> thumbnail
#   press_id integer // 언론사                  ==> company
#   category_id integer // 카테고리 ID           ==> subject
#   topic_id integer // 요약 ID                 ==> cluster2nd

# load .env
load_dotenv()

# 🔧 네이버 API 인증 정보
client_id = os.environ.get('NAVER_CLIENT_ID')
client_secret = os.environ.get('NAVER_KEY')

# 🔍 검색어 설정
query = 'AI 뉴스'
enc_query = urllib.parse.quote(query)

# 📦 결과 저장 리스트
news_data = []

# 📰 뉴스 본문 크롤링 함수
def get_news_body(link):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(link, headers=headers, timeout=5)
        if res.status_code != 200:
            return f"[{res.status_code}] 접근 실패"
        soup = BeautifulSoup(res.text, 'html.parser')

        article = soup.select_one('#dic_area') or soup.select_one('.newsct_article')
        if article:
            return article.get_text(strip=True)

        paragraphs = soup.find_all('p')
        if paragraphs:
            return ' '.join(p.get_text(strip=True) for p in paragraphs[:5])
        return "[본문 없음]"
    except Exception as e:
        return f"[에러] {e}"

# 🖼 썸네일 이미지 주소 추출 함수
def get_news_thumbnail(link):
    headers = {'User-Agent': 'Mozilla/5.0'}
    img_tag = []
    try:
        res = requests.get(link, headers=headers, timeout=5)
        if res.status_code != 200:
            return ""
        soup = BeautifulSoup(res.text, 'html.parser')

        # 1순위: id="img1"
        img_tag = soup.find('img', id='img1')
        if img_tag and img_tag.get('src'):
            return img_tag['data-src']

        # 2순위: class로 '대표 이미지' 추정
        # img_tag = soup.find('img', class_='_LAZY_LOADING _LAZY_LOADING_INIT_HIDE')  # 네이버 뉴스 대표 이미지 class 자주 사용됨
        # if img_tag and img_tag.get('src'):
        #     return img_tag['data-src']

        # 3순위: 가장 첫 번째 <img> 태그 (단, 광고 필터링 필요할 수 있음)
        # img_tag = soup.find('img')
        # if img_tag and img_tag.get('src'):
        #     return img_tag['src']

        print(img_tag['data-src'])
        return img_tag['data-src']
    except Exception as e:
        print(e)
        return ""

# 언론사 추출 함수
def get_news_company(link):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(link, headers=headers, timeout=5)
        if res.status_code != 200:
            return ""
        soup = BeautifulSoup(res.text, 'html.parser')

        # 언론사 로고 이미지에서 title 속성 추출
        logo_img = soup.find('img', class_='media_end_head_top_logo_img')
        if logo_img and logo_img.get('title'):
            return logo_img['title'].strip()

        return ""
    except Exception:
        return ""

def get_news_subject(link):
    try:
        match = re.search(r"sid=(\d+)", link)
        if match:
            return match.group(1)
        else:
            return ""
    except:
        return ""


# 🔁 API 호출 및 데이터 수집
for i in range(100):
    start_index = i * 10 + 1
    url = f"https://openapi.naver.com/v1/search/news.json?query={enc_query}&display=10&start={start_index}&sort=date"

    print(f"🚀 {i+1}번째 요청 (시작 인덱스: {start_index})")
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        news_items = response.json()['items']
        for item in news_items:
            title = item['title'].replace('<b>', '').replace('</b>', '')
            link = item['link']
            upload_date = item['pubDate']

            if 'n.news.naver.com' not in link:
                # print(f"   [SKIP] 네이버 뉴스 링크가 아닙니다: {link}")
                continue

            content = get_news_body(link)
            thumbnail = get_news_thumbnail(link)
            company = get_news_company(link)
            subject = get_news_subject(link)
            #upload_date = get_upload_date(link)


            news_data.append({
                'title': title,
                'url': link,
                'contents': content,
                'thumbnail': thumbnail,
                'company': company,
                'subject': subject,
                'upload_date': upload_date
            })
    else:
        print(f"❌ 에러 발생: {response.status_code} - {response.text}")
        break

    time.sleep(0.5)

# 📄 DataFrame으로 저장하고 엑셀로 출력
print("\n데이터 수집 완료! 엑셀 파일로 저장합니다...")
df = pd.DataFrame(news_data)
df.to_excel("/app/data/news_1000_results.xlsx", index=False)
print(f"✅ 총 {len(df)}개의 뉴스를 '/app/data/news_1000_results.xlsx'로 저장 완료!")


########## DB ############

# KUBERNETES_SERVICE_HOST 환경 변수가 정의되어 있지 않으면
# 'backend/.env' 파일에서 환경 변수를 로드합니다.
# Docker 환경에서는 이 변수가 정의될 가능성이 높으므로, 로컬 개발 환경에서 유용합니다.
if os.environ.get("KUBERNETES_SERVICE_HOST") is None:
    # .env 파일의 경로를 명시적으로 지정합니다.
    # 프로젝트 구조에 따라 'backend/.env' 대신 '.env'만 필요할 수도 있습니다.
    # 예: load_dotenv(dotenv_path=".env")
    load_dotenv(dotenv_path=".env")

# MySQL 연결 설정
# 환경 변수가 설정되어 있지 않을 경우를 대비하여 .get() 메서드를 사용하고 기본값을 제공하는 것이 좋습니다.
try:
    conn = mysql.connector.connect(
        host=os.environ.get("DB_HOST", "localhost"), # 환경 변수 없으면 'localhost' 기본값
        user=os.environ.get("DB_USER", "root"),       # 환경 변수 없으면 'root' 기본값
        password=os.environ.get("DB_PASSWORD", ""),   # 환경 변수 없으면 빈 문자열 기본값
        database=os.environ.get("DB_NAME", "test_db"), # 환경 변수 없으면 'test_db' 기본값
    )
    cursor = conn.cursor()
    print("✅ 데이터베이스 연결 성공!")
except mysql.connector.Error as err:
    print(f"❌ 데이터베이스 연결 오류: {err}")
    # 오류 발생 시 프로그램 종료 또는 적절한 예외 처리
    exit()

# news_raw 테이블 생성 (없을 경우)
create_table_sql = """
CREATE TABLE IF NOT EXISTS news_raw (
    id INT AUTO_INCREMENT PRIMARY KEY,
    title TEXT,
    url VARCHAR(767) UNIQUE, -- UNIQUE 인덱스 추가 (TEXT 타입에 직접 인덱스 불가, VARCHAR 길이 제한)
    contents LONGTEXT,       -- LONGTEXT로 변경
    thumbnail TEXT,
    company VARCHAR(100),
    subject VARCHAR(10),
    upload_date DATETIME
);
"""
try:
    cursor.execute(create_table_sql)
    conn.commit()
    print("✅ 'news_raw' 테이블 확인/생성 완료!")
except mysql.connector.Error as err:
    print(f"❌ 테이블 생성 오류: {err}")
    exit()

# INSERT 쿼리
insert_sql = """
INSERT INTO news_raw (title, url, contents, thumbnail, company, subject, upload_date)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
    title = VALUES(title),
    contents = VALUES(contents),
    thumbnail = VALUES(thumbnail),
    company = VALUES(company),
    subject = VALUES(subject),
    upload_date = VALUES(upload_date)
"""

for news in news_data:
    try:
        cursor.execute(insert_sql, (
            news['title'],
            news['url'],
            news['contents'],
            news['thumbnail'],
            news['company'],
            news['subject'],
            pd.to_datetime(news['upload_date'])  # pubDate 문자열 → datetime 변환
        ))
    except Exception as e:
        print("❌ INSERT 실패:", e)
        continue

conn.commit()
print("✅ DB에 뉴스 데이터 저장 완료!")

######### 정리 ############
if 'cursor' in locals() and cursor:
    cursor.close()
if 'conn' in locals() and conn:
    conn.close()
print("✅ 데이터베이스 연결 종료.")