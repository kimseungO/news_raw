# Python 3.8 버전의 경량 Debian 기반 이미지를 사용합니다.
FROM python:3.8-slim-buster

# 컨테이너 내 작업 디렉토리를 설정합니다.
WORKDIR /app

# mysql-connector-python 및 konlpy (JPype1)에 필요한 시스템 종속성을 설치합니다.
# default-jdk는 Java Development Kit를 설치하여 JVM을 제공합니다.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    default-jdk \
    build-essential \
    default-libmysqlclient-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# requirements_py38.txt 파일을 컨테이너에 복사하고 Python 종속성을 설치합니다.
# --no-cache-dir 옵션은 설치 후 pip 캐시를 삭제하여 이미지 크기를 줄입니다.
COPY korean_stopwords.txt .
COPY JPype1-1.2.0-cp38-cp38-manylinux1_x86_64.whl .
COPY JPype1-1.2.0-cp38-cp38-manylinux2010_x86_64.whl .

COPY requirements_py38.txt .
RUN pip install --no-cache-dir -r requirements_py38.txt

# .env 파일과 Python 스크립트들을 컨테이너의 작업 디렉토리로 복사합니다.
# .env 파일은 보안상 Dockerfile에 직접 포함하는 것보다 Docker Compose의 environment 섹션을 통해 전달하는 것이 더 안전할 수 있습니다.
# 하지만 현재 구조에서는 COPY를 통해 컨테이너 내부에 두는 방식도 가능합니다.
COPY .env .
COPY naverapi.py .
COPY news_cluster.py .

# 데이터 파일이 저장될 디렉토리를 생성합니다.
# 이 디렉토리는 docker-compose.yml에서 호스트의 'data' 디렉토리와 마운트될 것입니다.
RUN mkdir -p /app/data

# 컨테이너가 시작될 때 실행될 기본 명령을 정의합니다.
# 이 이미지는 initContainer로 사용될 것이므로, 필요한 스크립트들을 순서대로 실행합니다.
CMD ["bash", "-c", "python3 naverapi.py && python3 news_cluster.py"]