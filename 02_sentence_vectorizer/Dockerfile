FROM eclipse-temurin:17-jdk-jammy

# 우분투/데비안 기반 환경에서 파이썬3와 pip를 안정적으로 설치
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# python 명령어가 python3로 동작하도록 링크 설정
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# 파이썬 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# FastAPI 서버 실행
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
