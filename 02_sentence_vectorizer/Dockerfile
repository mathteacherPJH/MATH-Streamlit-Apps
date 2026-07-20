FROM python:3.11-slim

# 시스템 자바(OpenJDK) 및 필수 빌드 도구 설치
RUN apt-get update && apt-get install -y \
    openjdk-17-jdk \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 환경 변수 설정
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# 작업 디렉토리 설정
WORKDIR /app

# 라이브러리 목록 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# FastAPI 서버 실행 (포트 10000)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
