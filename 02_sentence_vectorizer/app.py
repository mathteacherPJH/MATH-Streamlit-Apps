from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os

import os

# Render(Ubuntu/Debian 리눅스) 환경에서의 OpenJDK 기본 경로 지정
os.environ['JAVA_HOME'] = '/usr/lib/jvm/default-java'
os.environ['PATH'] += f":{os.environ['JAVA_HOME']}/bin"

app = FastAPI()

# 🌐 CORS 설정: 깃허브 블로그 주소에서 보내는 요청만 허용 (보안 강화)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 나중에 본인의 깃허브 블로그 주소로 제한해도 됩니다.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 형태소 분석기 초기화 (Okt)
okt = Okt()

def lemmatize_core(text: str) -> str:
    if not text.strip():
        return ""
    # 형태소 추출 및 기본형 복원
    raw_tokens = okt.morphs(text, stem=True)
    return " ".join(raw_tokens)

# 프론트엔드에서 받을 데이터 구조 정의
class NLPRequest(BaseModel):
    sentences: List[str]
    custom_vocab: List[str] = []

@app.post("/api/analyze")
def analyze_sentences(data: NLPRequest):
    sentences = [s.strip() for s in data.sentences if s.strip()]
    if not sentences:
        return {"vocabulary": [], "one_hot": [], "frequency": [], "processed_corpus": []}

    # 1. 형태소 분석 및 기본형 변환 수행
    processed_corpus = [lemmatize_core(s) for s in sentences]

    # 2. scikit-learn CountVectorizer 연산
    vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
    try:
        frequency_matrix = vectorizer.fit_transform(processed_corpus).toarray()
        vocabulary_features = list(vectorizer.get_feature_names_out())
        
        # 사용자가 수동으로 추가한 단어가 있다면 단어집합에 합산
        for custom_w in data.custom_vocab:
            if custom_w not in vocabulary_features:
                vocabulary_features.append(custom_w)
                
        # 만약 커스텀 단어가 추가되어 벡터 차원이 달라질 경우를 대비해 재생성
        if data.custom_vocab:
            vectorizer_custom = CountVectorizer(token_pattern=r'(?u)\b\w+\b', vocabulary=vocabulary_features)
            frequency_matrix = vectorizer_custom.fit_transform(processed_corpus).toarray()

        one_hot_matrix = np.where(frequency_matrix > 0, 1, 0)
    except Exception as e:
        return {"error": str(e)}

    return {
        "vocabulary": vocabulary_features,
        "processed_corpus": processed_corpus,
        "one_hot": one_hot_matrix.tolist(),
        "frequency": frequency_matrix.tolist()
    }
