import os
import jpype
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

os.environ['JAVA_HOME'] = '/usr/lib/jvm/default-java'
if not jpype.isJVMStarted():
    try:
        jvm_path = jpype.getDefaultJVMPath()
        jpype.startJVM(jvm_path)
    except Exception as e:
        print(f"JVM Startup Error: {e}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

okt = Okt()

def lemmatize_core(text: str) -> str:
    if not text.strip():
        return ""
    raw_tokens = okt.morphs(text, stem=True)
    return " ".join(raw_tokens)

class NLPRequest(BaseModel):
    sentences: List[str]
    custom_vocab: List[str] = []

@app.post("/api/analyze")
def analyze_sentences(data: NLPRequest):
    sentences = [s.strip() for s in data.sentences if s.strip()]
    if not sentences:
        return {"vocabulary": [], "one_hot": [], "frequency": [], "processed_corpus": []}

    processed_corpus = [lemmatize_core(s) for s in sentences]

    vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
    try:
        frequency_matrix = vectorizer.fit_transform(processed_corpus).toarray()
        vocabulary_features = list(vectorizer.get_feature_names_out())
        
        for custom_w in data.custom_vocab:
            if custom_w not in vocabulary_features:
                vocabulary_features.append(custom_w)
                
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
