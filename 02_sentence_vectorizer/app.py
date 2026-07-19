import streamlit as st
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import os

os.environ['JAVA_HOME'] = '/usr/lib/jvm/default-java'

st.set_page_config(page_title="문장 벡터화 프로그램", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background-color: #FBF9F4 !important;
        color: #2C2A29 !important;
        font-family: 'Inter', 'Malgun Gothic', -apple-system, sans-serif;
    }
    [data-testid="stHeader"] {
        display: none !important;
    }
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 1rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
    }
    
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px 20px;
        border-bottom: 1px solid rgba(44, 42, 41, 0.08);
        margin-left: -1.5rem;
        margin-right: -1.5rem;
        margin-bottom: 1.5rem;
        background-color: #FBF9F4;
    }
    .back-link {
        text-decoration: none;
        color: #1B4332;
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 0.15em;
    }
    .nav-title {
        color: #1B4332;
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 0.15em;
    }

    div[data-testid="stColumn"] {
        background-color: #ffffff !important;
        border: 1px solid rgba(44, 42, 41, 0.08) !important;
        border-radius: 12px !important;
        padding: 25px !important;
        box-shadow: 0 4px 20px rgba(44, 42, 41, 0.01) !important;
    }
    
    .stTextArea textarea {
        background-color: transparent !important;
        border: none !important;
        color: #2C2A29 !important;
        font-size: 0.95rem !important;
        height: 480px !important;
    }
    .stTextArea div[data-baseweb="textarea"] {
        border: none !important;
        background-color: transparent !important;
    }
    
    div.stButton > button {
        background-color: #1B4332 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 14px 0px !important;
        font-size: 1rem !important;
        font-weight: bold !important;
        letter-spacing: 0.05em !important;
        box-shadow: none !important;
        transition: background-color 0.2s ease;
        width: 100% !important;
    }
    div.stButton > button:hover {
        background-color: #2d5a45 !important;
        color: #ffffff !important;
    }
    
    div[data-testid="stBox"] {
        background-color: #F7F5F0 !important;
        border: 1px solid rgba(44, 42, 41, 0.06) !important;
        border-radius: 6px !important;
        padding: 15px !important;
        margin-bottom: 15px;
    }
    
    div[data-testid="stBox"] p {
        margin: 0 !important;
        font-size: 0.95rem !important;
        color: #333333 !important;
    }

    div[data-testid="stTabList"] {
        background-color: #F2EFE9 !important;
        padding: 5px !important;
        border-radius: 8px !important;
        gap: 4px !important;
        border: none !important;
    }
    button[data-testid="stTab"] {
        background-color: transparent !important;
        color: #666666 !important;
        border: none !important;
        padding: 8px 16px !important;
        border-radius: 6px !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }
    button[data-testid="stTab"][aria-selected="true"] {
        background-color: #ffffff !important;
        color: #1B4332 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
    }
    div[data-testid="stTabTabpanel"] {
        padding-top: 20px !important;
    }
    
    h3 {
        font-size: 0.95rem !important;
        font-weight: bold !important;
        color: #2C2A29 !important;
        margin-bottom: 10px !important;
        margin-top: 15px !important;
    }
    
    .stDataFrame div {
        border: none !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="top-nav">
        <a href="https://mathteacherpjh.github.io/MATH-Program/" target="_parent" class="back-link">← BACK</a>
        <div class="nav-title">VECTORIZATION ANALYSIS</div>
    </div>
""", unsafe_allow_html=True)

@st.cache_resource
def get_okt_tokenizer():
    return Okt()

okt = get_okt_tokenizer()

def lemmatize_core(text):
    if not text.strip():
        return ""
    raw_tokens = okt.morphs(text, stem=True)
    return " ".join(raw_tokens)

left_panel, right_panel = st.columns([1, 1.6], gap="large")

with left_panel:
    user_input = st.text_area(
        label="문장 입력창",
        placeholder="여기에 분석할 문장들을 입력하세요. 엔터(줄바꿈)를 기준으로 각각 다른 문장으로 인식하여 행렬을 구성합니다.",
        label_visibility="collapsed"
    )
    
    execute_analysis = st.button("분석 실행 ⚡")

with right_panel:
    if execute_analysis and user_input.strip():
        sentences_inputs = [s.strip() for s in user_input.split("\n") if s.strip()]
        
        if sentences_inputs:
            processed_corpus = [lemmatize_core(s) for s in sentences_inputs]
            vectorizer_engine = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
            
            try:
                frequency_matrix = vectorizer_engine.fit_transform(processed_corpus).toarray()
                vocabulary_features = vectorizer_engine.get_feature_names_out()
                one_hot_matrix = np.where(frequency_matrix > 0, 1, 0)
                
                tab_summary, tab_vocab, tab_onehot, tab_freq = st.tabs([
                    "전체보기", "1. 단어집합", "2. 원-핫 벡터", "3. 빈도수 벡터"
                ])
                
                with tab_summary:
                    st.write("### 📋 종합 분석 데이터 요약")
                    mapping_labels = " | ".join([f"{index}: **{word}**" for index, word in enumerate(vocabulary_features)])
                    st.markdown(f"<div data-testid='stBox'><p>{mapping_labels}</p></div>", unsafe_allow_html=True)
                    
                    matrix_dataframe = pd.DataFrame(
                        frequency_matrix, 
                        columns=vocabulary_features, 
                        index=[f"문장 {k+1}" for k in range(len(sentences_inputs))]
                    )
                    st.write("### 📊 단어별 등장 빈도 매트릭스 (DTM)")
                    st.dataframe(matrix_dataframe, use_container_width=True)
                    
                with tab_vocab:
                    st.write("### 📌 구축된 단어집합 (Vocabulary)")
                    badges_html = " ".join([f"`{token}`" for token in vocabulary_features])
                    st.markdown(badges_html)
                    
                with tab_onehot:
                    st.write("### 🔢 원-핫 벡터 (One-Hot Vector)")
                    for idx, (original_text, clean_text) in enumerate(zip(sentences_inputs, processed_corpus)):
                        st.markdown(f"**문장 {idx+1}:** \"{original_text}\"")
                        tokens_line = ", ".join(clean_text.split())
                        st.markdown(f"<div data-testid='stBox'><p>추출 토큰: [ {tokens_line} ]</p></div>", unsafe_allow_html=True)
                        st.code(f"{one_hot_matrix[idx].tolist()}", language="json")
                        
                with tab_freq:
                    st.write("### 📊 빈도수 벡터 (Frequency Vector)")
                    for idx, (original_text, clean_text) in enumerate(zip(sentences_inputs, processed_corpus)):
                        st.markdown(f"**문장 {idx+1}:** \"{original_text}\"")
                        tokens_line = ", ".join(clean_text.split())
                        st.markdown(f"<div data-testid='stBox'><p>추출 토큰: [ {tokens_line} ]</p></div>", unsafe_allow_html=True)
                        st.code(f"{frequency_matrix[idx].tolist()}", language="json")
                        
            except ValueError:
                st.warning("유효한 형태소 토큰이 문장 안에 존재하지 않습니다.")
    else:
        st.markdown(
            "<div style='color: #666666; font-size: 0.95rem; text-align: center; padding-top: 100px;'>"
            "왼쪽 패널에 문장을 입력한 뒤 <b>'분석 실행 ⚡'</b> 버튼을 누르면 수학적 변환 행렬이 이곳에 실시간으로 출력됩니다."
            "</div>", 
            unsafe_allow_html=True
        )
