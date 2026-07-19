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
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #FBF9F4 !important;
        color: #2C2A29 !important;
        font-family: 'Malgun Gothic', -apple-system, sans-serif;
    }
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0) !important;
    }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    div[data-testid="stColumn"] {
        background-color: rgba(255, 255, 255, 0.73) !important;
        border: 1px solid rgba(44, 42, 41, 0.1) !important;
        border-radius: 12px !important;
        padding: 20px !important;
        box-shadow: 0 10px 30px rgba(44, 42, 41, 0.03) !important;
    }
    h1 {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.05em !important;
        margin-bottom: 0.2rem !important;
        color: #1B4332 !important;
    }
    .custom-caption {
        font-size: 0.9rem !important;
        color: #666666 !important;
        margin-bottom: 1.5rem !important;
    }
    .back-btn-container {
        margin-bottom: 1rem;
    }
    .back-btn {
        display: inline-flex;
        align-items: center;
        text-decoration: none;
        color: #2C2A29;
        font-size: 0.85rem;
        font-weight: bold;
        background-color: rgba(44, 42, 41, 0.05);
        padding: 6px 12px;
        border-radius: 6px;
        border: 1px solid rgba(44, 42, 41, 0.1);
        transition: all 0.2s ease;
    }
    .back-btn:hover {
        background-color: #1B4332;
        color: #FBF9F4;
        border-color: #1B4332;
    }
    h2, h3, [data-testid="stMarkdownContainer"] p strong {
        color: #1B4332 !important;
    }
    .stCodeBlock {
        border-radius: 8px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="back-btn-container"><a href="https://mathteacherpjh.github.io/MATH-Program/" target="_parent" class="back-btn">← HISTORY BACK</a></div>', unsafe_allow_html=True)

st.markdown('<h1>SENTENCE VECTORIZER ⚡</h1>', unsafe_allow_html=True)
st.markdown('<div class="custom-caption">파이썬 KoNLPy 대용량 사전 기반 고성능 로컬 문장 벡터화 시스템</div>', unsafe_allow_html=True)

@st.cache_resource
def get_okt_tokenizer():
    return Okt()

okt = get_okt_tokenizer()

def lemmatize_core(text):
    if not text.strip():
        return ""
    raw_tokens = okt.morphs(text, stem=True)
    return " ".join(raw_tokens)

left_panel, right_panel = st.columns([1.2, 2], gap="large")

with left_panel:
    st.subheader("문장 입력 (100자 이내)")
    
    if "input_fields_count" not in st.session_state:
        st.session_state.input_fields_count = 3
        
    sentences_inputs = []
    for i in range(st.session_state.input_fields_count):
        default_string = "불편한 편의점이라는 책을 읽고 감동을 받았습니다." if i == 0 else ""
        user_text = st.text_area(f"문장 {i+1}", value=default_string, max_chars=100, key=f"user_input_field_{i}", height=75)
        if user_text.strip():
            sentences_inputs.append(user_text.strip())
            
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("+ 추가", use_container_width=True):
            if st.session_state.input_fields_count < 10:
                st.session_state.input_fields_count += 1
                st.rerun()
    with btn_col2:
        if st.button("- 제거", use_container_width=True):
            if st.session_state.input_fields_count > 3:
                st.session_state.input_fields_count -= 1
                st.rerun()

with right_panel:
    st.subheader("분석 결과 및 벡터 변환")
    
    if sentences_inputs:
        processed_corpus = [lemmatize_core(s) for s in sentences_inputs]
        vectorizer_engine = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
        
        try:
            frequency_matrix = vectorizer_engine.fit_transform(processed_corpus).toarray()
            vocabulary_features = vectorizer_engine.get_feature_names_out()
            one_hot_matrix = np.where(frequency_matrix > 0, 1, 0)
            
            tab_summary, tab_vocab, tab_onehot, tab_freq = st.tabs(["전체보기", "1. 단어집합", "2. 원-핫 벡터", "3. 빈도수 벡터"])
            
            with tab_vocab:
                st.write("### 📌 구축된 단어집합 (Vocabulary)")
                badges_html = " ".join([f"`{token}`" for token in vocabulary_features])
                st.markdown(badges_html)
                
            with tab_onehot:
                st.write("### 🔢 원-핫 벡터 (One-Hot Vector)")
                for idx, (original_text, clean_text) in enumerate(zip(sentences_inputs, processed_corpus)):
                    st.markdown(f"**문장 {idx+1}:** \"{original_text}\"")
                    st.caption(f"→ 엔진 자동 추출 토큰 리스트: `[ {', '.join(clean_text.split())} ]`")
                    st.code(f"{one_hot_matrix[idx].tolist()}", language="json")
                    st.write("---")
                    
            with tab_freq:
                st.write("### 📊 빈도수 벡터 (Frequency Vector)")
                for idx, (original_text, clean_text) in enumerate(zip(sentences_inputs, processed_corpus)):
                    st.markdown(f"**문장 {idx+1}:** \"{original_text}\"")
                    st.caption(f"→ 엔진 자동 추출 토큰 리스트: `[ {', '.join(clean_text.split())} ]`")
                    st.code(f"{frequency_matrix[idx].tolist()}", language="json")
                    st.write("---")
                    
            with tab_summary:
                st.write("### 📋 종합 분석 데이터 요약")
                st.write("**단어 집합 매핑 인덱스 정보:**")
                mapping_labels = " | ".join([f"{index}: **{word}**" for index, word in enumerate(vocabulary_features)])
                st.info(mapping_labels)
                
                matrix_dataframe = pd.DataFrame(
                    frequency_matrix, 
                    columns=vocabulary_features, 
                    index=[f"문장 {k+1}" for k in range(len(sentences_inputs))]
                )
                st.write("**단어별 등장 빈도 매트릭스 (DTM):**")
                st.dataframe(matrix_dataframe, use_container_width=True)
                
        except ValueError:
            st.warning("유효한 형태소 토큰이 문장 안에 존재하지 않습니다.")
    else:
        st.info("왼쪽 패널에 텍스트를 입력하면 분석 변환 행렬이 실시간으로 출력됩니다.")
