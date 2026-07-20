import streamlit as st
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import os

os.environ['JAVA_HOME'] = '/usr/lib/jvm/default-java'

st.set_page_config(page_title="문장 벡터화 프로그램", layout="wide", initial_sidebar_state="collapsed")

# 🎨 안정적인 오리지널 테마 CSS 주입
st.markdown("""
    <style>
    /* 상하단 불필요한 기본 헤더/푸터 숨김 */
    [data-testid="stHeader"], footer, [data-testid="stFooterBlock"] { display: none !important; }
    
    :root {
        --c-bg-main: #FBF9F4; 
        --c-bg-card: rgba(255, 255, 255, 0.85); 
        --c-text-main: #2C2A29;
        --c-text-muted: #7D7569;
        --c-border: rgba(44, 42, 41, 0.12);
        --c-brand-green: #1B4332;
        --c-brand-green-hover: #123024;
        --c-tag-bg: #E8F5E9;
        --c-tag-text: #2E7D32;
        --c-vector-text: #0D47A1;
        --c-tab-bg: #F4F1EA;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--c-bg-main) !important;
        color: var(--c-text-main) !important;
        font-family: 'Malgun Gothic', -apple-system, sans-serif !important;
    }

    /* 상단 탑바 디자인 */
    .top-navbar {
        position: fixed;
        top: 0; left: 0; width: 100%;
        background-color: rgba(251, 249, 244, 0.9);
        backdrop-filter: blur(12px);
        border-bottom: 1px solid var(--c-border);
        z-index: 1000;
        padding: 0 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-sizing: border-box;
        height: 50px;
    }
    .back-btn-link {
        background: transparent; border: none; color: var(--c-brand-green) !important;
        font-family: "Consolas", monospace; font-size: 13px; font-weight: 900;
        letter-spacing: 0.15em; cursor: pointer; display: flex; align-items: center;
        gap: 6px; text-decoration: none !important; transition: all 0.2s ease;
    }
    .back-btn-link:hover { transform: translateX(-3px); opacity: 0.8; }
    .navbar-brand-text { font-size: 13px; font-weight: bold; color: var(--c-brand-green); letter-spacing: 0.15em; text-transform: uppercase; }

    /* 메인 여백 세팅 */
    .block-container {
        padding-top: 65px !important;
        padding-bottom: 20px !important;
        padding-left: 20px !important;
        padding-right: 20px !important;
    }

    /* 카드 콘테이너 스타일 */
    .card-box {
        background-color: var(--c-bg-card);
        border: 1px solid var(--c-border);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(44, 42, 41, 0.03);
        margin-bottom: 15px;
    }

    .panel-title { 
        font-size: 15px; 
        color: var(--c-brand-green); 
        font-weight: bold; 
        margin-bottom: 12px;
    }

    /* 문장 입력 라벨 및 카운터 */
    .input-label-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
        font-size: 11px;
        font-weight: bold;
        color: var(--c-brand-green);
    }
    .char-count { font-weight: normal; color: var(--c-text-muted); }

    /* 입력 텍스트 에어리어 커스텀 */
    .stTextArea textarea {
        background-color: #ffffff !important;
        border: 1px solid var(--c-border) !important;
        border-radius: 8px !important;
        font-size: 13px !important;
        color: var(--c-text-main) !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--c-brand-green) !important;
    }

    /* 메인 분석 버튼 */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: bold !important;
    }
    
    /* 탭 디자인 오버라이딩 (안정적인 커스텀) */
    div[data-testid="stTabList"] {
        background-color: var(--c-tab-bg) !important;
        padding: 4px !important;
        border-radius: 8px !important;
        gap: 4px !important;
    }
    button[data-testid="stTab"] {
        border-radius: 6px !important;
        font-size: 12px !important;
        font-weight: bold !important;
        color: var(--c-text-muted) !important;
        border: none !important;
    }
    button[data-testid="stTab"][aria-selected="true"] {
        background-color: #ffffff !important;
        color: var(--c-brand-green) !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06) !important;
    }

    /* 결과 박스 세팅 */
    .result-section-title {
        font-size: 13px; font-weight: bold; color: var(--c-brand-green); margin: 12px 0 6px 0;
    }
    .result-box-native { 
        background-color: rgba(244, 241, 234, 0.6); padding: 12px; border-radius: 8px; min-height: 48px; 
        display: flex; flex-wrap: wrap; justify-content: flex-start; gap: 8px; align-items: center;
        border: 1px solid var(--c-border); box-sizing: border-box; width: 100%;
    }
    .vector-container { display: flex; flex-direction: column; gap: 8px; width: 100%; }
    .vector-row { background: #ffffff; border: 1px solid var(--c-border); border-radius: 6px; padding: 8px 10px; }
    .sentence-summary { font-size: 12px; font-weight: bold; color: var(--c-brand-green); margin-bottom: 2px; }
    .tokens-summary { font-size: 11px; color: var(--c-text-muted); margin-bottom: 4px; font-style: italic; }
    .vector-display {
        font-family: 'Consolas', monospace; font-size: 12px; background: #F8F9FA; padding: 4px 8px;
        border-radius: 4px; border: 1px solid rgba(44, 42, 41, 0.08); color: var(--c-vector-text); word-break: break-all;
    }
    .native-badge {
        background-color: var(--c-tag-bg); color: var(--c-tag-text); padding: 4px 10px; border-radius: 4px; font-size: 12px;
        border: 1px solid rgba(165, 214, 167, 0.6); font-weight: bold; display: inline-block;
    }
    .placeholder-text { font-size: 12px; color: var(--c-text-muted); font-style: italic; }
    </style>
""", unsafe_allow_html=True)

# 1. 상단 탑바 렌더링
st.markdown("""
    <nav class="top-navbar">
        <a href="#" onclick="window.parent.history.back(); return false;" class="back-btn-link">← BACK</a>
        <div class="navbar-brand-text">SENTENCE VECTORIZER</div>
    </nav>
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

# 2. 메인 좌우 레이아웃 분할
left_col, right_col = st.columns([1.1, 1.9], gap="medium")

with left_col:
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">문장 입력 (100자 이내)</div>', unsafe_allow_html=True)
    
    # 문장 수 제어 세션 초기화
    if "input_fields_count" not in st.session_state:
        st.session_state.input_fields_count = 3
        
    # + 추가 / - 제거 버튼 배치
    btn_c1, btn_c2 = st.columns(2)
    with btn_c1:
        if st.button("- 제거", use_container_width=True) and st.session_state.input_fields_count > 3:
            st.session_state.input_fields_count -= 1
            st.rerun()
    with btn_c2:
        if st.button("+ 추가", use_container_width=True) and st.session_state.input_fields_count < 10:
            st.session_state.input_fields_count += 1
            st.rerun()

    st.write("") # 미세 여백

    sentences_inputs = []
    for i in range(st.session_state.input_fields_count):
        default_string = "불편한 편의점이라는 책을 읽고 감동을 받았습니다." if i == 0 else ""
        
        # 문장 번호 및 글자수 표시 라벨 HTML
        st.markdown(f'''
            <div class="input-label-row">
                <span>문장 {i+1}</span>
            </div>
        ''', unsafe_allow_html=True)
        
        user_text = st.text_area(
            label=f"hidden_lbl_{i}", 
            value=default_string, 
            max_chars=100, 
            key=f"f_input_{i}", 
            height=68,
            label_visibility="collapsed"
        )
        
        if user_text.strip():
            sentences_inputs.append(user_text.strip())
            
    st.write("")
    execute_analysis = st.button("분석 실행 ⚡", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    
    # 📌 요청하신 4개의 탭 가로 복원
    tab_all, tab_vocab, tab_onehot, tab_freq = st.tabs([
        "전체보기", "1. 단어집합", "2. 원-핫 벡터", "3. 빈도수 벡터"
    ])
    
    has_results = execute_analysis and len(sentences_inputs) > 0
    
    if has_results:
        processed_corpus = [lemmatize_core(s) for s in sentences_inputs]
        vectorizer_engine = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
        try:
            frequency_matrix = vectorizer_engine.fit_transform(processed_corpus).toarray()
            vocabulary_features = vectorizer_engine.get_feature_names_out()
            one_hot_matrix = np.where(frequency_matrix > 0, 1, 0)
        except ValueError:
            has_results = False

    def render_vocab_section():
        st.markdown('<div class="result-section-title">1. 단어집합</div>', unsafe_allow_html=True)
        st.markdown('<div class="result-box-native">', unsafe_allow_html=True)
        if has_results:
            badges = "".join([f'<span class="native-badge" style="margin:2px;">{token}</span>' for token in vocabulary_features])
            st.markdown(badges, unsafe_allow_html=True)
        else:
            st.markdown('<span class="placeholder-text">문장을 입력하고 버튼을 누르면 생성된 단어집합 토큰이 표시됩니다.</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    def render_onehot_section():
        st.markdown('<div class="result-section-title">2. 원-핫 벡터 (One-Hot Vector)</div>', unsafe_allow_html=True)
        st.markdown('<div class="result-box-native"><div class="vector-container">', unsafe_allow_html=True)
        if has_results:
            for idx, (original_text, clean_text) in enumerate(zip(sentences_inputs, processed_corpus)):
                st.markdown(f"""
                    <div class="vector-row">
                        <div class="sentence-summary">문장 {idx+1}: "{original_text}"</div>
                        <div class="tokens-summary">→ 추출 토큰: [ {", ".join(clean_text.split())} ]</div>
                        <div class="vector-display">[ {", ".join(map(str, one_hot_matrix[idx]))} ]</div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<span class="placeholder-text">문장별 0과 1의 원-핫 행렬이 표시되는 공간입니다.</span>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

    def render_freq_section():
        st.markdown('<div class="result-section-title">3. 빈도수 벡터 (Frequency Vector)</div>', unsafe_allow_html=True)
        st.markdown('<div class="result-box-native"><div class="vector-container">', unsafe_allow_html=True)
        if has_results:
            for idx, (original_text, clean_text) in enumerate(zip(sentences_inputs, processed_corpus)):
                st.markdown(f"""
                    <div class="vector-row">
                        <div class="sentence-summary">문장 {idx+1}: "{original_text}"</div>
                        <div class="tokens-summary">→ 추출 토큰: [ {", ".join(clean_text.split())} ]</div>
                        <div class="vector-display">[ {", ".join(map(str, frequency_matrix[idx]))} ]</div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<span class="placeholder-text">단어 등장 빈도수 행렬이 표시되는 공간입니다.</span>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

    # 각 탭 내부 콘텐츠 출력
    with tab_all:
        render_vocab_section()
        render_onehot_section()
        render_freq_section()
    with tab_vocab:
        render_vocab_section()
    with tab_onehot:
        render_onehot_section()
    with tab_freq:
        render_freq_section()

    st.markdown('</div>', unsafe_allow_html=True)
