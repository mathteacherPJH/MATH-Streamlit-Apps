import streamlit as st
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import os

# 자바 환경변수 세팅
os.environ['JAVA_HOME'] = '/usr/lib/jvm/default-java'

st.set_page_config(page_title="문장 벡터화 프로그램", layout="wide", initial_sidebar_state="collapsed")

# 🎨 스트림릿 위젯을 '원하는 거.png' 디자인으로 강제 개조하는 독한 CSS
st.markdown("""
    <style>
    /* 상하단 기본 요소 완전히 날리기 */
    [data-testid="stHeader"], footer, [data-testid="stFooterBlock"] { display: none !important; }
    
    :root {
        --c-bg-main: #FBF9F4; 
        --c-bg-card: rgba(255, 255, 255, 0.75); 
        --c-text-main: #2C2A29;
        --c-text-muted: #7D7569;
        --c-border: rgba(44, 42, 41, 0.12);
        --c-brand-green: #1B4332;
        --c-tab-bg: #F4F1EA;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--c-bg-main) !important;
        color: var(--c-text-main) !important;
        font-family: 'Malgun Gothic', -apple-system, sans-serif !important;
        overflow: hidden !important; /* 1페이지 고정을 위한 강제 스크롤 차단 */
    }

    /* 배경 캔버스 클릭 방지 */
    #neural-canvas {
        position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: 1; pointer-events: none;
    }

    /* 1. 상단 내비바 */
    .top-navbar {
        position: fixed; top: 0; left: 0; width: 100%; height: 50px;
        background-color: rgba(251, 249, 244, 0.85); backdrop-filter: blur(16px);
        border-bottom: 1px solid var(--c-border); z-index: 1000;
        padding: 0 24px; display: flex; align-items: center; justify-content: space-between;
    }
    .back-btn-link {
        color: var(--c-brand-green) !important; font-family: "Consolas", monospace; font-size: 13px; font-weight: 900;
        letter-spacing: 0.15em; text-decoration: none !important; cursor: pointer; display: flex; align-items: center; gap: 6px;
    }
    .navbar-brand-text { font-size: 13px; font-weight: bold; color: var(--c-brand-green); letter-spacing: 0.15em; text-transform: uppercase; }

    /* 2. 스트림릿 메인 레이아웃 100vh 피팅 (스크롤 없애기) */
    .block-container {
        padding: 70px 24px 20px 24px !important;
        max-width: 100% !important;
        height: 100vh !important;
    }
    div[data-testid="stHorizontalBlock"] {
        gap: 20px !important;
        height: 100% !important;
    }

    /* 3. 좌우 칼럼을 '카드' 디자인으로 직접 변환 */
    div[data-testid="stColumn"] {
        background-color: var(--c-bg-card) !important;
        border: 1px solid var(--c-border) !important;
        border-radius: 12px !important;
        padding: 24px !important;
        box-shadow: 0 10px 30px rgba(44, 42, 41, 0.03) !important;
        backdrop-filter: blur(6px) !important;
        height: calc(100vh - 95px) !important; 
        overflow-y: auto !important; /* 카드 내부만 스크롤 허용 */
    }

    /* 4. 좌측 컨트롤 버튼 및 입력창 디자인 */
    .stButton > button {
        border-radius: 6px !important;
        font-weight: bold !important;
        border: 1px solid var(--c-border) !important;
    }
    .main-run-btn > button {
        background-color: var(--c-brand-green) !important;
        color: white !important;
        padding: 12px !important;
        font-size: 16px !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 12px rgba(27, 67, 50, 0.15) !important;
    }
    .main-run-btn > button:hover { background-color: #123024 !important; }

    .custom-label { font-size: 11px; font-weight: bold; color: var(--c-brand-green); margin-bottom: 2px; }
    .custom-counter { font-size: 11px; color: var(--c-text-muted); text-align: right; margin-top: -12px; margin-bottom: 8px; position: relative; z-index: 10; padding-right: 8px;}
    
    .stTextArea textarea {
        border-radius: 8px !important;
        border: 1px solid var(--c-border) !important;
        font-size: 13px !important;
        background-color: rgba(255,255,255,0.8) !important;
        margin-bottom: 0px !important;
    }
    .stTextArea textarea:focus { border-color: var(--c-brand-green) !important; }

    /* 5. 📌 대망의 탭(Tab) 오버라이딩 (베이지 박스 + 하얀 입체 버튼) */
    div[data-testid="stTabList"] {
        display: flex !important;
        background-color: var(--c-tab-bg) !important;
        padding: 5px !important;
        border-radius: 8px !important;
        gap: 4px !important;
        border: 1px solid rgba(44,42,41,0.05) !important;
    }
    button[data-testid="stTab"] {
        flex: 1 !important;
        background: transparent !important;
        border: none !important;
        color: var(--c-text-muted) !important;
        font-size: 12px !important;
        font-weight: bold !important;
        padding: 8px 0 !important;
        border-radius: 6px !important;
    }
    button[data-testid="stTab"][aria-selected="true"] {
        background-color: white !important;
        color: var(--c-brand-green) !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05) !important;
    }
    /* 스트림릿 기본 빨간 밑줄 완전히 삭제 */
    div[data-baseweb="tab-highlight-title"], div[data-testid="stTab"] > div { display: none !important; }

    /* 6. 우측 결과 박스 및 Placeholder(빈 상자) 디자인 */
    .res-title { font-size: 13px; font-weight: bold; color: var(--c-brand-green); margin: 12px 0 6px 2px; }
    .res-box { 
        background-color: rgba(244, 241, 234, 0.5); padding: 14px; border-radius: 8px; 
        border: 1px solid var(--c-border); min-height: 55px; margin-bottom: 12px;
    }
    .placeholder-text { font-size: 12px; color: var(--c-text-muted); font-style: italic; }
    
    .vector-row { background: rgba(255,255,255,0.7); border: 1px solid var(--c-border); border-radius: 6px; padding: 8px 10px; margin-bottom: 8px; }
    .v-sum { font-size: 12px; font-weight: bold; color: var(--c-brand-green); margin-bottom: 2px; }
    .v-tok { font-size: 11px; color: var(--c-text-muted); margin-bottom: 4px; font-style: italic; }
    .v-mat { font-family: 'Consolas', monospace; font-size: 12px; background: white; padding: 4px 8px; border-radius: 4px; border: 1px solid rgba(44,42,41,0.05); color: #0D47A1; word-break: break-all; }
    .v-badge { background-color: #E8F5E9; color: #2E7D32; padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: bold; border: 1px solid rgba(165,214,167,0.6); display: inline-block; margin: 2px; }
    
    /* 스크롤바 디자인 */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(44,42,41,0.15); border-radius: 3px; }
    </style>

    <!-- 배경 애니메이션 -->
    <canvas id="neural-canvas"></canvas>
    <script>
    const bgCanvas = document.getElementById('neural-canvas');
    const bgCtx = bgCanvas.getContext('2d');
    let particles = [];
    function resizeBgCanvas() { bgCanvas.width = window.innerWidth; bgCanvas.height = window.innerHeight; initParticles(); }
    class Particle {
        constructor() { this.x = Math.random() * bgCanvas.width; this.y = Math.random() * bgCanvas.height; this.vx = (Math.random() - 0.5) * 0.35; this.vy = (Math.random() - 0.5) * 0.35; this.radius = Math.random() * 2 + 1.5; }
        update() { this.x += this.vx; this.y += this.vy; if (this.x < 0 || this.x > bgCanvas.width) this.vx *= -1; if (this.y < 0 || this.y > bgCanvas.height) this.vy *= -1; }
        draw() { bgCtx.beginPath(); bgCtx.arc(this.x, this.y, this.radius, 0, Math.PI * 2); bgCtx.fillStyle = 'rgba(27, 67, 50, 0.15)'; bgCtx.fill(); }
    }
    function initParticles() { particles = []; for (let i = 0; i < Math.floor((bgCanvas.width * bgCanvas.height) / 9000); i++) particles.push(new Particle()); }
    function animateBg() {
        bgCtx.clearRect(0, 0, bgCanvas.width, bgCanvas.height);
        for (let i = 0; i < particles.length; i++) {
            particles[i].update(); particles[i].draw();
            for (let j = i + 1; j < particles.length; j++) {
                const dist = Math.sqrt(Math.pow(particles[i].x - particles[j].x, 2) + Math.pow(particles[i].y - particles[j].y, 2));
                if (dist < 110) { bgCtx.beginPath(); bgCtx.moveTo(particles[i].x, particles[i].y); bgCtx.lineTo(particles[j].x, particles[j].y); bgCtx.strokeStyle = `rgba(27, 67, 50, ${0.08 * (1 - dist / 110)})`; bgCtx.lineWidth = 1.0; bgCtx.stroke(); }
            }
        }
        requestAnimationFrame(animateBg);
    }
    window.addEventListener('resize', resizeBgCanvas);
    setTimeout(() => { resizeBgCanvas(); animateBg(); }, 100);
    </script>
""", unsafe_allow_html=True)

# 상단 내비바
st.markdown("""
    <nav class="top-navbar">
        <a href="#" onclick="window.parent.history.back(); return false;" class="back-btn-link">← BACK</a>
        <div class="navbar-brand-text">SENTENCE VECTORIZER</div>
    </nav>
""", unsafe_allow_html=True)

# KoNLPy 캐싱
@st.cache_resource
def get_okt_tokenizer():
    return Okt()
okt = get_okt_tokenizer()

def lemmatize_core(text):
    if not text.strip(): return ""
    return " ".join(okt.morphs(text, stem=True))

# ---------------------------------------------------------
# 메인 레이아웃 구성
# ---------------------------------------------------------
left_col, right_col = st.columns([1.1, 1.9])

# ====== [좌측 패널] ======
with left_col:
    if "input_fields_count" not in st.session_state:
        st.session_state.input_fields_count = 3
        
    # 타이틀과 [제거/추가] 버튼을 일렬로 나란히
    t_col, c1, c2 = st.columns([2, 0.7, 0.7])
    with t_col:
        st.markdown('<div style="font-size:16px; font-weight:bold; color:#1B4332; margin-top:5px;">문장 입력 (100자 이내)</div>', unsafe_allow_html=True)
    with c1:
        if st.button("- 제거", use_container_width=True) and st.session_state.input_fields_count > 3:
            st.session_state.input_fields_count -= 1
            st.rerun()
    with c2:
        if st.button("+ 추가", use_container_width=True) and st.session_state.input_fields_count < 10:
            st.session_state.input_fields_count += 1
            st.rerun()
            
    st.markdown('<div style="margin-bottom: 10px;"></div>', unsafe_allow_html=True)

    sentences_inputs = []
    for i in range(st.session_state.input_fields_count):
        st.markdown(f'<div class="custom-label">문장 {i+1}</div>', unsafe_allow_html=True)
        default_string = "불편한 편의점이라는 책을 읽고 감동을 받았습니다." if i == 0 else ""
        
        user_text = st.text_area(label=f"hidden_{i}", value=default_string, max_chars=100, key=f"f_{i}", label_visibility="collapsed", height=68)
        # 글자 수 카운터
        st.markdown(f'<div class="custom-counter">{len(user_text)}/100</div>', unsafe_allow_html=True)
        
        if user_text.strip():
            sentences_inputs.append(user_text.strip())
    
    st.markdown('<div class="main-run-btn">', unsafe_allow_html=True)
    execute_analysis = st.button("분석 실행 ⚡", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ====== [우측 패널] ======
with right_col:
    # 📌 베이지색 디자인으로 덮어씌워진 탭
    tab_all, tab_vocab, tab_onehot, tab_freq = st.tabs(["전체보기", "1. 단어집합", "2. 원-핫 벡터", "3. 빈도수 벡터"])
    
    if "custom_vocab" not in st.session_state:
        st.session_state.custom_vocab = []

    has_results = execute_analysis and len(sentences_inputs) > 0
    
    if has_results:
        processed_corpus = [lemmatize_core(s) for s in sentences_inputs]
        vectorizer_engine = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
        try:
            frequency_matrix = vectorizer_engine.fit_transform(processed_corpus).toarray()
            vocabulary_features = list(vectorizer_engine.get_feature_names_out())
            for custom_w in st.session_state.custom_vocab:
                if custom_w not in vocabulary_features: vocabulary_features.append(custom_w)
            one_hot_matrix = np.where(frequency_matrix > 0, 1, 0)
        except ValueError:
            has_results = False

    # 📌 [중복 키 에러 방지] 탭별로 key_suffix를 다르게 부여
    def render_vocab_section(key_suffix):
        v1, v2, v3 = st.columns([2.5, 1.5, 0.7])
        with v1: st.markdown('<div class="res-title">1. 단어집합</div>', unsafe_allow_html=True)
        with v2: new_word = st.text_input("new", placeholder="새 원형 단어", key=f"n_word_{key_suffix}", label_visibility="collapsed")
        with v3: 
            if st.button("추가", type="primary", key=f"btn_{key_suffix}", use_container_width=True):
                if new_word.strip() and new_word.strip() not in st.session_state.custom_vocab:
                    st.session_state.custom_vocab.append(new_word.strip())
                    st.rerun()
                    
        st.markdown('<div class="res-box">', unsafe_allow_html=True)
        if has_results:
            badges = "".join([f'<span class="v-badge">{token}</span>' for token in vocabulary_features])
            st.markdown(badges, unsafe_allow_html=True)
        else:
            st.markdown('<span class="placeholder-text">문장을 입력하고 분석 실행을 누르면 단어집합 토큰이 표시됩니다.</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    def render_onehot_section():
        st.markdown('<div class="res-title">2. 원-핫 벡터 (One-Hot Vector)</div><div class="res-box">', unsafe_allow_html=True)
        if has_results:
            for idx, (original_text, clean_text) in enumerate(zip(sentences_inputs, processed_corpus)):
                st.markdown(f"""<div class="vector-row"><div class="v-sum">문장 {idx+1}: "{original_text}"</div><div class="v-tok">→ 추출 토큰: [ {", ".join(clean_text.split())} ]</div><div class="v-mat">[ {", ".join(map(str, one_hot_matrix[idx]))} ]</div></div>""", unsafe_allow_html=True)
        else:
            st.markdown('<span class="placeholder-text">문장별 0과 1의 원-핫 행렬이 표시되는 공간입니다.</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    def render_freq_section():
        st.markdown('<div class="res-title">3. 빈도수 벡터 (Frequency Vector)</div><div class="res-box">', unsafe_allow_html=True)
        if has_results:
            for idx, (original_text, clean_text) in enumerate(zip(sentences_inputs, processed_corpus)):
                st.markdown(f"""<div class="vector-row"><div class="v-sum">문장 {idx+1}: "{original_text}"</div><div class="v-tok">→ 추출 토큰: [ {", ".join(clean_text.split())} ]</div><div class="v-mat">[ {", ".join(map(str, frequency_matrix[idx]))} ]</div></div>""", unsafe_allow_html=True)
        else:
            st.markdown('<span class="placeholder-text">단어 등장 빈도수 행렬이 표시되는 공간입니다.</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_all:
        render_vocab_section("all")
        render_onehot_section()
        render_freq_section()
    with tab_vocab: render_vocab_section("t1")
    with tab_onehot: render_onehot_section()
    with tab_freq: render_freq_section()
