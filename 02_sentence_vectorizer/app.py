import streamlit as st
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import os

os.environ['JAVA_HOME'] = '/usr/lib/jvm/default-java'

st.set_page_config(page_title="문장 벡터화 프로그램", layout="wide", initial_sidebar_state="collapsed")

# 1. 스트림릿 순정 칼럼(column)을 원본 카드 디자인으로 오버라이딩하는 CSS
st.markdown("""
    <style>
    /* 기본 스트림릿 패딩 및 여백 완전 초기화 */
    [data-testid="stHeader"], [data-testid="stFooterBlock"] { display: none !important; }
    .block-container { padding: 0rem !important; max-width: 100% !important; }
    
    :root {
        --c-bg-main: #FBF9F4; 
        --c-bg-card: rgba(255, 255, 255, 0.73); 
        --c-text-main: #2C2A29;
        --c-text-muted: #7D7569;
        --c-border: rgba(44, 42, 41, 0.1);
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
        letter-spacing: 0.03em !important;
        margin: 0 !important;
        padding: 0 !important;
        box-sizing: border-box !important;
    }

    #neural-canvas {
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        z-index: 1;
        pointer-events: none;
    }

    /* 고정 상단 탑바 */
    .top-navbar {
        position: fixed;
        top: 0; left: 0; width: 100%;
        background-color: rgba(251, 249, 244, 0.85);
        backdrop-filter: blur(16px);
        border-bottom: 1px solid var(--c-border);
        z-index: 1000;
        padding: 14px 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-sizing: border-box;
        height: 55px;
    }
    
    .back-btn-link {
        background: transparent; border: none; color: var(--c-brand-green) !important;
        font-family: "Consolas", monospace; font-size: 13px; font-weight: 900;
        letter-spacing: 0.15em; cursor: pointer; display: flex; align-items: center;
        gap: 6px; text-decoration: none !important; transition: all 0.2s ease;
    }
    .back-btn-link:hover { transform: translateX(-3px); opacity: 0.8; text-decoration: none !important; }
    .navbar-brand-text { font-size: 13px; font-weight: bold; color: var(--c-brand-green); letter-spacing: 0.15em; text-transform: uppercase; }

    /* 메인 칼럼 배치 구역 */
    .stHorizontalBlock {
        padding: 75px 24px 24px 24px !important;
        position: relative;
        z-index: 10;
        gap: 20px !important;
    }

    /* 📌 스트림릿 칼럼 자체를 '카드 상자'로 변신시키는 핵심 스타일 */
    div[data-testid="stColumn"] > div {
        background-color: var(--c-bg-card) !important;
        border: 1px solid var(--c-border) !important;
        border-radius: 12px !important;
        padding: 24px !important;
        backdrop-filter: blur(6px) !important;
        box-shadow: 0 10px 30px rgba(44, 42, 41, 0.03) !important;
        min-height: calc(100vh - 105px) !important;
        box-sizing: border-box !important;
    }

    .panel-title { margin: 0 0 12px 0; font-size: 16px; color: var(--c-brand-green); font-weight: bold; display: flex; justify-content: space-between; align-items: center; }
    
    /* 입력 박스 레이아웃 매칭 */
    .textarea-wrapper { position: relative; display: flex; flex-direction: column; width: 100%; margin-bottom: 12px; }
    .textarea-wrapper span { font-size: 11px; color: var(--c-brand-green); margin-bottom: 4px; font-weight: bold; }
    .textarea-wrapper .char-counter { position: absolute; bottom: 8px; right: 12px; font-size: 11px; color: var(--c-text-muted); pointer-events: none; z-index: 99; }

    .stTextArea textarea {
        width: 100% !important; height: 70px !important; padding: 10px 12px !important; padding-bottom: 24px !important; box-sizing: border-box !important;
        border: 1px solid var(--c-border) !important; border-radius: 8px !important; resize: none !important; font-size: 14px !important;
        line-height: 1.5 !important; background-color: rgba(255, 255, 255, 0.8) !important; color: var(--c-text-main) !important; outline: none !important;
    }
    .stTextArea textarea:focus { border-color: var(--c-brand-green) !important; }
    .stTextArea div[data-baseweb="textarea"] { border: none !important; background: transparent !important; }

    /* 상단 조작 버튼 (+ 추가, - 제거) */
    .stButton div button {
        padding: 4px 10px !important; font-size: 12px !important; font-weight: bold !important; border: 1px solid var(--c-border) !important;
        background: white !important; border-radius: 4px !important; color: var(--c-text-main) !important; transition: all 0.2s !important;
    }
    .stButton div button:hover { background: #eee !important; color: var(--c-text-main) !important; }

    /* 하단 대형 분석 실행 버튼 */
    .run-btn-box div button {
        width: 100% !important; padding: 14px 20px !important; background-color: var(--c-brand-green) !important; color: white !important; border: none !important; border-radius: 8px !important;
        font-size: 16px !important; font-weight: bold !important; letter-spacing: 0.05em !important;
        box-shadow: 0 4px 12px rgba(27, 67, 50, 0.15) !important; margin-top: 10px !important;
    }
    .run-btn-box div button:hover { background-color: var(--c-brand-green-hover) !important; color: white !important; }

    /* 우측 탭 커스텀 */
    div[data-testid="stTabList"] {
        display: flex !important; gap: 4px !important; background: var(--c-tab-bg) !important; padding: 4px !important; border-radius: 8px !important; border: 1px solid rgba(44, 42, 41, 0.05) !important; width: 100% !important;
    }
    button[data-testid="stTab"] {
        flex: 1 !important; border: none !important; background: transparent !important; padding: 8px 4px !important; font-size: 12px !important; font-weight: bold !important;
        color: var(--c-text-muted) !important; border-radius: 6px !important; transition: all 0.2s !important; text-align: center !important;
    }
    button[data-testid="stTab"][aria-selected="true"] {
        background: #FFFFFF !important; color: var(--c-brand-green) !important; box-shadow: 0 2px 6px rgba(44,42,41,0.08) !important;
    }
    div[data-testid="stTabTabpanel"] { border: none !important; padding-top: 18px !important; }

    /* 결과물 카드 표현 */
    .result-section h3 { margin: 0 0 8px 4px; font-size: 14px; color: var(--c-brand-green); font-weight: bold; }
    .result-box-native { 
        background-color: rgba(244, 241, 234, 0.6); padding: 14px; border-radius: 8px; min-height: 48px; 
        display: flex; flex-wrap: wrap; justify-content: flex-start; gap: 8px; align-items: center;
        border: 1px solid var(--c-border); box-sizing: border-box; width: 100%; 
    }

    .vector-container { display: flex; flex-direction: column; gap: 10px; width: 100%; }
    .vector-row { background: rgba(255, 255, 255, 0.6); border: 1px solid var(--c-border); border-radius: 6px; padding: 10px 12px; }
    .vector-row .sentence-summary { font-size: 12px; font-weight: bold; color: var(--c-brand-green); margin-bottom: 4px; }
    .vector-row .tokens-summary { font-size: 11px; color: var(--c-text-muted); margin-bottom: 6px; font-style: italic; }
    .vector-display {
        font-family: 'Consolas', monospace; font-size: 13px; background: #fff; padding: 6px 10px;
        border-radius: 4px; border: 1px solid rgba(44, 42, 41, 0.05); color: var(--c-vector-text); word-break: break-all;
    }

    .native-badge {
        background-color: var(--c-tag-bg); color: var(--c-tag-text); padding: 6px 12px; border-radius: 4px; font-size: 13px;
        border: 1px solid rgba(165, 214, 167, 0.6); font-weight: bold; display: inline-block;
    }
    </style>

    <canvas id="neural-canvas"></canvas>
    <script>
    const bgCanvas = document.getElementById('neural-canvas');
    const bgCtx = bgCanvas.getContext('2d');
    let particles = [];
    const mouse = { x: null, y: null, radius: 150 };

    function resizeBgCanvas() {
        bgCanvas.width = window.innerWidth;
        bgCanvas.height = window.innerHeight;
        initParticles();
    }

    class Particle {
        constructor() {
            this.x = Math.random() * bgCanvas.width;
            this.y = Math.random() * bgCanvas.height;
            this.vx = (Math.random() - 0.5) * 0.35;
            this.vy = (Math.random() - 0.5) * 0.35;
            this.radius = Math.random() * 2 + 1.5; 
        }
        update() {
            this.x += this.vx; this.y += this.vy;
            if (this.x < 0 || this.x > bgCanvas.width) this.vx *= -1;
            if (this.y < 0 || this.y > bgCanvas.height) this.vy *= -1;
            let dx = mouse.x - this.x; let dy = mouse.y - this.y;
            let distance = Math.sqrt(dx * dx + dy * dy);
            if (distance < mouse.radius) {
                const force = (mouse.radius - distance) / mouse.radius;
                this.x -= dx * force * 0.03; this.y -= dy * force * 0.03;
            }
        }
        draw() {
            bgCtx.beginPath(); bgCtx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            bgCtx.fillStyle = 'rgba(27, 67, 50, 0.15)'; bgCtx.fill();
        }
    }

    function initParticles() {
        particles = [];
        const particleCount = Math.floor((bgCanvas.width * bgCanvas.height) / 9000);
        for (let i = 0; i < particleCount; i++) particles.push(new Particle());
    }

    function animateBg() {
        bgCtx.clearRect(0, 0, bgCanvas.width, bgCanvas.height);
        for (let i = 0; i < particles.length; i++) {
            particles[i].update(); particles[i].draw();
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x; const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 110) {
                    bgCtx.beginPath(); bgCtx.moveTo(particles[i].x, particles[i].y); bgCtx.lineTo(particles[j].x, particles[j].y);
                    bgCtx.strokeStyle = `rgba(27, 67, 50, ${0.08 * (1 - dist / 110)})`; bgCtx.lineWidth = 1.0; bgCtx.stroke();
                }
            }
        }
        requestAnimationFrame(animateBg);
    }

    window.addEventListener('mousemove', (e) => { mouse.x = e.clientX; mouse.y = e.clientY; });
    window.addEventListener('mouseout', () => { mouse.x = null; mouse.y = null; });
    window.addEventListener('resize', resizeBgCanvas);
    
    setTimeout(() => { resizeBgCanvas(); animateBg(); }, 100);
    </script>
""", unsafe_allow_html=True)

# 2. 상단 탑바
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

# 3. 메인 레이아웃 (좌우 칼럼)
left_col, right_col = st.columns([1.2, 2])

with left_col:
    if "input_fields_count" not in st.session_state:
        st.session_state.input_fields_count = 3
        
    st.markdown('<div class="panel-title"><span>문장 입력 (100자 이내)</span></div>', unsafe_allow_html=True)
    
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("- 제거") and st.session_state.input_fields_count > 3:
            st.session_state.input_fields_count -= 1
            st.rerun()
    with btn_col2:
        if st.button("+ 추가") and st.session_state.input_fields_count < 10:
            st.session_state.input_fields_count += 1
            st.rerun()
            
    sentences_inputs = []
    for i in range(st.session_state.input_fields_count):
        st.markdown(f'<div class="textarea-wrapper"><span>문장 {i+1}</span>', unsafe_allow_html=True)
        default_string = "불편한 편의점이라는 책을 읽고 감동을 받았습니다." if i == 0 else ""
        user_text = st.text_area(label=f"lbl_{i}", value=default_string, max_chars=100, key=f"f_{i}", label_visibility="collapsed")
        st.markdown(f'<div class="char-counter">{len(user_text)}/100</div></div>', unsafe_allow_html=True)
        if user_text.strip():
            sentences_inputs.append(user_text.strip())
    
    st.markdown('<div class="run-btn-box">', unsafe_allow_html=True)
    execute_analysis = st.button("분석 실행 ⚡")
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    tab_all, tab_vocab, tab_onehot, tab_freq = st.tabs(["전체보기", "1. 단어집합", "2. 원-핫 벡터", "3. 빈도수 벡터"])
    
    if execute_analysis and sentences_inputs:
        processed_corpus = [lemmatize_core(s) for s in sentences_inputs]
        vectorizer_engine = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
        
        try:
            frequency_matrix = vectorizer_engine.fit_transform(processed_corpus).toarray()
            vocabulary_features = vectorizer_engine.get_feature_names_out()
            one_hot_matrix = np.where(frequency_matrix > 0, 1, 0)
            
            def render_vocab_section():
                st.markdown('<div class="result-section"><h3>1. 단어집합</h3><div class="result-box-native">', unsafe_allow_html=True)
                badges = "".join([f'<span class="native-badge" style="margin:4px;">{token}</span>' for token in vocabulary_features])
                st.markdown(badges, unsafe_allow_html=True)
                st.markdown('</div></div><br>', unsafe_allow_html=True)

            def render_onehot_section():
                st.markdown('<div class="result-section"><h3>2. 원-핫 벡터 (One-Hot Vector)</h3><div class="result-box-native" style="display:block;"><div class="vector-container">', unsafe_allow_html=True)
                for idx, (original_text, clean_text) in enumerate(zip(sentences_inputs, processed_corpus)):
                    st.markdown(f"""
                        <div class="vector-row">
                            <div class="sentence-summary">문장 {idx+1}: "{original_text}"</div>
                            <div class="tokens-summary">→ 로컬 엔진 추출 복원 기본형 리스트: [ {", ".join(clean_text.split())} ]</div>
                            <div class="vector-display">[ {", ".join(map(str, one_hot_matrix[idx]))} ]</div>
                        </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div></div></div><br>', unsafe_allow_html=True)

            def render_freq_section():
                st.markdown('<div class="result-section"><h3>3. 빈도수 벡터 (Frequency Vector)</h3><div class="result-box-native" style="display:block;"><div class="vector-container">', unsafe_allow_html=True)
                for idx, (original_text, clean_text) in enumerate(zip(sentences_inputs, processed_corpus)):
                    st.markdown(f"""
                        <div class="vector-row">
                            <div class="sentence-summary">문장 {idx+1}: "{original_text}"</div>
                            <div class="tokens-summary">→ 로컬 엔진 추출 복원 기본형 리스트: [ {", ".join(clean_text.split())} ]</div>
                            <div class="vector-display">[ {", ".join(map(str, frequency_matrix[idx]))} ]</div>
                        </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div></div></div>', unsafe_allow_html=True)

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

        except ValueError:
            st.warning("유효한 형태소 토큰이 문장 안에 존재하지 않습니다.")
    else:
        st.markdown("""
            <div style='color: #7D7569; font-size: 14px; text-align: center; padding-top: 150px; font-weight: bold;'>
                왼쪽 패널에 문장을 입력한 뒤 '분석 실행 ⚡' 버튼을 누르면<br>수학적 변환 행렬이 이곳에 실시간으로 출력됩니다.
            </div>
        """, unsafe_allow_html=True)
