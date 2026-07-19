import streamlit as st
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import os

os.environ['JAVA_HOME'] = '/usr/lib/jvm/default-java'

st.set_page_config(page_title="문장 벡터화 프로그램", layout="wide", initial_sidebar_state="collapsed")

# 1. 기존 오리지널 CSS 스타일시트 및 배경 엔진 100% 주입
st.markdown("""
    <style>
    /* 스트림릿 기본 프레임워크 강제 리셋 */
    [data-testid="stHeader"] { display: none !important; }
    .block-container { padding: 0rem !important; max-width: 100% !important; }
    div[data-testid="stHorizontalBlock"] { gap: 0rem !important; }
    
    /* 원본 변수 및 레이아웃 시스템 복원 */
    :root {
        --c-bg-main: #FBF9F4; 
        --c-bg-card: rgba(255, 255, 255, 0.73); 
        --c-text-main: #2C2A29;
        --c-text-muted: #7D7569;
        --c-border: rgba(44, 42, 41, 0.1);
        --c-accent: #2383E2;
        --c-accent-hover: #1A6CB8;
        --c-brand-green: #1B4332;
        --c-brand-green-hover: #123024;
        --c-tag-bg: #E8F5E9;
        --c-tag-text: #2E7D32;
        --c-removed-bg: #FFEBEE;
        --c-removed-text: #C62828;
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
        z-index: 1;
        pointer-events: none;
    }

    /* 상단 탑바 컴포넌트 복원 */
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
    }
    .back-btn-link {
        background: transparent;
        border: none;
        color: var(--c-brand-green);
        font-family: "Consolas", monospace;
        font-size: 13px;
        font-weight: 900;
        letter-spacing: 0.15em;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 6px;
        text-decoration: none;
        transition: all 0.2s ease;
    }
    .back-btn-link:hover { transform: translateX(-3px); opacity: 0.8; }
    .navbar-brand-text { font-size: 13px; font-weight: bold; color: var(--c-brand-green); letter-spacing: 0.15em; text-transform: uppercase; }

    /* 메인 배치 스펙 복원 */
    .main-container-fixed {
        display: flex;
        width: 100vw;
        height: 100vh;
        padding: 24px;
        padding-top: 85px;
        gap: 20px;
        position: relative;
        z-index: 10;
        box-sizing: border-box;
    }

    .orig-card {
        background-color: var(--c-bg-card);
        border: 1px solid var(--c-border);
        border-radius: 12px;
        padding: 24px;
        box-sizing: border-box;
        backdrop-filter: blur(6px);
        box-shadow: 0 10px 30px rgba(44, 42, 41, 0.03);
        display: flex;
        flex-direction: column;
        height: 100%;
        width: 100%;
    }

    .panel-title { margin: 0 0 12px 0; font-size: 16px; color: var(--c-brand-green); font-weight: bold; display: flex; justify-content: space-between; align-items: center; }
    
    /* 상단 내부 수동 조절 버튼 크기 세팅 */
    .stButton div button {
        padding: 4px 10px !important; font-size: 12px !important; font-weight: bold !important; border: 1px solid var(--c-border) !important;
        background: white !important; border-radius: 4px !important; color: var(--c-text-main) !important; transition: all 0.2s !important;
        height: auto !important; width: auto !important; line-height: 1.5 !important;
    }
    .stButton div button:hover { background: #eee !important; color: var(--c-text-main) !important; border-color: var(--c-border) !important; }

    /* 좌측 내부 메인 분석 실행 버튼 커스텀 (따로 지정) */
    .run-btn-container div button {
        width: 100% !important; padding: 14px 20px !important; background-color: var(--c-brand-green) !important; color: white !important; border: none !important; border-radius: 8px !important;
        font-size: 16px !important; font-weight: bold !important; letter-spacing: 0.05em !important;
        box-shadow: 0 4px 12px rgba(27, 67, 50, 0.15) !important; height: auto !important;
    }
    .run-btn-container div button:hover { background-color: var(--c-brand-green-hover) !important; color: white !important; }

    .textarea-list-scroll { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; margin-bottom: 16px; padding-right: 4px; }
    .textarea-wrapper { position: relative; display: flex; flex-direction: column; width: 100%; }
    .textarea-wrapper span { font-size: 11px; color: var(--c-brand-green); margin-bottom: 4px; font-weight: bold; }
    .textarea-wrapper .char-counter { position: absolute; bottom: 6px; right: 12px; font-size: 11px; color: var(--c-text-muted); pointer-events: none; z-index: 99; }

    .stTextArea textarea {
        width: 100% !important; height: 70px !important; padding: 10px 12px !important; padding-bottom: 22px !important; box-sizing: border-box !important;
        border: 1px solid var(--c-border) !important; border-radius: 8px !important; resize: none !important; font-size: 14px !important;
        line-height: 1.5 !important; background-color: rgba(255, 255, 255, 0.8) !important; color: var(--c-text-main) !important; outline: none !important;
    }
    .stTextArea textarea:focus { border-color: var(--c-brand-green) !important; box-shadow: none !important; }
    .stTextArea div[data-baseweb="textarea"] { border: none !important; background: transparent !important; }

    /* 우측 탭 컴포넌트 100% 복원 */
    div[data-testid="stTabList"] {
        display: flex !important; gap: 4px !important; background: var(--c-tab-bg) !important; padding: 4px !important; border-radius: 8px !important; border: 1px solid rgba(44, 42, 41, 0.05) !important; width: 100% !important;
    }
    button[data-testid="stTab"] {
        flex: 1 !important; border: none !important; background: transparent !important; padding: 8px 4px !important; font-size: 12px !important; font-weight: bold !important;
        color: var(--c-text-muted) !important; cursor: pointer !important; border-radius: 6px !important; transition: all 0.2s !important; white-space: nowrap !important; text-align: center !important;
    }
    button[data-testid="stTab"][aria-selected="true"] {
        background: #FFFFFF !important; color: var(--c-brand-green) !important; box-shadow: 0 2px 6px rgba(44,42,41,0.08) !important; border: none !important;
    }
    div[data-testid="stTabTabpanel"] { border: none !important; padding-top: 15px !important; height: calc(100vh - 200px) !important; overflow-y: auto !important; padding-right: 4px !important; }

    /* 결과 박스 내부 테두리 요소 복원 */
    .right-panel-content-area { display: flex; flex-direction: column; gap: 18px; width: 100%; box-sizing: border-box; }
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

    /* 단어집합 배지 디자인 스펙 싱크로 */
    .native-badge {
        background-color: var(--c-tag-bg); color: var(--c-tag-text); padding: 6px 12px; border-radius: 4px; font-size: 13px;
        border: 1px solid rgba(165, 214, 167, 0.6); font-weight: bold; display: inline-block;
    }

    /* 스크롤바 바인딩 */
    .textarea-list-scroll::-webkit-scrollbar, div[data-testid="stTabTabpanel"]::-webkit-scrollbar { width: 6px; }
    .textarea-list-scroll::-webkit-scrollbar-track, div[data-testid="stTabTabpanel"]::-webkit-scrollbar-track { background: transparent; }
    .textarea-list-scroll::-webkit-scrollbar-thumb, div[data-testid="stTabTabpanel"]::-webkit-scrollbar-thumb { background: rgba(44, 42, 41, 0.15); border-radius: 3px; }
    </style>

    <!-- 배경 뉴럴 구조 인터랙티브 캔버스 탑재 -->
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
    
    // 즉시 실행 트리거
    setTimeout(() => { resizeBgCanvas(); animateBg(); }, 100);
    </script>
""", unsafe_allow_html=True)

# 2. 고정 탑 내비게이션 바 컴포넌트 렌더링
st.markdown("""
    <nav class="top-navbar">
        <a href="https://mathteacherpjh.github.io/MATH-Program/" target="_parent" class="back-btn-link">← BACK</a>
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

# 3. 레이아웃 분할 파싱
main_layout_div = st.container()
with main_layout_div:
    st.markdown('<div class="main-container-fixed">', unsafe_allow_html=True)
    
    # 가로축 비율 매핑을 위한 칼럼 정의
    left_col, right_col = st.columns([1.2, 2])
    
    with left_col:
        st.markdown('<div class="orig-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title"><span>문장 입력 (100자 이내)</span></div>', unsafe_allow_html=True)
        
        # 입력 필드 카운트 세션 초기화
        if "input_fields_count" not in st.session_state:
            st.session_state.input_fields_count = 3
            
        # 플러스 마이너스 기능 구현을 위한 버튼 배치
        ctrl_col1, ctrl_col2 = st.columns(2)
        with ctrl_col1:
            if st.button("- 제거") and st.session_state.input_fields_count > 3:
                st.session_state.input_fields_count -= 1
                st.rerun()
        with ctrl_col2:
            if st.button("+ 추가") and st.session_state.input_fields_count < 10:
                st.session_state.input_fields_count += 1
                st.rerun()
                
        st.markdown('<div class="textarea-list-scroll">', unsafe_allow_html=True)
        sentences_inputs = []
        for i in range(st.session_state.input_fields_count):
            st.markdown(f'<div class="textarea-wrapper"><span>문장 {i+1}</span>', unsafe_allow_html=True)
            
            default_string = "불편한 편의점이라는 책을 읽고 감동을 받았습니다." if i == 0 else ""
            user_text = st.text_area(
                label=f"input_lbl_{i}", 
                value=default_string, 
                max_chars=100, 
                key=f"user_input_field_{i}",
                label_visibility="collapsed"
            )
            
            # 실시간 글자수 카운터 매핑
            st.markdown(f'<div class="char-counter">{len(user_text)}/100</div></div>', unsafe_allow_html=True)
            if user_text.strip():
                sentences_inputs.append(user_text.strip())
                
        st.markdown('</div>', unsafe_allow_html=True) # 리스트 스크롤 엔드
        
        # 메인 트리거 버튼 구역
        st.markdown('<div class="run-btn-container">', unsafe_allow_html=True)
        execute_analysis = st.button("분석 실행 ⚡")
        st.markdown('</div></div>', unsafe_allow_html=True) # 레프트 카드 엔드

    with right_col:
        st.markdown('<div class="orig-card">', unsafe_allow_html=True)
        
        # 기본 데이터 세팅 유무 판정 후 탭 활성화
        tab_all, tab_vocab, tab_onehot, tab_freq = st.tabs([
            "전체보기", "1. 단어집합", "2. 원-핫 벡터", "3. 빈도수 벡터"
        ])
        
        if execute_analysis and sentences_inputs:
            processed_corpus = [lemmatize_core(s) for s in sentences_inputs]
            vectorizer_engine = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
            
            try:
                frequency_matrix = vectorizer_engine.fit_transform(processed_corpus).toarray()
                vocabulary_features = vectorizer_engine.get_feature_names_out()
                one_hot_matrix = np.where(frequency_matrix > 0, 1, 0)
                
                # 가독 데이터 구조 렌더링 펑션화 대체
                def render_vocab_section():
                    st.markdown('<div class="result-section"><h3>1. 단어집합</h3><div class="result-box-native">', unsafe_allow_html=True)
                    badges = "".join([f'<span class="native-badge" style="margin:4px;">{token}</span>' for token in vocabulary_features])
                    st.markdown(badges, unsafe_allow_html=True)
                    st.markdown('</div></div><br>', unsafe_allow_html=True)

                def render_onehot_section():
                    st.markdown('<div class="result-section"><h3>2. 원-핫 벡터 (One-Hot Vector)</h3><div class="result-box-native" style="display:block;"><div class="vector-container">', unsafe_allow_html=True)
                    for idx, (original_text, clean_text) in enumerate(zip(sentences_inputs, processed_corpus)):
                        tokens_line = ", ".join(clean_text.split())
                        st.markdown(f"""
                            <div class="vector-row">
                                <div class="sentence-summary">문장 {idx+1}: "{original_text}"</div>
                                <div class="tokens-summary">→ 로컬 엔진 추출 복원 기본형 리스트: [ {tokens_line} ]</div>
                                <div class="vector-display">[ {", ".join(map(str, one_hot_matrix[idx]))} ]</div>
                            </div>
                        """, unsafe_allow_html=True)
                    st.markdown('</div></div></div><br>', unsafe_allow_html=True)

                def render_freq_section():
                    st.markdown('<div class="result-section"><h3>3. 빈도수 벡터 (Frequency Vector)</h3><div class="result-box-native" style="display:block;"><div class="vector-container">', unsafe_allow_html=True)
                    for idx, (original_text, clean_text) in enumerate(zip(sentences_inputs, processed_corpus)):
                        tokens_line = ", ".join(clean_text.split())
                        st.markdown(f"""
                            <div class="vector-row">
                                <div class="sentence-summary">문장 {idx+1}: "{original_text}"</div>
                                <div class="tokens-summary">→ 로컬 엔진 추출 복원 기본형 리스트: [ {tokens_line} ]</div>
                                <div class="vector-display">[ {", ".join(map(str, frequency_matrix[idx]))} ]</div>
                            </div>
                        """, unsafe_allow_html=True)
                    st.markdown('</div></div></div>', unsafe_allow_html=True)

                # 탭별 독립 스코프 마킹
                with tab_all:
                    st.markdown('<div class="right-panel-content-area">', unsafe_allow_html=True)
                    render_vocab_section()
                    render_onehot_section()
                    render_freq_section()
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with tab_vocab:
                    st.markdown('<div class="right-panel-content-area">', unsafe_allow_html=True)
                    render_vocab_section()
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with tab_onehot:
                    st.markdown('<div class="right-panel-content-area">', unsafe_allow_html=True)
                    render_onehot_section()
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with tab_freq:
                    st.markdown('<div class="right-panel-content-area">', unsafe_allow_html=True)
                    render_freq_section()
                    st.markdown('</div>', unsafe_allow_html=True)

            except ValueError:
                st.warning("유효한 형태소 토큰이 문장 안에 존재하지 않습니다.")
        else:
            st.markdown("""
                <div style='color: #7D7569; font-size: 14px; text-align: center; padding-top: 150px; font-weight: bold;'>
                    왼쪽 패널에 문장을 입력한 뒤 '분석 실행 ⚡' 버튼을 누르면<br>수학적 변환 행렬이 이곳에 실시간으로 출력됩니다.
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True) # 라이트 카드 엔드
        
    st.markdown('</div>', unsafe_allow_html=True) # 전체 컨테이너 엔드
