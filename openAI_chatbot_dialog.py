import streamlit as st
import time
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")



# 세션 상태 초기화
if 'chat_open' not in st.session_state:
    st.session_state.chat_open = False
if 'messages' not in st.session_state:
    st.session_state.messages = []

# API 호출 함수
def call_api(user_message):
    # Open API 인증 및 객체생성
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        completion = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {
                # 시스템 프롬프트
                    "role" : "system",
                    "content" : "너는 달리기 강습 전문가야."
                },
                # 사용자 프롬프트
                {
                    "role":"user",
                    "content" : user_message
                }
            ]
        )
        return completion.choices[0].message.content
     
    except Exception as e:
        return f"오류 발생: {str(e)}"

# 커스텀 CSS
st.markdown("""
<style>
    /* 챗봇 토글 버튼을 우측 하단에 고정 */
    .stButton button[kind="secondary"] {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        font-size: 24px;
        z-index: 1000;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    /* 챗봇 컨테이너 스타일 */
    .chatbot-box {
        position: fixed;
        bottom: 90px;
        right: 20px;
        width: 400px;
        max-height: 600px;
        background: white;
        border-radius: 15px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        z-index: 999;
        padding: 20px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)


# 챗봇 토글 버튼 (우측 하단 고정)
# 빈 컬럼을 사용하여 우측에 배치
cols = st.columns([10, 1])
with cols[1]:
    if st.button("💬", key="chatbot_btn", type="secondary"):
        st.session_state.chat_open = not st.session_state.chat_open

# 챗봇이 열려있을 때
if st.session_state.chat_open:
    # 플로팅 박스처럼 보이게 하기
    with st.container():
        st.markdown("---")
        
        # 챗봇 헤더
        header_col1, header_col2 = st.columns([4, 1])
        with header_col1:
            st.markdown("### 💬 AI 챗봇")
        with header_col2:
            if st.button("✕", key="close_chat"):
                st.session_state.chat_open = False
                st.rerun()
        
        st.caption("무엇을 도와드릴까요?")
        
        # 채팅 히스토리 표시 영역
        chat_container = st.container()
        with chat_container:
            if len(st.session_state.messages) == 0:
                st.info("👋 안녕하세요! 러닝에 관해 무엇이든 물어보세요.")
            else:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
        
        # 사용자 입력 영역
        user_input = st.chat_input("메시지를 입력하세요...", key="chat_input")
        
        if user_input:
            # 사용자 메시지 추가
            st.session_state.messages.append({
                "role": "user", 
                "content": user_input
            })
            
            # API 호출 중 로딩 표시
            with st.spinner("AI가 생각 중입니다..."):
                # API 호출
                bot_response = call_api(user_input)
            
            # 봇 응답 추가
            st.session_state.messages.append({
                "role": "assistant", 
                "content": bot_response
            })
            
            # 화면 새로고침
            st.rerun()
        
        # 채팅 초기화 버튼
        if len(st.session_state.messages) > 0:
            if st.button("🗑️ 대화 초기화", key="clear_chat"):
                st.session_state.messages = []
# # ==================== 메인 페이지 ====================
# st.title("🏃‍♂️ 러너스클럽에 오신 것을 환영합니다!")

# st.write("""
# ## 메인 콘텐츠
# 여기는 메인 페이지입니다. 
# 우측 하단의 챗봇 버튼을 클릭하여 AI와 대화해보세요!
# """)

# # 샘플 콘텐츠
# col1, col2, col3 = st.columns(3)
# with col1:
#     st.metric("총 러닝 거리", "125 km", "+5 km")
# with col2:
#     st.metric("평균 속도", "6:30 min/km", "-0:15")
# with col3:
#     st.metric("이번 주 목표", "75%", "+10%")

# st.subheader("최근 활동")
# st.write("- 2025-10-30: 10km 러닝")
# st.write("- 2025-10-28: 5km 러닝")
# st.write("- 2025-10-26: 15km 러닝")

# ==================== 플로팅 챗봇 ====================
                st.rerun()