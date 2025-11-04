import streamlit as st
import openai
from datetime import datetime

from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv(override=True)

# 환경변수 불러오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from openai import OpenAI
# openai api 인증 (환경 변수 사용)

if OPENAI_API_KEY:
    try:
        # Global client 객체 생성
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        # 키가 있지만 문제가 있을 경우 (예: 잘못된 형식)
        st.error(f"⚠️ OpenAI 클라이언트 초기화 오류: {e}. API 키를 확인해주세요.")
        client = None
else:
    # 키가 아예 없을 경우
    client = None
    # st.warning("⚠️ OpenAI API Key가 환경 변수(OPENAI_API_KEY)에 설정되지 않았습니다.")


# 페이지 설정
st.set_page_config(
    page_title="러닝 가이드",
    page_icon="🏃",
    layout="wide"
)

# 세션 스테이트 초기화
if 'selected_category' not in st.session_state:
    st.session_state.selected_category = None
if 'selected_subcategory' not in st.session_state:
    st.session_state.selected_subcategory = None
if 'chat_mode' not in st.session_state:
    st.session_state.chat_mode = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'initial_question' not in st.session_state:
    st.session_state.initial_question = ""


# 카테고리 및 하위 카테고리 데이터
categories = {
    "🏃 초보자 가이드": [
        "러닝 시작하기",
        "올바른 자세",
        "준비운동",
        "첫 주 계획"
    ],
    "👟 장비 & 용품": [
        "러닝화 선택법",
        "러닝 의류",
        "필수 액세서리",
        "계절별 장비"
    ],
    "📊 트레이닝 계획": [
        "5km 준비",
        "10km 준비",
        "하프 마라톤",
        "풀 마라톤"
    ],
    "💪 컨디셔닝": [
        "근력 운동",
        "스트레칭",
        "회복 방법",
        "부상 예방"
    ],
    "🍎 영양 & 식단": [
        "러닝 전 식사",
        "러닝 후 식사",
        "수분 보충",
        "보충제 가이드"
    ],
    "🏆 대회 & 이벤트": [
        "대회 준비",
        "레이스 전략",
        "대회 찾기",
        "기록 관리"
    ]
}

# 챗봇 응답 생성 함수
def get_chatbot_response(user_message, context=""):
    # 전역 클라이언트 객체 사용
    if not client:
        return "⚠️ OpenAI API 키가 설정되지 않았거나 초기화에 실패했습니다. `.env` 파일을 확인해주세요."
    
    try:
        system_message = f"""당신은 러닝 전문가입니다. 사용자의 러닝 관련 질문에 친절하고 상세하게 답변해주세요.
        답변은 한국어로 제공하며, 초보자도 이해할 수 있도록 쉽게 설명해주세요.
        {f'현재 주제: {context}' if context else ''}"""
        
        messages = [{"role": "system", "content": system_message}]
        
        # 대화 히스토리 추가 (최근 5개만)
        for msg in st.session_state.chat_history[-5:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        messages.append({"role": "user", "content": user_message})
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"⚠️ 오류가 발생했습니다: {str(e)}\n\nAPI 호출에 문제가 없는지 확인해주세요."

# 챗봇 모드
if st.session_state.chat_mode:
    # 사이드바
    with st.sidebar:
        st.markdown("### ⚙️ 설정")
        
        # Home으로 돌아가기
        if st.button("🏠 Home 화면 돌아가기", use_container_width=True, type="primary"):
            st.session_state.chat_mode = False
            st.session_state.chat_history = []
            st.session_state.initial_question = ""
            st.rerun()
        
        st.markdown("---")
        
        # 대화 기록 요약
        if st.session_state.chat_history:
            st.markdown("### 📝 대화 기록")
            st.caption(f"총 {len(st.session_state.chat_history)//2}개의 질문")
    
    # 메인 챗봇 화면
    
    # R3: '다른 질문하기' 버튼을 헤더 영역에 배치
    col_title, col_reset = st.columns([4, 1])
    
    with col_title:
        st.markdown("<h1 style='text-align: left;'>🏃 러닝 가이드 챗봇</h1>", unsafe_allow_html=True)
    
    with col_reset:
        st.markdown("<br>", unsafe_allow_html=True) # 제목과 높이 맞추기 위한 공백
        if st.button("🔄 다른 질문하기", use_container_width=True, help="새로운 대화를 시작합니다"):
            st.session_state.chat_history = []
            st.session_state.initial_question = ""
            st.rerun()
    
    st.markdown("<p style='text-align: left; color: gray;'>러닝에 대해 궁금한 점을 물어보세요</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # 채팅 컨테이너
    chat_container = st.container(height=500) # 고정 높이 설정으로 스크롤 용이하게
    
    with chat_container:
        # 대화 내역 표시
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    with st.chat_message("user", avatar="🧑"):
                        st.write(message["content"])
                        st.caption(message["time"])
                else:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.write(message["content"])
                        st.caption(message["time"])
        else:
            # 첫 화면 안내 메시지
            st.info("💡 러닝에 관해 무엇이든 물어보세요!")
    
    # 입력창
    st.markdown("---")
    
    # 초기 질문이 있으면 자동으로 처리
    if st.session_state.initial_question and not st.session_state.chat_history:
        user_input = st.session_state.initial_question
        st.session_state.initial_question = ""
        
        # 사용자 메시지 추가
        current_time = datetime.now().strftime("%H:%M")
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "time": current_time
        })
        
        # 봇 응답 생성
        with st.spinner("답변을 생성하는 중..."):
            bot_response = get_chatbot_response(user_input)
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": bot_response,
            "time": current_time
        })
        
        st.rerun()
    
    # 일반 입력
    user_input = st.chat_input("메시지를 입력하세요...")
    
    if user_input:
        # 사용자 메시지 추가
        current_time = datetime.now().strftime("%H:%M")
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "time": current_time
        })
        
        # 봇 응답 생성
        with st.spinner("답변을 생성하는 중..."):
            bot_response = get_chatbot_response(user_input)
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": bot_response,
            "time": current_time
        })
        
        st.rerun()

# 메인 페이지
elif st.session_state.selected_category is None:
    # 헤더
    st.markdown("<h1 style='text-align: center;'>🏃 러닝 가이드</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # 검색창
    search_query = st.text_input(
        "",
        placeholder="러닝에 대해 궁금한 점을 검색해보세요... (예: 러닝화 추천, 5km 훈련법)",
        key="search_box"
    )
    
    if search_query:
        st.session_state.chat_mode = True
        st.session_state.initial_question = search_query
        st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 추천 질문 버튼들
    st.subheader("💡 추천 질문")
    col1, col2, col3, col4 = st.columns(4)
    
    # 추천 질문 내용 수정 (원래 코드의 오타/비논리적 질문 수정)
    recommended_questions = [
        "초보자 러닝 시작 방법",
        "러닝화 추천",
        "부상 예방 스트레칭",
        "마라톤 식단"
    ]
    
    cols = [col1, col2, col3, col4]
    for idx, question in enumerate(recommended_questions):
        with cols[idx]:
            if st.button(question, use_container_width=True):
                st.session_state.chat_mode = True
                st.session_state.initial_question = question
                st.rerun()
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # 카테고리 그리드
    st.subheader("📚 카테고리")
    
    # 2열로 카테고리 배치
    cols = st.columns(2)
    
    for idx, (category, subcategories) in enumerate(categories.items()):
        col = cols[idx % 2]
        
        with col:
            if st.button(
                category,
                key=f"cat_{idx}",
                use_container_width=True,
                type="primary"
            ):
                st.session_state.selected_category = category
                st.session_state.selected_subcategory = None
                st.rerun()
            
            st.markdown("<br>", unsafe_allow_html=True)

# 카테고리 상세 페이지
else:
    # 사이드바에 하위 카테고리 표시
    with st.sidebar:
        st.title("📋 하위 카테고리")
        
        st.markdown("---")
        
        subcategories = categories[st.session_state.selected_category]
        
        for subcategory in subcategories:
            if st.button(
                subcategory,
                key=f"subcat_{subcategory}",
                use_container_width=True,
                type="secondary" if st.session_state.selected_subcategory != subcategory else "primary"
            ):
                st.session_state.selected_subcategory = subcategory
                st.rerun()
    
    # 메인 컨텐츠 영역 - 제목과 HOME 버튼을 같은 줄에 배치
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title(st.session_state.selected_category)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🏠 HOME", key="home_btn", type="secondary"):
            st.session_state.selected_category = None
            st.session_state.selected_subcategory = None
            st.rerun()
    
    st.markdown("---")
    
    if st.session_state.selected_subcategory:
        st.header(f"📖 {st.session_state.selected_subcategory}")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 예시 컨텐츠
        st.write(f"**{st.session_state.selected_subcategory}**에 대한 상세 가이드가 여기에 표시됩니다.")
        
        # AI 챗봇 질문하기 버튼
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("💬 이 주제에 대해 AI에게 질문하기", use_container_width=True, type="primary"):
                st.session_state.chat_mode = True
                st.session_state.initial_question = f"{st.session_state.selected_subcategory}에 대해 알려주세요"
                st.rerun()
        
        # 예시 컨텐츠
        st.markdown("""
        ### 주요 내용
        
        이 섹션에서는 다음과 같은 내용을 다룹니다:
        - 기본 개념 및 중요성
        - 실전 팁과 노하우
        - 자주 하는 실수
        - 전문가 조언
        
        ### 관련 자료
        - 참고 영상
        - 추천 글
        - 커뮤니티 토론
        """)
        
    else:
        st.info("👈 왼쪽 사이드바에서 하위 카테고리를 선택해주세요.")
        
        # 카테고리 개요
        st.markdown("### 이 카테고리에서는...")
        
        cols = st.columns(2)
        subcategories = categories[st.session_state.selected_category]
        
        for idx, subcat in enumerate(subcategories):
            with cols[idx % 2]:
                st.markdown(f"**{subcat}**")
                st.write("이 주제에 대한 상세한 가이드를 제공합니다.")
                st.markdown("<br>", unsafe_allow_html=True)


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

# # 커스텀 CSS
# st.markdown("""
# <style>
#     /* 챗봇 토글 버튼을 우측 하단에 고정 */
#     .stButton button[kind="secondary"] {
#         position: fixed;
#         bottom: 20px;
#         right: 20px;
#         width: 60px;
#         height: 60px;
#         border-radius: 50%;
#         font-size: 24px;
#         z-index: 1000;
#         box-shadow: 0 4px 12px rgba(0,0,0,0.3);
#     }
    
#     /* 챗봇 컨테이너 스타일 */
#     .chatbot-box {
#         position: fixed;
#         bottom: 90px;
#         right: 20px;
#         width: 400px;
#         max-height: 600px;
#         background: white;
#         border-radius: 15px;
#         box-shadow: 0 8px 24px rgba(0,0,0,0.15);
#         z-index: 999;
#         padding: 20px;
#         overflow-y: auto;
#     }
# </style>
# """, unsafe_allow_html=True)


# # 챗봇 토글 버튼 (우측 하단 고정)
# # 빈 컬럼을 사용하여 우측에 배치
# cols = st.columns([10, 1])
# with cols[1]:
#     if st.button("💬", key="chatbot_btn", type="secondary"):
#         st.session_state.chat_open = not st.session_state.chat_open

# # 챗봇이 열려있을 때
# if st.session_state.chat_open:
#     # 플로팅 박스처럼 보이게 하기
#     with st.container():
#         st.markdown("---")
        
#         # 챗봇 헤더
#         header_col1, header_col2 = st.columns([4, 1])
#         with header_col1:
#             st.markdown("### 💬 AI 챗봇")
#         with header_col2:
#             if st.button("✕", key="close_chat"):
#                 st.session_state.chat_open = False
#                 st.rerun()
        
#         st.caption("무엇을 도와드릴까요?")
        
#         # 채팅 히스토리 표시 영역
#         chat_container = st.container()
#         with chat_container:
#             if len(st.session_state.messages) == 0:
#                 st.info("👋 안녕하세요! 러닝에 관해 무엇이든 물어보세요.")
#             else:
#                 for message in st.session_state.messages:
#                     with st.chat_message(message["role"]):
#                         st.write(message["content"])
        
#         # 사용자 입력 영역
#         user_input = st.chat_input("메시지를 입력하세요...", key="chat_input")
        
#         if user_input:
#             # 사용자 메시지 추가
#             st.session_state.messages.append({
#                 "role": "user", 
#                 "content": user_input
#             })
            
#             # API 호출 중 로딩 표시
#             with st.spinner("AI가 생각 중입니다..."):
#                 # API 호출
#                 bot_response = call_api(user_input)
            
#             # 봇 응답 추가
#             st.session_state.messages.append({
#                 "role": "assistant", 
#                 "content": bot_response
#             })
            
#             # 화면 새로고침
#             st.rerun()
        
#         # 채팅 초기화 버튼
#         if len(st.session_state.messages) > 0:
#             if st.button("🗑️ 대화 초기화", key="clear_chat"):
#                 st.session_state.messages = []
