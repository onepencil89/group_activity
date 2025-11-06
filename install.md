# group_activity

챗봇 기능 추가

앞단 import 및 def
```
# OPENAI API 

from dotenv import load_dotenv
from openai import OpenAI

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
```
실행명령어

```
    st.markdown("---")
    col1, col2, col3 = st.columns([5, 10, 5])
    with col2:
        if st.button("달리기에 관해 무엇이든 물어보세요!😎", key="chatbot_btn", use_container_width=True):
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
                st.rerun()
    


    col1, col2, col3 = st.columns([5, 10, 5])
    with col2:
        if st.button("◀️ 처음으로 돌아가기", use_container_width=True):
            # 세션 초기화
            st.session_state.show_results = False
            st.session_state.selected_tournament = None
            st.session_state.uploaded_image = None
            st.rerun()
```
