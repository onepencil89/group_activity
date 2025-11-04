import streamlit as st

# 세션 상태 초기화
if 'show_tabs' not in st.session_state:
    st.session_state.show_tabs = False

st.title("환영합니다 러너스클럽에 오신것을!")

# expander 안에 버튼
with st.expander("궁금한것이 무엇일까요?"):
    if st.button("시간이 없어"):
        st.session_state.show_tabs = True
        # st.session_state.show_tabs = not st.session_state.show_tabs
    
        tab1, tab2, tab3 = st.tabs(["퇴근시간 전", "퇴근시간 후 ", "주말"])
        
        with tab1:
            st.write("30분만 아침에 일찍 일어놔봐")
        
        with tab2:
            st.write("지하철 1~2정거장 전에 내려서 뛰어")
        
        with tab3:
            st.write("뛰기 좋은 날이지")


    if st.button("질문 2"):
        st.session_state.show_tabs = True
    # 버튼 클릭하면 탭 표시
    if st.session_state.show_tabs:
        tab1, tab2, tab3 = st.tabs(["123", "244", "3"])
        
        with tab1:
            st.write("첫 번째 탭 내용")
        
        with tab2:
            st.write("두 번째 탭 내용")
        
        with tab3:
            st.write("세 번째 탭 내용")        
    
    if st.button("질문 3"):
        st.session_state.show_tabs = True
