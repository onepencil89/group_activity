# app.py
import streamlit as st

# 컨테이너
with st.container():
    st.write("이것은 컨테이너입니다")
    st.button("컨테이너 내부 버튼")

# 확장 가능한 섹션

with st.expander("자세히 보기"):
    # 버튼 스타일의 텍스트
    if st.button("숨겨진 내용이 여기 있습니다", key="hidden_button"):
        st.success("클릭되었습니다!")
        # 명령 수행

    # if click:
    #     tab1, tab2, tab3 = st.tabs(["고양이", "강아지", "새"])

    #     with tab1:
    #         st.header("고양이")
    #         st.write("🐱 고양이 관련 내용")

    #     with tab2:
    #         st.header("강아지")
    #         st.write("🐶 강아지 관련 내용")

    #     with tab3:
    #         st.header("송아지")


# # 사이드바에 요소 추가
# st.sidebar.title("사이드바")
# st.sidebar.write("사이드바 내용")

# # 사이드바에 입력 위젯
# option = st.sidebar.selectbox(
#     "옵션을 선택하세요",
#     ["옵션1", "옵션2", "옵션3"]
# )

# # 사이드바 버튼
# if st.sidebar.button("사이드바 버튼"):
#     st.write("사이드바 버튼이 클릭되었습니다")

# tab1, tab2, tab3 = st.tabs(["고양이", "강아지", "새"])

# with tab1:
#     st.header("고양이")
#     st.write("🐱 고양이 관련 내용")

# with tab2:
#     st.header("강아지")
#     st.write("🐶 강아지 관련 내용")

# with tab3:
#     st.header("송아지")
