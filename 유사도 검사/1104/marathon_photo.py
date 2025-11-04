"""
마라톤 사진 검색 플랫폼
대회 선택 → 사진 업로드 → 새 화면에서 코스 지도 + 유사 사진 표시
"""

import streamlit as st
from PIL import Image
import gpxpy
import folium
from streamlit_folium import folium_static

# ==========================================
# GPX지도 설정
# ==========================================

def load_marathon_course(tournament_name):
    """
    대회 이름에 따라 GPX 파일 로드
    """
    gpx_files = {
        "JTBC 마라톤": "../data/2025_JTBC.gpx",
        "춘천 마라톤": "../data/chuncheon_marathon.gpx",
    }
    
    if tournament_name in gpx_files:
        try:
            with open(gpx_files[tournament_name], 'r') as f:
                gpx = gpxpy.parse(f)
            
            coordinates = []
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        coordinates.append([point.latitude, point.longitude])
            
            return coordinates
        except FileNotFoundError:
            return None
    return None

def create_course_map(coordinates, photo_locations=None):
    """
    코스 지도 + 사진 위치 표시
    """
    if not coordinates:
        return None
    
    # 중심점 계산
    center_lat = sum([c[0] for c in coordinates]) / len(coordinates)
    center_lon = sum([c[1] for c in coordinates]) / len(coordinates)
    
    # 지도 생성
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles='CartoDB positron'
    )
    
    # 코스 라인
    folium.PolyLine(
        coordinates,
        color='#FF4444',
        weight=5,
        opacity=0.8,
        popup='마라톤 코스'
    ).add_to(m)
    
    # 출발/도착 마커
    folium.Marker(
        coordinates[0],
        popup='🏁 출발',
        icon=folium.Icon(color='green', icon='play')
    ).add_to(m)
    
    folium.Marker(
        coordinates[-1],
        popup='🎯 도착',
        icon=folium.Icon(color='red', icon='stop')
    ).add_to(m)

     # 7단계: 10km마다 거리 마커 추가
    total_points = len(coordinates)
    for km in [10, 20, 21.0975, 30, 40]:
        # 포인트 인덱스 계산 (비율로)
        idx = int((km / 42.195) * total_points)
        if idx < total_points:
            folium.CircleMarker(
                location=coordinates[idx],
                radius=8,
                popup=f'{km}km 지점',
                color='blue',
                fill=True,
                fillColor='lightblue',
                fillOpacity=0.7
            ).add_to(m)
    
    # 사진 위치 표시
    if photo_locations:
        for photo in photo_locations:
            folium.Marker(
                [photo['lat'], photo['lon']],
                popup=folium.Popup(
                    f"""
                    <div style='width: 200px;'>
                        <img src='{photo['thumbnail']}' style='width: 100%;'><br>
                        <b>{photo['name']}</b><br>
                        <small>{photo['distance']:.1f}km 지점</small>
                    </div>
                    """,
                    max_width=220
                ),
                icon=folium.Icon(color='orange', icon='camera')
            ).add_to(m)
    
    return m

# ==========================================
# 페이지 설정
# ==========================================
st.set_page_config(
    page_title="마라톤 사진 검색",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 간결한 CSS 스타일
# ==========================================
st.markdown("""
<style>
    /* 전체 배경 */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%);
    }
    
    /* 드롭다운 스타일 */
    .stSelectbox {
        font-size: 18px;
    }
    
    /* 버튼 스타일 */
    .stButton>button {
        background: linear-gradient(90deg, #4a90e2 0%, #50e3c2 100%);
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 15px 30px;
        border-radius: 12px;
        border: none;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(74, 144, 226, 0.4);
    }
    
    /* 업로드 영역 */
    .stFileUploader {
        border: 2px dashed #4a90e2;
        border-radius: 12px;
        padding: 30px;
        background: white;
    }
    
    /* 카드 스타일 */
    .info-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #4a90e2;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    /* 사진 카드 */
    .photo-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #e0e7ff;
        text-align: center;
        transition: all 0.3s;
        cursor: pointer;
    }
    
    .photo-card:hover {
        transform: scale(1.05);
        border-color: #4a90e2;
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.3);
    }
    
    /* 제목 */
    h1 {
        color: #2c3e50;
        text-align: center;
        font-size: 48px;
        margin-bottom: 30px;
    }
    
    h2 {
        color: #34495e;
        font-size: 28px;
    }
    
    h3 {
        color: #4a90e2;
        font-size: 22px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 세션 스테이트 초기화
# ==========================================
if 'selected_tournament' not in st.session_state:
    st.session_state.selected_tournament = None

if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

if 'show_results' not in st.session_state:
    st.session_state.show_results = False

# ==========================================
# 대회 데이터
# ==========================================
tournaments = {
    "JTBC 마라톤": {
        "date": "2025년 11월 2일",
        "distance": "42.195km",
        "participants": "30,000명",
        "course": "잠실종합운동장 → 광화문 → 남산 → 한강 → 잠실",
        "icon": "🏃‍♂️",
        "color": "#FF6B6B"
    },
    "춘천 마라톤": {
        "date": "2025년 10월 26일",
        "distance": "42.195km",
        "participants": "15,000명",
        "course": "의암호 → 소양강 → 춘천시가지 → 의암호",
        "icon": "🏔️",
        "color": "#4ECDC4"
    }
    # },
    # "제주 국제 마라톤": {
    #     "date": "2024년 11월 5일",
    #     "distance": "42.195km",
    #     "participants": "12,000명",
    #     "course": "제주시 → 애월 → 한림 → 제주시",
    #     "icon": "🌊",
    #     "color": "#45B7D1"
    # },
    # "부산 국제 마라톤": {
    #     "date": "2024년 4월 14일",
    #     "distance": "42.195km",
    #     "participants": "25,000명",
    #     "course": "광안리 → 해운대 → 마린시티 → 광안리",
    #     "icon": "🌉",
    #     "color": "#FFA07A"
    # }
}

# ==========================================
# 페이지 1: 대회 선택 및 사진 업로드
# ==========================================
if not st.session_state.show_results:
    
    # 타이틀
    st.title("🏃 High 러너스 🏃")
    st.caption("AI가 마라톤 코스에서 당신의 사진을 찾아드립니다")
    st.markdown("---")
    
    # 중앙 정렬 레이아웃
    col1, col2, col3 = st.columns([1, 80, 1])
    
    with col2:
        # 1단계: 대회 선택
        st.markdown("### 1️⃣ 대회 선택")
        selected = st.selectbox(
            "참가한 마라톤 대회를 선택하세요",
            options=["대회를 선택해주세요"] + list(tournaments.keys()),
            key="tournament_selectbox"
        )
        
        # 대회가 선택되면 세션에 저장
        if selected != "대회를 선택해주세요":
            st.session_state.selected_tournament = selected
            
            # # 선택된 대회 정보 표시
            # info = tournaments[selected]
            # st.markdown(f"""
            # <div class="info-card">
            #     <h3>{info['icon']} {selected}</h3>
            #     <p style='margin: 5px 0; color: #666;'>
            #         📅 <b>일시:</b> {info['date']}<br>
            #         📏 <b>거리:</b> {info['distance']}<br>
            #         👥 <b>참가자:</b> {info['participants']}<br>
            #         📍 <b>코스:</b> {info['course']}
            #     </p>
            # </div>
            # """, unsafe_allow_html=True)
            
            # st.markdown("---")
            
            # 2단계: 사진 업로드
            st.markdown("### 2️⃣ 사진 업로드")
            uploaded_file = st.file_uploader(
                "Drag and drop file here",
                type=['png', 'jpg', 'jpeg'],
                key="photo_uploader",
                help="마라톤 사진을 업로드하세요 (최대 200MB)"
            )
            
            # 사진이 업로드되면
            if uploaded_file:
                # 이미지 읽기 및 세션에 저장
                image = Image.open(uploaded_file)
                st.session_state.uploaded_image = image
                
                # # 미리보기 표시
                # st.success(f"✅ {uploaded_file.name} 업로드 완료!")
                # st.image(image, caption="업로드된 사진", use_container_width=True)
                
                # st.markdown("---")
                
                # 검색 버튼
                if st.button("🔍 코스 및 추천 사진 보기", type="primary"):
                    st.session_state.show_results = True
                    st.rerun()
        
        else:
            st.info("👆 위에서 대회를 먼저 선택해주세요")

# ==========================================
# 페이지 2: 코스 지도 + 유사 사진
# ==========================================
else:
    # 선택된 대회 정보 가져오기
    tournament_name = st.session_state.selected_tournament
    tournament_info = tournaments[tournament_name]
    
    # 상단 헤더
    st.markdown(f"""
    <div style='text-align: center; padding: 20px; background: white; border-radius: 12px; margin-bottom: 30px;'>
        <h1 style='margin: 0; font-size: 36px;'>{tournament_info['icon']} {tournament_name}</h1>
        
    </div>
    """, unsafe_allow_html=True)
    
    # 좌우 분할 (6:4 비율)
    left_col, right_col = st.columns([6, 4])
    
    # ==========================================
    # 왼쪽: 코스 지도 영역
    # ==========================================
    with left_col:
        st.markdown("### 🗺️ 마라톤 코스")
        
        # 대회 정보 카드
        st.markdown(f"""
        <div class="info-card">
            <p style='margin: 0; line-height: 1.8;'>
                📅 <b>일시:</b> {tournament_info['date']}<br>
                📏 <b>거리:</b> {tournament_info['distance']}<br>
                📍 <b>코스:</b> {tournament_info['course']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # 코스 지도 영역 (플레이스홀더)
        coordinates = load_marathon_course(tournament_name)
    
    if coordinates:
        st.success(f"✅ {tournament_name} 코스를 불러왔습니다!")
        
        # 지도 생성 및 표시
        m = create_course_map(coordinates)
        
        if m:
            folium_static(m, width=600, height=400)
    else:
        st.error("❌ 코스 데이터를 찾을 수 없습니다.")


        # st.markdown("""
        # <div style='background: white; border-radius: 12px; padding: 40px; text-align: center; min-height: 500px; border: 2px solid #e0e7ff;'>
        #     <div style='padding-top: 100px;'>
        #         <h2 style='color: #4a90e2; font-size: 64px; margin-bottom: 20px;'>🗺️</h2>
        #         <h3 style='color: #666;'>마라톤 코스 지도</h3>
        #         <p style='color: #999; margin-top: 20px;'>실제 구현시 Google Maps API 또는 Folium 사용</p>
        #         <br><br>
        #         <div style='display: flex; justify-content: space-around; margin-top: 60px;'>
        #             <div>
        #                 <div style='width: 80px; height: 80px; background: #e8f5e9; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 36px;'>🏁</div>
        #                 <p style='margin-top: 15px; color: #666; font-weight: bold;'>출발</p>
        #             </div>
        #             <div>
        #                 <div style='width: 80px; height: 80px; background: #fff3e0; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 36px;'>📸</div>
        #                 <p style='margin-top: 15px; color: #666; font-weight: bold;'>중간</p>
        #             </div>
        #             <div>
        #                 <div style='width: 80px; height: 80px; background: #fce4ec; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 36px;'>🎯</div>
        #                 <p style='margin-top: 15px; color: #666; font-weight: bold;'>도착</p>
        #             </div>
        #         </div>
        #     </div>
        # </div>
        # """, unsafe_allow_html=True)
    
    # ==========================================
    # 오른쪽: 유사한 사진들
    # ==========================================
    with right_col:
        # st.markdown("### 📍 코스 상 유사한 사진들")
        
        # 업로드한 사진 미리보기
        if st.session_state.uploaded_image:
            st.markdown("#### 🖼️ 검색한 사진")
            st.image(st.session_state.uploaded_image, use_container_width=True)
            # st.markdown("---")
        
    #     # AI 추천 사진
    #     st.success("✨ AI가 찾은 유사한 사진 5장")
        
    #     # # 5장의 추천 사진 (2열로 배치)
    #     # for i in range(5):
    #     #     st.markdown(f"""
    #     #     <div class="photo-card">
    #     #         <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    #     #                     height: 200px; 
    #     #                     border-radius: 8px; 
    #     #                     display: flex; 
    #     #                     align-items: center; 
    #     #                     justify-content: center; 
    #     #                     margin-bottom: 12px;'>
    #     #             <span style='font-size: 64px;'>🖼️</span>
    #     #         </div>
    #     #         <div style='text-align: left; padding: 5px;'>
    #     #             <p style='margin: 0; font-size: 16px; font-weight: bold; color: #2c3e50;'>
    #     #                 📍 {i*8 + 5}km 지점
    #     #             </p>
    #     #             <p style='margin: 5px 0 0 0; font-size: 14px; color: #4a90e2;'>
    #     #                 유사도: {95 - i*2}%
    #     #             </p>
    #     #         </div>
    #     #     </div>
    #     #     """, unsafe_allow_html=True)
            
    #     #     st.markdown("<br>", unsafe_allow_html=True)
    
    # # ==========================================
    # # 하단: 뒤로 가기 버튼
    # # ==========================================
    # st.markdown("---")
    
    col1, col2, col3 = st.columns([5, 10, 5])
    with col2:
        if st.button("◀️ 처음으로 돌아가기", use_container_width=True):
            # 세션 초기화
            st.session_state.show_results = False
            st.session_state.selected_tournament = None
            st.session_state.uploaded_image = None
            st.rerun()

# ==========================================
# 하단 푸터
# ==========================================
st.markdown("---")
st.caption("💡 Tip: 정확한 검색을 위해 선명한 사진을 업로드해주세요")