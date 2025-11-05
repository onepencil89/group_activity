"""
마라톤 사진 검색 플랫폼 - GPX 통합 버전 (최종 통합 버전)
주요 기능: 와이드 레이아웃, 지도 마커 썸네일/툴팁, 클릭 시 목록 숨김 및 상세 보기 전환, 새 창 구매 버튼
"""

import streamlit as st
from PIL import Image
import gpxpy
import folium
from streamlit_folium import folium_static
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import io
from datetime import datetime, timedelta
import random
import base64


# ==========================================
# ⚙️ Streamlit 초기 설정 (와이드 레이아웃 적용)
# ==========================================
st.set_page_config(layout="wide")

# ==========================================
# CLIP 모델 로드 및 캐싱 (모듈 레벨 함수)
# ==========================================
@st.cache_resource
def load_clip_model():
    """모델 로드 및 캐싱 (모듈 레벨 함수)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    return model, processor

# ==========================================
# ImageSimilarityFinder 클래스
# ==========================================
class ImageSimilarityFinder:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def get_image_embedding(self, image):
        """이미지의 임베딩 벡터 생성"""
        if self.model is None or self.processor is None:
            self.model, self.processor = load_clip_model()
        
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        else:
            image = image.convert('RGB')
            
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            embedding = self.model.get_image_features(**inputs)
        
        return embedding.cpu().numpy()

# ==========================================
# 🖼️ 이미지 표시 도우미 함수 (오류 해결 반영)
# ==========================================
def image_bytes_to_st_image(image_bytes, **kwargs):
    """
    이미지 바이트 데이터를 st.image에 안전하게 표시합니다.
    """
    st.image(io.BytesIO(image_bytes), **kwargs)


# ==========================================
# GPX 관련 함수
# ==========================================
def load_marathon_course(tournament_name):
    """대회 이름에 따라 GPX 파일 로드"""
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
            st.error(f"❌ GPX 파일을 찾을 수 없습니다: {gpx_files[tournament_name]}")
            return None
    return None

def assign_photo_locations(num_photos, coordinates, start_time):
    """사진에 GPX 코스 기반 위치와 시간 자동 할당"""
    if not coordinates or len(coordinates) == 0:
        return []
    
    total_points = len(coordinates)
    photo_locations = []
    
    for i in range(num_photos):
        idx = int((i / num_photos) * total_points)
        if idx >= total_points:
            idx = total_points - 1
        
        lat, lon = coordinates[idx]
        km = (idx / total_points) * 42.195
        minutes_elapsed = int(km * 6)
        photo_time = start_time + timedelta(minutes=minutes_elapsed)
        
        photo_locations.append({
            'lat': lat,
            'lon': lon,
            'km': round(km, 2),
            'time': photo_time.strftime("%Y-%m-%d %H:%M:%S"),
            'idx': idx
        })
    
    return photo_locations

def create_clickable_course_map(coordinates, photo_data=None):
    """
    클릭 가능한 GPX 코스 지도 생성
    - 코스 라인 표시
    - 출발/도착 마커
    - 이미 할당된 사진 위치 마커
    """
    if not coordinates:
        return None
    
    center_lat = sum([c[0] for c in coordinates]) / len(coordinates)
    center_lon = sum([c[1] for c in coordinates]) / len(coordinates)
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
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
        icon=folium.Icon(color='green', icon='play', prefix='fa')
    ).add_to(m)
    
    folium.Marker(
        coordinates[-1],
        popup='🎯 도착',
        icon=folium.Icon(color='red', icon='stop', prefix='fa')
    ).add_to(m)
    
    # km 지점 마커
    total_points = len(coordinates)
    for km in [10, 20, 21.0975, 30, 40]:
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
    
    # 이미 할당된 사진 위치에 마커 표시
    if photo_data:
        for photo_name, data in photo_data.items():
            folium.Marker(
                location=[data['lat'], data['lon']],
                popup=f"📷 {photo_name}",
                icon=folium.Icon(color='purple', icon='camera', prefix='fa')
            ).add_to(m)
    
    # 클릭 이벤트 활성화
    m.add_child(folium.LatLngPopup())
    
    return m

def create_course_map_with_photos(coordinates, photo_markers=None):
    """
    GPX 코스 지도 + 사진 마커 생성 
    (썸네일 마커, 툴팁 미리보기+풀스크린, 팝업 상세 보기 이동 버튼 포함)
    """
    if not coordinates:
        return None
    
    center_lat = sum([c[0] for c in coordinates]) / len(coordinates)
    center_lon = sum([c[1] for c in coordinates]) / len(coordinates)
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='CartoDB positron'
    )
    
    # 코스 라인 및 km 마커 (생략)
    folium.PolyLine(coordinates, color='#FF4444', weight=5, opacity=0.8, popup='마라톤 코스').add_to(m)
    folium.Marker(coordinates[0], popup='🏁 출발', icon=folium.Icon(color='green', icon='play', prefix='fa')).add_to(m)
    folium.Marker(coordinates[-1], popup='🎯 도착', icon=folium.Icon(color='red', icon='stop', prefix='fa')).add_to(m)
    total_points = len(coordinates)
    for km in [10, 20, 21.0975, 30, 40]:
        idx = int((km / 42.195) * total_points)
        if idx < total_points:
            folium.CircleMarker(location=coordinates[idx], radius=8, popup=f'{km}km 지점', color='blue', fill=True, fillColor='lightblue', fillOpacity=0.7).add_to(m)

    # 사진 마커 추가
    if photo_markers:
        for photo in photo_markers:
            img_base64 = photo.get('image_base64', '')
            similarity_percent = photo['similarity']
            photo_unique_id = f"{photo['tournament']}_{photo['name']}"

            # 유사도에 따른 테두리 색상 및 두께 설정
            if similarity_percent >= 90:
                border_style = '4px solid #FF0000' # 빨간색 강조
                marker_color = 'red'
            elif similarity_percent >= 80:
                border_style = '2px solid #FFA500' # 주황색 강조
                marker_color = 'orange'
            else:
                border_style = '1px solid #4a90e2' # 일반 파란색
                marker_color = 'blue'
            
            # 커스텀 HTML 아이콘 (Base64 썸네일 이미지)
            icon_html = f"""
            <div style="
                width: 30px; height: 30px; 
                border-radius: 50%; 
                overflow: hidden; 
                border: {border_style};
                box-shadow: 0 0 5px rgba(0,0,0,0.4);
                background-image: url('data:image/png;base64,{img_base64}');
                background-size: cover;
                background-position: center;
                cursor: pointer;
            "></div>
            """
            
            # HTML 마커 생성 (folium.DivIcon 사용)
            custom_icon = folium.DivIcon(
                icon_size=(30, 30),
                icon_anchor=(15, 15),
                html=icon_html 
            )

            # ツールチップ HTML (미리보기 + 풀스크린 기능)
            tooltip_image_html = f"""
            <div style='width: 150px; font-family: Arial; text-align: center; user-select: none;'>
                <img src='data:image/png;base64,{img_base64}' 
                     onclick="window.open('data:image/png;base64,{img_base64}', '_blank', 'fullscreen=yes');"
                     style='width: 100%; border-radius: 8px; border: {border_style}; cursor: pointer; margin-bottom: 5px;'>
                <div style='font-size: 12px; color: #333;'>
                    <b>{photo['name']}</b><br>
                    {photo['km']}km | <span style='color: {marker_color}; font-weight: bold;'>{similarity_percent:.1f}%</span>
                </div>
            </div>
            """
            
            # 팝업 HTML (상세 보기 버튼 포함 -> Session State 변경 트리거)
            popup_html = f"""
            <div style='width: 250px; font-family: Arial;'>
                <img src='data:image/png;base64,{img_base64}'  
                      style='width: 100%; border-radius: 8px; margin-bottom: 10px; border: {border_style};'>
                <div style='background: #f0f7ff; padding: 10px; border-radius: 8px;'>
                    <b style='color: #2c3e50; font-size: 16px;'>📸 {photo['name']}</b><br>
                    <hr style='margin: 8px 0; border: none; border-top: 1px solid #ddd;'>
                    <small style='color: #666;'>
                        📍 <b>위치:</b> {photo['km']}km 지점<br>
                        📅 <b>시간:</b> {photo['time']}<br>
                        🎯 <b>유사도:</b> <span style='color: {marker_color}; font-weight: bold;'>{similarity_percent:.1f}%</span><br>
                        👤 <b>촬영자:</b> {photo.get('photographer', '작가')}
                    </small>
                    <button id='detail_btn_{photo_unique_id}' 
                            onclick="window.parent.postMessage({{
                                type: 'streamlit:setSessionState', 
                                key: 'detailed_photo_id', 
                                value: '{photo_unique_id}'
                            }}, '*'); window.parent.postMessage({{type: 'streamlit:rerun'}}, '*')"
                            style='background-color: #4a90e2; color: white; border: none; padding: 10px; border-radius: 5px; width: 100%; cursor: pointer; margin-top: 10px;'>
                            🔍 상세 보기 및 구매
                    </button>
                </div>
            </div>
            """

            folium.Marker(
                [photo['lat'], photo['lon']],
                popup=folium.Popup(popup_html, max_width=270),
                icon=custom_icon,
                tooltip=folium.Tooltip(tooltip_image_html, max_width=200) 
            ).add_to(m)
            
    return m

# ==========================================
# 세션 스테이트 초기화 함수
# ==========================================
def initialize_session_state():
    """Streamlit 세션 상태를 초기화합니다."""
    if 'saved_photos' not in st.session_state:
        st.session_state.saved_photos = []
    if 'image_finder' not in st.session_state:
        st.session_state.image_finder = ImageSimilarityFinder()
    if 'selected_tournament' not in st.session_state:
        st.session_state.selected_tournament = None
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    if 'detailed_photo_id' not in st.session_state:
        st.session_state.detailed_photo_id = None
    # 리스트 클릭 시 상세 보기 모드 제어 변수
    if 'selected_similar_photo_id' not in st.session_state:
        st.session_state.selected_similar_photo_id = None
    if 'show_detail_view' not in st.session_state:
        st.session_state.show_detail_view = False


# ==========================================
# 세션 스테이트 초기화
# ==========================================
initialize_session_state()

# ==========================================
# CSS 스타일
# ==========================================
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%); }
    /* 버튼 스타일 유지 */
    .stButton>button {
        background: linear-gradient(90deg, #4a90e2 0%, #50e3c2 100%);
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 15px 30px;
        border-radius: 12px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(74, 144, 226, 0.4);
    }
    /* 구매 버튼 스타일 */
    .purchase-btn-style {
        background-color: #e35050; 
        color: white; 
        border: none; 
        padding: 10px; 
        border-radius: 5px; 
        width: 100%; 
        font-weight: bold; 
        cursor: pointer; 
        height: 50px;
        text-align: center;
        display: block;
        line-height: 30px; /* 버튼 텍스트 중앙 정렬 */
        text-decoration: none;
    }

    /* Full Screen 팝업 지원 CSS */
    @media all and (display-mode: fullscreen) {
        .leaflet-popup-content img {
            max-width: 100%;
            max-height: 100vh;
        }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 대회 데이터
# ==========================================
tournaments = {
    "JTBC 마라톤": {
        "date": "2025년 11월 2일",
        "start_time": "08:00:00",
        "distance": "42.195km",
        "icon": "🏃‍♂️"
    },
    "춘천 마라톤": {
        "date": "2025년 10월 26일",
        "start_time": "09:00:00",
        "distance": "42.195km",
        "icon": "🏔️"
    }
}

# ==========================================
# 사이드바: 모드 선택
# ==========================================
mode = st.sidebar.radio(
    "모드 선택",
    ["📸 작가 모드", "🔍 이용자 모드"],
    label_visibility="collapsed"
)

# ==========================================
# 작가 모드
# ==========================================
if mode == "📸 작가 모드":
    st.title("📸 작가 모드")
    st.markdown("---")
    
    selected_tournament = st.selectbox(
        "대회 선택",
        options=list(tournaments.keys())
    )
    
    # GPX 코스 로드
    if selected_tournament:
        coordinates = load_marathon_course(selected_tournament)
        
        if coordinates:
            st.subheader("📍 코스 지도")
            st.info("💡 지도를 클릭하면 해당 위치의 좌표가 선택된 사진에 할당됩니다")
            
            # 사진 업로드
            uploaded_files = st.file_uploader(
                "사진 업로드",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                # 세션 상태 초기화
                if 'photo_data' not in st.session_state:
                    st.session_state.photo_data = {}
                
                # 선택할 사진 리스트
                photo_names = [f.name for f in uploaded_files]
                selected_photo = st.selectbox(
                    "위치를 할당할 사진 선택",
                    options=photo_names
                )
                
                # 클릭 가능한 지도 생성
                m = create_clickable_course_map(coordinates, st.session_state)
                
                # 지도 표시 및 클릭 이벤트 처리
                map_data = folium_static(m, width=800, height=600)
                
                # 지도 클릭 시 좌표 추출 및 사진에 할당
                if map_data and map_data.get('last_clicked'):
                    clicked_lat = map_data['last_clicked']['lat']
                    clicked_lon = map_data['last_clicked']['lng']
                    
                    # 선택된 사진에 좌표 할당
                    st.session_state.photo_data[selected_photo] = {
                        'lat': clicked_lat,
                        'lon': clicked_lon,
                        'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    st.success(f"✅ {selected_photo}에 위치 할당 완료: ({clicked_lat:.6f}, {clicked_lon:.6f})")
                    st.rerun()
                
                # 할당된 위치 정보 표시
                if st.session_state.photo_data:
                    st.subheader("📋 할당된 위치 정보")
                    for photo_name, data in st.session_state.photo_data.items():
                        st.write(f"**{photo_name}**: 위도 {data['lat']:.6f}, 경도 {data['lon']:.6f}")



    
    
    
    # st.info("💡 8장의 사진을 업로드하면 코스 전체에 균등하게 배치됩니다")
    uploaded_files = st.file_uploader(
        "사진을 선택하세요",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        key="photographer_upload"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)}장의 사진이 업로드되었습니다! (AI 처리 대기 중)") 
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("💾 DB에 저장하기", type="primary"):
                coordinates = load_marathon_course(selected_tournament)
                
                if not coordinates:
                    st.error("❌ GPX 파일을 로드할 수 없습니다.")
                else:
                    start_datetime = datetime.strptime(
                        f"{tournaments[selected_tournament]['date']} {tournaments[selected_tournament]['start_time']}",
                        "%Y년 %m월 %d일 %H:%M:%S"
                    )
                    
                    photo_locations = assign_photo_locations(
                        len(uploaded_files[:8]),
                        coordinates,
                        start_datetime
                    )
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, (file, location) in enumerate(zip(uploaded_files[:8], photo_locations)):
                        status_text.text(f"🤖 AI 처리 중... ({idx+1}/{len(uploaded_files[:8])})")
                        
                        try:
                            image = Image.open(file)
                            embedding = st.session_state.image_finder.get_image_embedding(image)
                            
                            img_byte_arr = io.BytesIO()
                            image.save(img_byte_arr, format='PNG')
                            image_bytes = img_byte_arr.getvalue()
                            
                            img_base64 = base64.b64encode(image_bytes).decode()
                            
                            st.session_state.saved_photos.append({
                                'name': file.name,
                                'image_bytes': image_bytes,
                                'image_base64': img_base64,
                                'embedding': embedding,
                                'lat': location['lat'],
                                'lon': location['lon'],
                                'km': location['km'],
                                'time': location['time'],
                                'tournament': selected_tournament,
                                'photographer': '작가'
                            })
                            
                        except Exception as e:
                            st.error(f"❌ {file.name} 처리 중 오류: {str(e)}")
                        
                        progress_bar.progress((idx + 1) / len(uploaded_files[:8]))
                    
                    status_text.empty()
                    progress_bar.empty()
                    
                    st.success(f"🎉 {len(uploaded_files[:8])}장의 사진이 저장되었습니다!")
                    st.balloons()
                    
                    # st.markdown("### 📍 자동 할당된 위치 정보")
                    # for idx, loc in enumerate(photo_locations):
                    #     st.text(f"사진 {idx+1}: {loc['km']}km 지점 | {loc['time']}")

# ==========================================
# 이용자 모드
# ==========================================
else:
    if not st.session_state.show_results:
        # 페이지 1: 대회 선택 + 사진 업로드
        st.title("🏃 High 러너스 🏃")
        st.caption("AI가 마라톤 코스에서 당신의 사진을 찾아드립니다")
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 80, 1])
        
        with col2:
            st.markdown("### 1️⃣ 대회 선택")
            selected = st.selectbox(
                "참가한 마라톤 대회를 선택하세요",
                options=["대회를 선택해주세요"] + list(tournaments.keys()),
                key="tournament_selectbox"
            )
            
            if selected != "대회를 선택해주세요":
                st.session_state.selected_tournament = selected
                
                st.markdown("### 2️⃣ 사진 업로드")
                uploaded_file = st.file_uploader(
                    "Drag and drop file here",
                    type=['png', 'jpg', 'jpeg'],
                    key="photo_uploader"
                )
                
                if uploaded_file:
                    image = Image.open(uploaded_file)
                    st.session_state.uploaded_image = image
                    
                    if st.button("🔍 코스 및 추천 사진 보기", type="primary"):
                        st.session_state.show_results = True
                        st.session_state.detailed_photo_id = None
                        st.session_state.show_detail_view = False # 상세 보기 모드 비활성화
                        st.rerun()
                else:
                    st.info("👆 대회 선택 후, 검색할 사진을 올려주세요")
            else:
                st.info("👆 위에서 대회를 먼저 선택해주세요")
        
    else:
        # 페이지 2: 결과 페이지 (상세 보기 모드 vs. 지도 검색 모드)
        
        tournament_name = st.session_state.selected_tournament
        tournament_info = tournaments[tournament_name]
        
        # 헤더
        header_col1, header_col2 = st.columns([1, 9])
        with header_col1:
            # 상세 보기 상태일 때는 '뒤로가기' 버튼으로 동작
            if st.session_state.show_detail_view:
                if st.button("⬅️ 유사 사진 목록으로", type="secondary", key="back_to_list"):
                    st.session_state.show_detail_view = False
                    st.session_state.selected_similar_photo_id = None
                    st.rerun()
            # 검색 결과 모드일 때는 '처음으로' 버튼으로 동작
            else:
                if st.button("◀️ 처음으로", type="secondary"):
                    st.session_state.show_results = False
                    st.session_state.selected_tournament = None
                    st.session_state.uploaded_image = None
                    st.rerun()
        
        with header_col2:
            st.markdown(f"""
            <h1 style='text-align: center; color: #2c3e50;'>
                {tournament_info['icon']} {tournament_name}
            </h1>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 레이아웃: 지도 50% / 콘텐츠 50%
        map_col, content_col = st.columns([5, 5])
        
        # ----------------------------------------------------
        # 1. 지도 영역 (Map Column) - 유사도 검색 및 지도 생성
        # ----------------------------------------------------
        with map_col:
            st.markdown("### 🗺️ 마라톤 코스 및 발견된 사진 위치")
            
            coordinates = load_marathon_course(tournament_name)
            photo_markers = []
            
            if coordinates and st.session_state.uploaded_image:
                with st.spinner("🤖 유사한 사진을 검색하고 있습니다..."):
                    try:
                        query_embedding = st.session_state.image_finder.get_image_embedding(
                            st.session_state.uploaded_image
                        )
                        
                        for saved_photo in st.session_state.saved_photos:
                            if saved_photo['tournament'] != tournament_name:
                                continue
                            
                            similarity = cosine_similarity(
                                query_embedding,
                                saved_photo['embedding']
                            )[0][0]
                            similarity_percent = float(similarity * 100)
                            
                            saved_photo['similarity'] = similarity_percent 
                            saved_photo['id'] = f"{saved_photo['tournament']}_{saved_photo['name']}"
                            
                            if similarity_percent >= 70:
                                photo_markers.append(saved_photo)
                            
                        photo_markers.sort(key=lambda x: x['similarity'], reverse=True)

                        m = create_course_map_with_photos(coordinates, photo_markers)
                        
                        if m:
                            st.success(f"✅ {len(photo_markers)}개의 유사한 사진을 찾았습니다! (마커/리스트 클릭 시 상세 보기)")
                            folium_static(m, width=950, height=500) # 와이드 레이아웃에 맞춰 너비 조정
                        
                    except Exception as e:
                        st.error(f"❌ 오류: {str(e)}")
                        
            elif not coordinates:
                st.error("❌ GPX 파일을 로드할 수 없습니다.")

        # ----------------------------------------------------
        # 2. 콘텐츠 영역 (List/Detail Column) - 화면 전환 핵심 로직
        # ----------------------------------------------------
        with content_col:
            selected_photo_id = st.session_state.selected_similar_photo_id
            
            # 지도 마커 클릭 시 상세 보기 상태로 전환
            if st.session_state.detailed_photo_id: 
                selected_photo_id = st.session_state.detailed_photo_id
                st.session_state.detailed_photo_id = None 
                st.session_state.show_detail_view = True 

            selected_photo = next((p for p in photo_markers if p['id'] == selected_photo_id), None)
            
            # --- 상세 보기 화면 (선택된 이미지) ---
            if st.session_state.show_detail_view and selected_photo:
                # 선택된 이미지가 보이면서 검색한 사진과 리스트는 사라짐 (요청 반영)
                
                st.markdown("#### ✨ 선택된 이미지 상세")

                # 이미지 표시 (오류 해결 반영)
                image_bytes_to_st_image(selected_photo['image_bytes'], use_container_width=True)
                
                st.markdown("---")
                
                # 작가 정보
                st.markdown("##### 👤 촬영자 정보")
                
                col_prof1, col_prof2 = st.columns([1, 3])
                with col_prof1:
                    #  대신 임시 이미지 표시
                    st.markdown("", unsafe_allow_html=True) 
                with col_prof2:
                    st.markdown(f"**{selected_photo.get('photographer', '작가')}**")
                    st.caption("마라톤 전문 포토그래퍼")

                st.markdown("---")
                
                # 구매 버튼 구현 (새로운 Streamlit 창 열기)
                st.metric("가격", "5,000원", "고해상도 다운로드")
                
                # 새 Streamlit 창을 여는 버튼 (실제 앱 URL로 대체 필요)
                purchase_url = "https://share.streamlit.io/simple-purchase-page" 
                st.markdown(f'<a href="{purchase_url}" target="_blank">'
                            f'<button class="purchase-btn-style">' # CSS 클래스 사용
                            f'🛒 구매하기 (새 창 열림)'
                            f'</button></a>', unsafe_allow_html=True)


            # --- 유사 사진 목록 화면 ---
            else:
                st.markdown("#### 🖼️ 검색한 사진")
                if st.session_state.uploaded_image:
                    st.image(st.session_state.uploaded_image, width=200) 
                
                st.markdown("---")
                st.markdown("#### 🎯 유사한 사진 목록")
                
                if photo_markers:
                    for photo in photo_markers:
                        def set_selected_photo_and_show_detail(photo_id):
                            st.session_state.selected_similar_photo_id = photo_id
                            st.session_state.show_detail_view = True 

                        # 리스트 아이템 레이아웃
                        list_item_col1, list_item_col2 = st.columns([1, 2])
                        
                        with list_item_col1:
                            # 이미지 표시 (오류 해결 반영)
                            image_bytes_to_st_image(photo['image_bytes'], width=80) 

                        with list_item_col2:
                            st.markdown(f"**{photo['km']}km 지점**")
                            st.markdown(f"<span style='color:red;'>유사도: {photo['similarity']:.1f}%</span>", unsafe_allow_html=True)
                            
                            # '보기' 버튼 클릭 시 상세 보기 모드로 전환
                            if st.button("보기", key=f"list_btn_{photo['id']}"):
                                set_selected_photo_and_show_detail(photo['id'])
                                st.rerun()

                else:
                    st.info("검색 결과를 찾을 수 없습니다.")

# st.caption("💡 Tip: 작가 모드에서 사진을 먼저 업로드해야 검색이 가능합니다")