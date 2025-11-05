"""
마라톤 사진 검색 플랫폼
대회 선택 → 사진 업로드 → 새 화면에서 코스 지도 + 유사 사진 표시
"""

import streamlit as st
from PIL import Image
import gpxpy
import folium
from streamlit_folium import folium_static
import os
import glob
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import io
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 세션 상태 초기화
if 'chat_open' not in st.session_state:
    st.session_state.chat_open = False
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'selected_location' not in st.session_state:
    st.session_state.selected_location = None
if 'saved_photos' not in st.session_state:
    st.session_state.saved_photos = {}
if 'saved_count' not in st.session_state:
    st.session_state.saved_count = 0
if 'image_finder' not in st.session_state:
    st.session_state.image_finder = None


# API 호출 함수
def call_api(user_message):
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        completion = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": "너는 달리기 강습 전문가야."},
                {"role": "user", "content": user_message}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"오류 발생: {str(e)}"

# EXIF 데이터 추출 함수
def get_exif_datetime(image):
    """이미지에서 촬영 시간을 추출합니다."""
    try:
        exif_data = image._getexif()
        if not exif_data:
            return None
        
        datetime_tags = ['DateTimeOriginal', 'DateTime', 'DateTimeDigitized']
        
        for tag, value in exif_data.items():
            tag_name = TAGS.get(tag, tag)
            if tag_name in datetime_tags:
                try:
                    dt = datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S")
                    return dt
                except:
                    continue
        return None
    except Exception as e:
        print(f"시간 추출 오류: {str(e)}")
        return None

def get_gps_from_image(image):
    """이미지에서 GPS 좌표를 추출합니다."""
    try:
        exif_data = image._getexif()
        if not exif_data:
            return None, None
        
        gps_info = {}
        for tag, value in exif_data.items():
            tag_name = TAGS.get(tag, tag)
            if tag_name == 'GPSInfo':
                for gps_tag in value:
                    gps_tag_name = GPSTAGS.get(gps_tag, gps_tag)
                    gps_info[gps_tag_name] = value[gps_tag]
        
        if not gps_info:
            return None, None
        
        def convert_to_degrees(value):
            d = float(value[0])
            m = float(value[1])
            s = float(value[2])
            return d + (m / 60.0) + (s / 3600.0)
        
        lat = None
        lon = None
        
        if 'GPSLatitude' in gps_info and 'GPSLatitudeRef' in gps_info:
            lat = convert_to_degrees(gps_info['GPSLatitude'])
            if gps_info['GPSLatitudeRef'] == 'S':
                lat = -lat
        
        if 'GPSLongitude' in gps_info and 'GPSLongitudeRef' in gps_info:
            lon = convert_to_degrees(gps_info['GPSLongitude'])
            if gps_info['GPSLongitudeRef'] == 'W':
                lon = -lon
        
        return lat, lon
    except Exception as e:
        print(f"GPS 추출 오류: {str(e)}")
        return None, None
    

# ==========================================
# ImageSimilarityFinder 클래스
# ==========================================
class ImageSimilarityFinder:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    @st.cache_resource
    def load_model(_self):
        """모델 로드 (캐싱)"""
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.to(_self.device)
        return model, processor
    
    def get_image_embedding(self, image):
        """이미지의 임베딩 벡터 생성"""
        if self.model is None or self.processor is None:
            self.model, self.processor = self.load_model()
        
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        else:
            image = image.convert('RGB')
            
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            embedding = self.model.get_image_features(**inputs)
        
        return embedding.cpu().numpy()

# 이미지 파인더 초기화
if st.session_state.image_finder is None:
    st.session_state.image_finder = ImageSimilarityFinder()

# GPX 코스 로드
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
            return None
    return None

def create_clickable_course_map(coordinates):
    # """클릭 가능한 코스 지도 생성 (작가 모드용)"""
    if not coordinates:
        return None
    
    center_lat = sum([c[0] for c in coordinates]) / len(coordinates)
    center_lon = sum([c[1] for c in coordinates]) / len(coordinates)
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='CartoDB positron'
    )
    
    folium.PolyLine(
        coordinates,
        color='#FF4444',
        weight=5,
        opacity=0.8,
        popup='마라톤 코스 - 클릭하여 사진 촬영 위치 선택'
    ).add_to(m)
    
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
    
    return m

def create_course_map(coordinates, photo_locations=None):
    """코스 지도 + 사진 위치 표시 (이용자 모드용)"""
    if not coordinates:
        return None
    
    center_lat = sum([c[0] for c in coordinates]) / len(coordinates)
    center_lon = sum([c[1] for c in coordinates]) / len(coordinates)
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='CartoDB positron'
    )
    
    folium.PolyLine(
        coordinates,
        color='#FF4444',
        weight=5,
        opacity=0.8,
        popup='마라톤 코스'
    ).add_to(m)
    
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
    
    if photo_locations:
        for photo in photo_locations:
            folium.Marker(
                [photo['lat'], photo['lon']],
                popup=folium.Popup(
                    f"""
                    <div style='width: 200px;'>
                        <b>{photo['name']}</b><br>
                        <small>📍 {photo.get('location', '위치 정보 없음')}</small>
                    </div>
                    """,
                    max_width=220
                ),
                icon=folium.Icon(color='orange', icon='camera')
            ).add_to(m)
    
    return m

# ==========================================
# 세션 스테이트 초기화
# ==========================================
# if 'saved_photos' not in st.session_state:
#     st.session_state.saved_photos = []
# if 'saved_count' not in st.session_state:
#     st.session_state.saved_count = 0
# if 'image_finder' not in st.session_state:
#     st.session_state.image_finder = ImageSimilarityFinder()


# ==========================================
# GPX지도 설정
# ==========================================

# def load_marathon_course(tournament_name):
#     """
#     대회 이름에 따라 GPX 파일 로드
#     """
#     gpx_files = {
#         "JTBC 마라톤": "../data/2025_JTBC.gpx",
#         "춘천 마라톤": "../data/chuncheon_marathon.gpx",
#     }
    
#     if tournament_name in gpx_files:
#         try:
#             with open(gpx_files[tournament_name], 'r') as f:
#                 gpx = gpxpy.parse(f)
            
#             coordinates = []
#             for track in gpx.tracks:
#                 for segment in track.segments:
#                     for point in segment.points:
#                         coordinates.append([point.latitude, point.longitude])
            
#             return coordinates
#         except FileNotFoundError:
#             return None
#     return None

# def create_course_map(coordinates, photo_locations=None):
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
# 모드 선택

mode = st.radio(
    "모드 선택",
    ["📸 작가 모드", "🔍 이용자 모드"],
    label_visibility="collapsed"
)

# ==========================================
# 작가 모드
# ==========================================
if mode == "📸 작가 모드":
    st.markdown("### 📸 사진 업로드 및 AI 분류")
    st.info("💡 대회를 선택하고, 지도에서 촬영 위치를 클릭한 후, 사진을 업로드하세요.")
    
    st.markdown("### 1️⃣ 대회 선택")
    
    tournaments = {
        "JTBC 마라톤": {"date": "2025년 11월 2일", "distance": "42.195km"},
        "춘천 마라톤": {"date": "2025년 10월 26일", "distance": "42.195km"}
    }
    
    selected_tournament = st.selectbox(
        "사진을 업로드할 대회를 선택하세요",
        options=["대회를 선택해주세요"] + list(tournaments.keys()),
        key="photographer_tournament"
    )
    
    if selected_tournament != "대회를 선택해주세요":
        st.success(f"✅ **{selected_tournament}** 선택됨")
        
        if selected_tournament not in st.session_state.saved_photos:
            st.session_state.saved_photos[selected_tournament] = []
        
        st.markdown("---")
        st.markdown("### 2️⃣ 사진 촬영 위치 선택")
        st.caption("지도를 클릭하여 사진이 촬영된 위치를 선택하세요")
        
        coordinates = load_marathon_course(selected_tournament)
        
        if coordinates:
            m = create_clickable_course_map(coordinates)
            map_data = st_folium(m, width=700, height=500, key="photographer_map")
            
            if map_data and map_data.get('last_clicked'):
                clicked_lat = map_data['last_clicked']['lat']
                clicked_lon = map_data['last_clicked']['lng']
                st.session_state.selected_location = {
                    'lat': clicked_lat,
                    'lon': clicked_lon
                }
                st.success(f"📍 선택된 위치: 위도 {clicked_lat:.6f}, 경도 {clicked_lon:.6f}")
            
            if st.session_state.selected_location:
                st.markdown("---")
                st.markdown("### 3️⃣ 사진 업로드")
                uploaded_files = st.file_uploader(
                    "사진을 선택하세요 (여러 장 가능)",
                    type=['png', 'jpg', 'jpeg'],
                    accept_multiple_files=True,
                    key="photographer_upload"
                )
                
                if uploaded_files:
                    st.success(f"✅ {len(uploaded_files)}장의 사진이 업로드되었습니다!")
                    st.markdown("### 📷 업로드된 사진")
                    
                    cols = st.columns(4)
                    photo_data = []
                    
                    for idx, uploaded_file in enumerate(uploaded_files):
                        col = cols[idx % 4]
                        
                        with col:
                            image = Image.open(uploaded_file)
                            st.image(image, use_container_width=True)
                            
                            photo_datetime = get_exif_datetime(image)
                            
                            if photo_datetime:
                                st.success(f"⏰ 촬영 시간")
                                st.code(photo_datetime.strftime("%Y-%m-%d %H:%M:%S"))
                            else:
                                st.warning("⚠️ 시간 정보 없음")
                            
                            exif_lat, exif_lon = get_gps_from_image(image)
                            
                            if exif_lat and exif_lon:
                                st.info("📷 EXIF GPS 사용")
                                use_lat, use_lon = exif_lat, exif_lon
                            else:
                                st.info("📍 선택한 위치 사용")
                                use_lat = st.session_state.selected_location['lat']
                                use_lon = st.session_state.selected_location['lon']
                            
                            manual_location = st.text_input(
                                "위치 설명 (선택)",
                                placeholder="예: 서울역 앞",
                                key=f"location_{idx}"
                            )
                            
                            photo_data.append({
                                'image': image,
                                'name': uploaded_file.name,
                                'location': manual_location,
                                'latitude': use_lat,
                                'longitude': use_lon,
                                'photo_datetime': photo_datetime,
                                'uploaded_file': uploaded_file
                            })
                    
                    st.markdown("---")
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("💾 DB에 저장하기", type="primary"):
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            saved_count = 0
                            
                            for idx, photo in enumerate(photo_data):
                                status_text.text(f"🤖 AI 처리 중... ({idx + 1}/{len(photo_data)})")
                                
                                try:
                                    embedding = st.session_state.image_finder.get_image_embedding(photo['image'])
                                    photo['embedding'] = embedding
                                    photo['upload_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    photo['tournament'] = selected_tournament
                                    
                                    img_byte_arr = io.BytesIO()
                                    photo['image'].save(img_byte_arr, format='PNG')
                                    photo['image_bytes'] = img_byte_arr.getvalue()
                                    
                                    saved_count += 1
                                    
                                except Exception as e:
                                    st.error(f"❌ {photo['name']} 처리 중 오류: {str(e)}")
                                    continue
                                
                                progress_bar.progress((idx + 1) / len(photo_data))
                            
                            st.session_state.saved_photos[selected_tournament].extend(photo_data)
                            st.session_state.saved_count += saved_count
                            
                            status_text.empty()
                            progress_bar.empty()
                            
                            st.success(f"✅ {saved_count}장의 사진이 **{selected_tournament}**에 저장되었습니다!")
                            
                            time_count = sum(1 for p in photo_data if p['photo_datetime'] is not None)
                            st.info(f"⏰ 촬영 시간 데이터: {time_count}/{len(photo_data)}장")
                            
                            st.balloons()
                            st.session_state.selected_location = None
                            st.rerun()
            else:
                st.warning("⚠️ 먼저 지도에서 사진 촬영 위치를 클릭해주세요!")
        
        else:
            st.error("❌ 코스 데이터를 불러올 수 없습니다.")
        
        if selected_tournament in st.session_state.saved_photos:
            saved_count = len(st.session_state.saved_photos[selected_tournament])
            if saved_count > 0:
                st.markdown("---")
                st.markdown(f"### 📊 현재 저장된 사진: **{saved_count}장**")
                
                time_photos = [p for p in st.session_state.saved_photos[selected_tournament] 
                              if p.get('photo_datetime') is not None]
                st.info(f"⏰ 촬영 시간 포함: {len(time_photos)}장 ({len(time_photos)/saved_count*100:.1f}%)")
    
    else:
        st.info("👆 먼저 대회를 선택해주세요")


# ==========================================
# 이용자 모드
# ==========================================
else:
    st.set_page_config(
        page_title="마라톤 사진 검색",
        page_icon="🏃‍♂️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    st.markdown("""
    <style>
        .main { background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%); }
        .stSelectbox { font-size: 18px; }
        .stButton>button {
            background: linear-gradient(90deg, #4a90e2 0%, #50e3c2 100%);
            color: white; font-size: 18px; font-weight: bold;
            padding: 15px 30px; border-radius: 12px; border: none;
            width: 100%; transition: all 0.3s;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(74, 144, 226, 0.4);
        }
        .info-card {
            background: white; padding: 20px; border-radius: 12px;
            border-left: 4px solid #4a90e2;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 20px;
        }
        h1 { color: #2c3e50; text-align: center; font-size: 48px; margin-bottom: 30px; }
        h3 { color: #4a90e2; font-size: 22px; }
    </style>
    """, unsafe_allow_html=True)

    if 'selected_tournament' not in st.session_state:
        st.session_state.selected_tournament = None
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False

    tournaments = {
        "JTBC 마라톤": {
            "date": "2025년 11월 2일", "distance": "42.195km",
            "participants": "30,000명",
            "course": "잠실종합운동장 → 광화문 → 남산 → 한강 → 잠실",
            "icon": "🏃‍♂️"
        },
        "춘천 마라톤": {
            "date": "2025년 10월 26일", "distance": "42.195km",
            "participants": "15,000명",
            "course": "의암호 → 소양강 → 춘천시가지 → 의암호",
            "icon": "🏔️"
        }
    }

    if not st.session_state.show_results:
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
                    key="photo_uploader",
                    help="마라톤 사진을 업로드하세요"
                )
                
                if uploaded_file:
                    image = Image.open(uploaded_file)
                    st.session_state.uploaded_image = image
                    
                    if st.button("🔍 코스 및 추천 사진 보기", type="primary"):
                        st.session_state.show_results = True
                        st.rerun()
            else:
                st.info("👆 위에서 대회를 먼저 선택해주세요")

    else:
        tournament_name = st.session_state.selected_tournament
        tournament_info = tournaments[tournament_name]
        
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; background: white; border-radius: 12px; margin-bottom: 30px;'>
            <h1 style='margin: 0; font-size: 36px;'>{tournament_info['icon']} {tournament_name}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        left_col, right_col = st.columns([6, 4])
        
        with left_col:
            st.markdown("### 🗺️ 마라톤 코스")
            
            st.markdown(f"""
            <div class="info-card">
                <p style='margin: 0; line-height: 1.8;'>
                    📅 <b>일시:</b> {tournament_info['date']}<br>
                    📏 <b>거리:</b> {tournament_info['distance']}<br>
                    📍 <b>코스:</b> {tournament_info['course']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            coordinates = load_marathon_course(tournament_name)
        
            if coordinates:
                st.success(f"✅ {tournament_name} 코스를 불러왔습니다!")
                m = create_course_map(coordinates)
                if m:
                    folium_static(m, width=1300, height=600)
            else:
                st.error("❌ 코스 데이터를 찾을 수 없습니다.")

        with right_col:
            if st.session_state.uploaded_image:
                st.markdown("#### 🖼️ 검색한 사진")
                image = st.session_state.uploaded_image
                st.image(image, width=400)
                st.markdown("---")

                st.markdown("#### ⚙️ 검색 옵션")
                
                top_k = st.slider("📊 표시할 결과 개수", min_value=1, max_value=20, value=5)
                similarity_threshold = st.slider(
                    "🎯 최소 유사도 (%)", min_value=0, max_value=100, value=70,
                    help="높을수록 더 비슷한 사진만 표시됩니다"
                )
                
                st.markdown("---")
                
                search_button = st.button("🔍 유사 사진 검색", type="primary", use_container_width=True)
                
                if search_button:
                    if tournament_name not in st.session_state.saved_photos or len(st.session_state.saved_photos[tournament_name]) == 0:
                        st.warning(f"⚠️ **{tournament_name}**에 저장된 사진이 없습니다. 먼저 작가 모드에서 사진을 업로드해주세요.")
                    else:
                        with st.spinner("🤖 AI가 코스 위에서 유사한 사진을 찾고 있습니다..."):
                            try:
                                query_image = st.session_state.uploaded_image
                                query_embedding = st.session_state.image_finder.get_image_embedding(query_image)
                                
                                results = []
                                for saved_photo in st.session_state.saved_photos[tournament_name]:
                                    if 'embedding' not in saved_photo:
                                        continue
                                    
                                    similarity = cosine_similarity(query_embedding, saved_photo['embedding'])[0][0]
                                    similarity_percent = float(similarity * 100)
                                    
                                    if similarity_percent >= similarity_threshold:
                                        results.append({
                                            'photo': saved_photo,
                                            'similarity': similarity_percent
                                        })
                                
                                results.sort(key=lambda x: x['similarity'], reverse=True)
                                results = results[:top_k]
                                
                                if len(results) == 0:
                                    st.warning("😔 조건에 맞는 사진을 찾지 못했습니다.")
                                else:
                                    st.success(f"✅ **{len(results)}장**의 유사한 사진을 찾았습니다!")
                                    st.markdown("---")
                                    
                                    for idx, result in enumerate(results):
                                        col1, col2 = st.columns([1, 2])
                                        
                                        with col1:
                                            result_image = Image.open(io.BytesIO(result['photo']['image_bytes']))
                                            st.image(result_image, use_container_width=True)
                                        
                                        with col2:
                                            st.markdown(f"**#{idx + 1}**")
                                            st.markdown(f"**📍 {result['photo'].get('location', '위치 미상')}**")
                                            st.markdown(f"**📷 {result['photo']['name']}**")
                                            
                                            if result['photo'].get('photo_datetime'):
                                                st.caption(f"⏰ 촬영: {result['photo']['photo_datetime'].strftime('%Y-%m-%d %H:%M:%S')}")
                                            
                                            if result['photo'].get('latitude') and result['photo'].get('longitude'):
                                                st.caption(f"📍 GPS: {result['photo']['latitude']:.6f}, {result['photo']['longitude']:.6f}")
                                            
                                            similarity_val = float(result['similarity'] / 100)
                                            st.progress(similarity_val)
                                            st.caption(f"유사도: {result['similarity']:.2f}%")
                                            
                                            if 'upload_timestamp' in result['photo']:
                                                st.caption(f"업로드: {result['photo']['upload_timestamp']}")
                                        
                                        st.markdown("---")
                            
                            except Exception as e:
                                st.error(f"❌ 검색 중 오류가 발생했습니다: {str(e)}")
            
            else:
                st.markdown("""
                <div style='padding: 50px 20px; text-
                """)

    # # ==========================================
    # # 간결한 CSS 스타일
    # # ==========================================
    # st.markdown("""
    # <style>
    #     /* 전체 배경 */
    #     .main {
    #         background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%);
    #     }
        
    #     /* 드롭다운 스타일 */
    #     .stSelectbox {
    #         font-size: 18px;
    #     }
        
    #     /* 버튼 스타일 */
    #     .stButton>button {
    #         background: linear-gradient(90deg, #4a90e2 0%, #50e3c2 100%);
    #         color: white;
    #         font-size: 18px;
    #         font-weight: bold;
    #         padding: 15px 30px;
    #         border-radius: 12px;
    #         border: none;
    #         width: 100%;
    #         transition: all 0.3s;
    #     }
        
    #     .stButton>button:hover {
    #         transform: translateY(-2px);
    #         box-shadow: 0 6px 20px rgba(74, 144, 226, 0.4);
    #     }
        
    #     /* 업로드 영역 */
    #     .stFileUploader {
    #         border: 2px dashed #4a90e2;
    #         border-radius: 12px;
    #         padding: 30px;
    #         background: white;
    #     }
        
    #     /* 카드 스타일 */
    #     .info-card {
    #         background: white;
    #         padding: 20px;
    #         border-radius: 12px;
    #         border-left: 4px solid #4a90e2;
    #         box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    #         margin-bottom: 20px;
    #     }
        
    #     /* 사진 카드 */
    #     .photo-card {
    #         background: white;
    #         padding: 15px;
    #         border-radius: 10px;
    #         border: 2px solid #e0e7ff;
    #         text-align: center;
    #         transition: all 0.3s;
    #         cursor: pointer;
    #     }
        
    #     .photo-card:hover {
    #         transform: scale(1.05);
    #         border-color: #4a90e2;
    #         box-shadow: 0 4px 12px rgba(74, 144, 226, 0.3);
    #     }
        
    #     /* 제목 */
    #     h1 {
    #         color: #2c3e50;
    #         text-align: center;
    #         font-size: 48px;
    #         margin-bottom: 30px;
    #     }
        
    #     h2 {
    #         color: #34495e;
    #         font-size: 28px;
    #     }
        
    #     h3 {
    #         color: #4a90e2;
    #         font-size: 22px;
    #     }
    # </style>
    # """, unsafe_allow_html=True)

    # # ==========================================
    # # 세션 스테이트 초기화
    # # ==========================================
    # if 'selected_tournament' not in st.session_state:
    #     st.session_state.selected_tournament = None

    # if 'uploaded_image' not in st.session_state:
    #     st.session_state.uploaded_image = None

    # if 'show_results' not in st.session_state:
    #     st.session_state.show_results = False

    # # ==========================================
    # # 대회 데이터
    # # ==========================================
    # tournaments = {
    #     "JTBC 마라톤": {
    #         "date": "2025년 11월 2일",
    #         "distance": "42.195km",
    #         "participants": "30,000명",
    #         "course": "잠실종합운동장 → 광화문 → 남산 → 한강 → 잠실",
    #         "icon": "🏃‍♂️",
    #         "color": "#FF6B6B"
    #     },
    #     "춘천 마라톤": {
    #         "date": "2025년 10월 26일",
    #         "distance": "42.195km",
    #         "participants": "15,000명",
    #         "course": "의암호 → 소양강 → 춘천시가지 → 의암호",
    #         "icon": "🏔️",
    #         "color": "#4ECDC4"
    #     }
    # }
    # # ==========================================
    # # 페이지 1: 대회 선택 및 사진 업로드
    # # ==========================================
    # if not st.session_state.show_results:
        
    #     # 타이틀
    #     st.title("🏃 High 러너스 🏃")
    #     st.caption("AI가 마라톤 코스에서 당신의 사진을 찾아드립니다")
    #     st.markdown("---")
        
    #     # 중앙 정렬 레이아웃
    #     col1, col2, col3 = st.columns([1, 80, 1])
        
    #     with col2:
    #         # 1단계: 대회 선택
    #         st.markdown("### 1️⃣ 대회 선택")
    #         selected = st.selectbox(
    #             "참가한 마라톤 대회를 선택하세요",
    #             options=["대회를 선택해주세요"] + list(tournaments.keys()),
    #             key="tournament_selectbox"
    #         )
            
    #         # 대회가 선택되면 세션에 저장
    #         if selected != "대회를 선택해주세요":
    #             st.session_state.selected_tournament = selected
                
    #             # 2단계: 사진 업로드
    #             st.markdown("### 2️⃣ 사진 업로드")
    #             uploaded_file = st.file_uploader(
    #                 "Drag and drop file here",
    #                 type=['png', 'jpg', 'jpeg'],
    #                 key="photo_uploader",
    #                 help="마라톤 사진을 업로드하세요 (최대 200MB)"
    #             )
                
    #             # 사진이 업로드되면
    #             if uploaded_file:
    #                 # 이미지 읽기 및 세션에 저장
    #                 image = Image.open(uploaded_file)
    #                 st.session_state.uploaded_image = image
                    
    #                 # # 미리보기 표시
    #                 # st.success(f"✅ {uploaded_file.name} 업로드 완료!")
    #                 # st.image(image, caption="업로드된 사진", use_container_width=True)
                    
    #                 # st.markdown("---")
                    
    #                 # 검색 버튼
    #                 if st.button("🔍 코스 및 추천 사진 보기", type="primary"):
    #                     st.session_state.show_results = True
    #                     st.rerun()
            
    #         else:
    #             st.info("👆 위에서 대회를 먼저 선택해주세요")

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
                    folium_static(m, width=1300, height=600)
            else:
                st.error("❌ 코스 데이터를 찾을 수 없습니다.")

        # ==========================================
        # 오른쪽: 유사한 사진들
        # ==========================================
        with right_col:
            # st.markdown("### 📍 코스 상 유사한 사진들")
            
            # 업로드한 사진 미리보기
            if st.session_state.uploaded_image:
                st.markdown("#### 🖼️ 검색한 사진")
                image = st.session_state.uploaded_image
                st.image(image, width=400)
                st.markdown("---")

                # 검색 옵션
                st.markdown("#### ⚙️ 검색 옵션")
                
                # 코스 구간 선택
                course_section = st.selectbox(
                    "📍 코스 구간 (선택사항)",
                    ["전체 코스", "0-10km", "10-20km", "20-30km", "30-42km"]
                )
                
                # 결과 개수
                top_k = st.slider(
                    "📊 표시할 결과 개수",
                    min_value=1,
                    max_value=20,
                    value=5
                )

                # 유사도 임계값
                similarity_threshold = st.slider(
                    "🎯 최소 유사도 (%)",
                    min_value=0,
                    max_value=100,
                    value=70,
                    help="높을수록 더 비슷한 사진만 표시됩니다"
                )
                
                st.markdown("---")
                
                # 검색 버튼
                search_button = st.button("🔍 유사 사진 검색", type="primary", use_container_width=True)
                
                if search_button:
                    if len(st.session_state.saved_photos) == 0:
                        st.warning("⚠️ 저장된 사진이 없습니다. 먼저 작가 모드에서 사진을 업로드해주세요.")
                    else:
                        with st.spinner("🤖 AI가 코스 위에서 유사한 사진을 찾고 있습니다..."):
                            try:
                                # 검색 이미지의 임베딩 생성
                                query_image = st.session_state.uploaded_image
                                query_embedding = st.session_state.image_finder.get_image_embedding(query_image)
                                
                                # 저장된 모든 이미지와 유사도 계산
                                results = []
                                for saved_photo in st.session_state.saved_photos:
                                    if 'embedding' not in saved_photo:
                                        continue
                                    
                                    # 유사도 계산
                                    similarity = cosine_similarity(query_embedding, saved_photo['embedding'])[0][0]
                                    similarity_percent = float(similarity * 100)
                                    
                                    # 임계값 필터
                                    if similarity_percent >= similarity_threshold:
                                        results.append({
                                            'photo': saved_photo,
                                            'similarity': similarity_percent
                                        })
                                
                                # 유사도 순으로 정렬
                                results.sort(key=lambda x: x['similarity'], reverse=True)
                                results = results[:top_k]
                                
                                # 결과 표시
                                if len(results) == 0:
                                    st.warning("😔 조건에 맞는 사진을 찾지 못했습니다.")
                                else:
                                    st.success(f"✅ **{len(results)}장**의 유사한 사진을 찾았습니다!")
                                    st.markdown("---")
                                    
                                    for idx, result in enumerate(results):
                                        col1, col2 = st.columns([1, 2])
                                        
                                        with col1:
                                            # 저장된 이미지 표시
                                            result_image = Image.open(io.BytesIO(result['photo']['image_bytes']))
                                            st.image(result_image, use_container_width=True)
                                        
                                        with col2:
                                            st.markdown(f"**#{idx + 1}**")
                                            st.markdown(f"**📍 {result['photo'].get('location', '위치 미상')}**")
                                            st.markdown(f"**📁 {result['photo']['name']}**")
                                            
                                            # 유사도 표시
                                            similarity_val = float(result['similarity'] / 100)
                                            st.progress(similarity_val)
                                            st.caption(f"유사도: {result['similarity']:.2f}%")
                                            
                                            # 타임스탬프
                                            if 'timestamp' in result['photo']:
                                                st.caption(f"업로드: {result['photo']['timestamp']}")
                                        
                                        st.markdown("---")
                                    
                                    # 세션 스테이트 업데이트
                                    st.session_state.uploaded_photo = image
                                    st.session_state.show_recommendations = True
                            
                            except Exception as e:
                                st.error(f"❌ 검색 중 오류가 발생했습니다: {str(e)}")
            
            else:
                # 업로드 전 안내
                st.markdown("""
                <div class="upload-area">
                    <div style='padding: 50px 20px;'>
                        <div style='font-size: 64px; margin-bottom: 20px;'>📤</div>
                        <h3 style='color: #666; margin-bottom: 10px;'>사진을 업로드하세요</h3>
                        <p style='color: #999;'>JPG, PNG 형식 지원</p>
                        <br>
                        <small style='color: #bbb;'>위 버튼을 클릭하여 파일을 선택하세요</small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
    # ==========================================
    # 푸터
    # ==========================================
    st.markdown("---")
    st.caption("🤖 Powered by OpenAI CLIP Model | 이미지 임베딩 기반 유사도 검색")            
        
        # ==========================================
        # 하단: 뒤로 가기 버튼
        # ==========================================
    col1, col2, col3 = st.columns([5, 10, 5])
    with col2:
        if st.button("무엇이든 물어보세요!", key="chatbot_btn", use_container_width=True):
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

# ==========================================
# 하단 푸터
# ==========================================
st.markdown("---")
st.caption("💡 Tip: 정확한 검색을 위해 선명한 사진을 업로드해주세요")