# """
# 마라톤 사진 검색 플랫폼
# 대회 선택 → 사진 업로드 → 새 화면에서 코스 지도 + 유사 사진 표시
# """

import streamlit as st
from PIL import Image, ExifTags
import gpxpy
import folium
from streamlit_folium import st_folium, folium_static
import os
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import io
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI


# ===== 파일 최상단 (import 아래) =====
default_state = {
    'chat_open': False,
    'messages': [],
    'selected_location': None,
    'saved_photos': {},          # ← dict
    'saved_count': 0,
    'image_finder': None,
    'selected_tournament': None,
    'uploaded_image': None,
    'show_results': False
}

for k, v in default_state.items():
    if k not in st.session_state:
        st.session_state[k] = v

load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 세션 상태 초기화
for key in ['chat_open', 'messages', 'selected_location', 'saved_photos', 'saved_count', 'image_finder']:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ['messages', 'saved_photos'] else None
if 'saved_photos' not in st.session_state:
    st.session_state.saved_photos = {}
if 'saved_count' not in st.session_state:
    st.session_state.saved_count = 0

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

# EXIF 데이터 추출
def get_exif_datetime(image):
    try:
        exif_data = image._getexif()
        if not exif_data:
            return None
        for tag, value in exif_data.items():
            tag_name = ExifTags.TAGS.get(tag, tag)
            if tag_name in ['DateTimeOriginal', 'DateTime', 'DateTimeDigitized']:
                try:
                    return datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S")
                except:
                    continue
        return None
    except:
        return None

def get_gps_from_image(image):
    try:
        exif_data = image._getexif()
        if not exif_data:
            return None, None
        gps_info = {}
        for tag, value in exif_data.items():
            tag_name = ExifTags.TAGS.get(tag, tag)
            if tag_name == 'GPSInfo':
                for gps_tag in value:
                    gps_tag_name = ExifTags.GPSTAGS.get(gps_tag, gps_tag)
                    gps_info[gps_tag_name] = value[gps_tag]
        if not gps_info:
            return None, None
        def convert_to_degrees(value):
            d, m, s = map(float, value)
            return d + (m / 60.0) + (s / 3600.0)
        lat = convert_to_degrees(gps_info['GPSLatitude']) * (-1 if gps_info['GPSLatitudeRef'] == 'S' else 1)
        lon = convert_to_degrees(gps_info['GPSLongitude']) * (-1 if gps_info['GPSLongitudeRef'] == 'W' else 1)
        return lat, lon
    except:
        return None, None

# ImageSimilarityFinder 클래스
class ImageSimilarityFinder:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @st.cache_resource
    def load_model(_self):
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.to(_self.device)
        return model, processor
    
    def get_image_embedding(self, image):
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
    gpx_files = {
        "JTBC 마라톤": "../data/2025_JTBC.gpx",
        "춘천 마라톤": "../data/chuncheon_marathon.gpx",
    }
    if tournament_name not in gpx_files:
        return None
    try:
        with open(gpx_files[tournament_name], 'r') as f:
            gpx = gpxpy.parse(f)
        coordinates = []
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    coordinates.append([point.latitude, point.longitude])
        return coordinates
    except:
        return None

# 클릭 가능한 지도 (작가 모드)
def create_clickable_course_map(coordinates):
    if not coordinates:
        return None
    center_lat = sum(c[0] for c in coordinates) / len(coordinates)
    center_lon = sum(c[1] for c in coordinates) / len(coordinates)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='CartoDB positron')
    folium.PolyLine(coordinates, color='#FF4444', weight=5, opacity=0.8).add_to(m)
    folium.Marker(coordinates[0], popup='출발', icon=folium.Icon(color='green', icon='play')).add_to(m)
    folium.Marker(coordinates[-1], popup='도착', icon=folium.Icon(color='red', icon='stop')).add_to(m)
    total_points = len(coordinates)
    for km in [10, 20, 21.0975, 30, 40]:
        idx = int((km / 42.195) * total_points)
        if idx < total_points:
            folium.CircleMarker(location=coordinates[idx], radius=8, popup=f'{km}km', color='blue', fill=True, fillColor='lightblue').add_to(m)
    return m

# 코스 지도 (이용자 모드)
def create_course_map(coordinates, photo_locations=None):
    if not coordinates:
        return None
    center_lat = sum(c[0] for c in coordinates) / len(coordinates)
    center_lon = sum(c[1] for c in coordinates) / len(coordinates)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='CartoDB positron')
    folium.PolyLine(coordinates, color='#FF4444', weight=5, opacity=0.8).add_to(m)
    folium.Marker(coordinates[0], popup='출발', icon=folium.Icon(color='green', icon='play')).add_to(m)
    folium.Marker(coordinates[-1], popup='도착', icon=folium.Icon(color='red', icon='stop')).add_to(m)
    total_points = len(coordinates)
    for km in [10, 20, 21.0975, 30, 40]:
        idx = int((km / 42.195) * total_points)
        if idx < total_points:
            folium.CircleMarker(location=coordinates[idx], radius=8, popup=f'{km}km', color='blue', fill=True, fillColor='lightblue').add_to(m)
    if photo_locations:
        for photo in photo_locations:
            folium.Marker(
                [photo['lat'], photo['lon']],
                popup=folium.Popup(f"<div style='width:200px'><b>{photo['name']}</b><br><small>위치: {photo.get('location', '미상')}</small></div>", max_width=220),
                icon=folium.Icon(color='orange', icon='camera')
            ).add_to(m)
    return m

# 모드 선택
mode = st.radio("모드 선택", ["작가 모드", "이용자 모드"], label_visibility="collapsed")

# ==========================================
# 작가 모드
# ==========================================
if mode == "작가 모드":
    st.markdown("### 사진 업로드 및 AI 분류")
    st.info("대회를 선택하고, 지도에서 촬영 위치를 클릭한 후, 사진을 업로드하세요.")
    
    tournaments = {
        "JTBC 마라톤": {"date": "2025년 11월 2일", "distance": "42.195km"},
        "춘천 마라톤": {"date": "2025년 10월 26일", "distance": "42.195km"}
    }
    
    selected_tournament = st.selectbox(
        "사진을 업로드할 대회를 선택하세요",
        options=["대회를 선택해주세요"] + list(tournaments.keys()),
        key="photographer_tournament"
    )
    
    if selected_tournament == "대회를 선택해주세요":
        st.info("먼저 대회를 선택해주세요")
    else:
        st.success(f"**{selected_tournament}** 선택됨")
        if selected_tournament not in st.session_state.saved_photos:
            st.session_state.saved_photos[selected_tournament] = []
        
        st.markdown("### 사진 촬영 위치 선택")
        coordinates = load_marathon_course(selected_tournament)
        if coordinates:
            m = create_clickable_course_map(coordinates)
            map_data = st_folium(m, width=700, height=500, key="photographer_map")
            if map_data and map_data.get('last_clicked'):
                lat, lon = map_data['last_clicked']['lat'], map_data['last_clicked']['lng']
                st.session_state.selected_location = {'lat': lat, 'lon': lon}
                st.success(f"선택된 위치: 위도 {lat:.6f}, 경도 {lon:.6f}")
        
        if st.session_state.selected_location:
            st.markdown("### 사진 업로드")
            uploaded_files = st.file_uploader(
                "사진을 선택하세요 (여러 장 가능)",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                key="photographer_upload"
            )
            if uploaded_files:
                st.success(f"{len(uploaded_files)}장 업로드됨")
                cols = st.columns(4)
                photo_data = []
                for idx, file in enumerate(uploaded_files):
                    with cols[idx % 4]:
                        image = Image.open(file)
                        st.image(image, use_container_width=True)
                        dt = get_exif_datetime(image)
                        if dt:
                            st.code(dt.strftime("%Y-%m-%d %H:%M:%S"))
                        else:
                            st.warning("시간 정보 없음")
                        exif_lat, exif_lon = get_gps_from_image(image)
                        use_lat = exif_lat if exif_lat else st.session_state.selected_location['lat']
                        use_lon = exif_lon if exif_lon else st.session_state.selected_location['lon']
                        manual_location = st.text_input("위치 설명", key=f"loc_{idx}", placeholder="예: 서울역 앞")
                        photo_data.append({
                            'image': image, 'name': file.name, 'location': manual_location,
                            'latitude': use_lat, 'longitude': use_lon, 'photo_datetime': dt,
                            'uploaded_file': file
                        })
                
                if st.button("DB에 저장하기", type="primary"):
                    progress = st.progress(0)
                    saved = 0
                    for i, photo in enumerate(photo_data):
                        st.write(f"처리 중... ({i+1}/{len(photo_data)})")
                        try:
                            embedding = st.session_state.image_finder.get_image_embedding(photo['image'])
                            img_bytes = io.BytesIO()
                            photo['image'].save(img_bytes, format='PNG')
                            photo.update({
                                'embedding': embedding,
                                'upload_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'tournament': selected_tournament,
                                'image_bytes': img_bytes.getvalue()
                            })
                            saved += 1
                        except Exception as e:
                            st.error(f"{photo['name']} 오류: {e}")
                        progress.progress((i + 1) / len(photo_data))
                    st.session_state.saved_photos[selected_tournament].extend(photo_data)
                    st.session_state.saved_count += saved
                    st.success(f"{saved}장 저장 완료!")
                    st.balloons()
                    st.session_state.selected_location = None
                    st.rerun()
        else:
            st.warning("지도에서 위치를 클릭해주세요")
        
        if st.session_state.saved_photos.get(selected_tournament):
            count = len(st.session_state.saved_photos[selected_tournament])
            time_count = sum(1 for p in st.session_state.saved_photos[selected_tournament] if p.get('photo_datetime'))
            st.markdown(f"### 저장된 사진: **{count}장**")
            st.info(f"촬영 시간 포함: {time_count}장")

# ==========================================
# 이용자 모드
# ==========================================
else:
    st.set_page_config(page_title="마라톤 사진 검색", page_icon="Runner", layout="wide", initial_sidebar_state="collapsed")
    st.markdown("""
    <style>
        .main { background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%); }
        .stButton>button { background: linear-gradient(90deg, #4a90e2 0%, #50e3c2 100%); color: white; font-weight: bold; padding: 15px; border-radius: 12px; border: none; width: 100%; }
        .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(74, 144, 226, 0.4); }
        .info-card { background: white; padding: 20px; border-radius: 12px; border-left: 4px solid #4a90e2; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 20px; }
        h1 { text-align: center; font-size: 48px; margin-bottom: 30px; }
    </style>
    """, unsafe_allow_html=True)

    for key in ['selected_tournament', 'uploaded_image', 'show_results']:
        if key not in st.session_state:
            st.session_state[key] = None if key != 'show_results' else False

    tournaments = {
        "JTBC 마라톤": {"date": "2025년 11월 2일", "distance": "42.195km", "course": "잠실→광화문→남산→한강→잠실", "icon": "Runner"},
        "춘천 마라톤": {"date": "2025년 10월 26일", "distance": "42.195km", "course": "의암호→소양강→춘천→의암호", "icon": "Mountain"}
    }

    if not st.session_state.show_results:
        st.title("High 러너스")
        st.caption("AI가 마라톤 코스에서 당신의 사진을 찾아드립니다")
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            selected = st.selectbox("대회 선택", options=["대회를 선택해주세요"] + list(tournaments.keys()))
            if selected != "대회를 선택해주세요":
                st.session_state.selected_tournament = selected
                uploaded = st.file_uploader("사진 업로드", type=['png', 'jpg', 'jpeg'])
                if uploaded:
                    st.session_state.uploaded_image = Image.open(uploaded)
                    if st.button("코스 및 사진 보기", type="primary"):
                        st.session_state.show_results = True
                        st.rerun()
            else:
                st.info("대회를 선택해주세요")
    else:
        info = tournaments[st.session_state.selected_tournament]
        st.markdown(f"<h1 style='text-align:center'>{info['icon']} {st.session_state.selected_tournament}</h1>", unsafe_allow_html=True)
        left, right = st.columns([6, 4])
        
        with left:
            st.markdown("### 마라톤 코스")
            st.markdown(f"""
            <div class="info-card">
                <p>일시: {info['date']}<br>거리: {info['distance']}<br>코스: {info['course']}</p>
            </div>
            """, unsafe_allow_html=True)
            coords = load_marathon_course(st.session_state.selected_tournament)
            if coords:
                m = create_course_map(coords)
                folium_static(m, width=1300, height=600)
            else:
                st.error("코스 데이터 없음")
        
        with right:
            if st.session_state.uploaded_image:
                st.markdown("#### 검색한 사진")
                st.image(st.session_state.uploaded_image, width=400)
                st.markdown("---")
                st.markdown("#### 검색 옵션")
                top_k = st.slider("결과 개수", 1, 20, 5)
                threshold = st.slider("최소 유사도 (%)", 0, 100, 70)
                if st.button("유사 사진 검색", type="primary"):
                    photos = st.session_state.saved_photos.get(st.session_state.selected_tournament, [])
                    if not photos:
                        st.warning("저장된 사진 없음")
                    else:
                        with st.spinner("검색 중..."):
                            query_emb = st.session_state.image_finder.get_image_embedding(st.session_state.uploaded_image)
                            results = []
                            for p in photos:
                                if 'embedding' not in p:
                                    continue
                                sim = cosine_similarity(query_emb, p['embedding'])[0][0] * 100
                                if sim >= threshold:
                                    results.append((p, sim))
                            results.sort(key=lambda x: x[1], reverse=True)
                            results = results[:top_k]
                            if not results:
                                st.warning("결과 없음")
                            else:
                                st.success(f"{len(results)}장 발견")
                                for i, (p, sim) in enumerate(results):
                                    c1, c2 = st.columns([1, 2])
                                    with c1:
                                        st.image(Image.open(io.BytesIO(p['image_bytes'])), use_container_width=True)
                                    with c2:
                                        st.markdown(f"**#{i+1}**")
                                        st.markdown(f"**위치:** {p.get('location', '미상')}")
                                        st.markdown(f"**파일:** {p['name']}")
                                        if p.get('photo_datetime'):
                                            st.caption(f"촬영: {p['photo_datetime'].strftime('%Y-%m-%d %H:%M:%S')}")
                                        if p.get('latitude'):
                                            st.caption(f"GPS: {p['latitude']:.6f}, {p['longitude']:.6f}")
                                        st.progress(sim / 100)
                                        st.caption(f"유사도: {sim:.2f}%")
                                        if 'upload_timestamp' in p:
                                            st.caption(f"업로드: {p['upload_timestamp']}")
                                    st.markdown("---")
            else:
                st.info("사진을 업로드해주세요")

    # 뒤로 가기
    if st.session_state.show_results:
        if st.button("처음으로 돌아가기"):
            st.session_state.show_results = False
            st.session_state.selected_tournament = None
            st.session_state.uploaded_image = None
            st.rerun()

st.markdown("---")
st.caption("Powered by CLIP | AI 기반 사진 검색")