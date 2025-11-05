"""
마라톤 사진 검색 플랫폼 - GPX 통합 버전 (오류 수정 최종안)
- exif undefined 오류 해결
- folium_static → st_folium 교체
- 세션 및 변수명 정리
"""

import streamlit as st
from PIL import Image, ExifTags
import gpxpy
import folium
from streamlit_folium import st_folium
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import io
import zipfile
from datetime import datetime, timedelta
import base64
import random

# ==================================================
# 🕒 EXIF 데이터 안전 파싱
# ==================================================
def extract_exif_data(image):
    """EXIF 데이터를 안전하게 추출"""
    exif_data = {}
    try:
        raw_exif = image._getexif()
        if raw_exif:
            for tag, value in raw_exif.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                exif_data[tag_name] = value
    except Exception:
        pass
    return exif_data


def safe_parse_time(time_str):
    """EXIF의 DateTime 포맷을 datetime 객체로 변환"""
    try:
        return datetime.strptime(time_str, "%Y:%m:%d %H:%M:%S")
    except Exception:
        return datetime.now()


# ==================================================
# ⚙️ Streamlit 초기 설정
# ==================================================
st.set_page_config(layout="wide", page_title="마라톤 사진 검색 플랫폼")

# ==================================================
# 거리 추정 함수
# ==================================================
def estimate_km_from_gpx(lat, lon, course_points):
    if not course_points:
        return 0.0
    min_dist = float('inf')
    km_point = 0
    for i, (clat, clon) in enumerate(course_points):
        dist = ((lat - clat)**2 + (lon - clon)**2)**0.5
        if dist < min_dist:
            min_dist = dist
            km_point = i / 1000
    return round(km_point, 2)


# ==================================================
# CLIP 모델 로드
# ==================================================
@st.cache_resource
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    return model, processor


class ImageSimilarityFinder:
    def __init__(self):
        self.model, self.processor = load_clip_model()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_image_embedding(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embedding = self.model.get_image_features(**inputs)
        return embedding.cpu().numpy()


# ==================================================
# 세션 초기화
# ==================================================
def initialize_session_state():
    defaults = {
        'saved_photos': [],
        'image_finder': ImageSimilarityFinder(),
        'selected_tournament': None,
        'uploaded_image': None,
        'show_results': False,
        'detailed_photo_id': None,
        'selected_similar_photo_id': None,
        'show_detail_view': False,
        'selected_for_download': set(),
        'selected_latlon': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()

# ==================================================
# GPX 관련
# ==================================================
def load_marathon_course(tournament_name):
    gpx_files = {
        "JTBC 마라톤": "../data/2025_JTBC.gpx",
        "춘천 마라톤": "../data/chuncheon_marathon.gpx",
    }
    if tournament_name in gpx_files:
        try:
            with open(gpx_files[tournament_name], "r") as f:
                gpx = gpxpy.parse(f)
            coords = []
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        coords.append([point.latitude, point.longitude])
            return coords
        except FileNotFoundError:
            st.error(f"GPX 파일이 존재하지 않습니다: {gpx_files[tournament_name]}")
    return None


# ==================================================
# 지도 생성
# ==================================================
def create_course_map_with_photos(coordinates, photo_markers=None):
    if not coordinates:
        return None

    center_lat = np.mean([c[0] for c in coordinates])
    center_lon = np.mean([c[1] for c in coordinates])
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")

    folium.PolyLine(coordinates, color="#FF4444", weight=5).add_to(m)
    folium.Marker(coordinates[0], popup="🏁 출발", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(coordinates[-1], popup="🎯 도착", icon=folium.Icon(color="red")).add_to(m)

    if photo_markers:
        for photo in photo_markers:
            img_b64 = photo.get("thumb_base64", "")
            html = f"""
            <div style='text-align:center'>
                <img src='data:image/jpeg;base64,{img_b64}' width='100'><br>
                <b>{photo['name']}</b><br>{photo['km']}km | {photo['similarity']:.1f}%
            </div>
            """
            folium.Marker(
                [photo['lat'], photo['lon']],
                popup=folium.Popup(html, max_width=250),
                tooltip=f"{photo['similarity']:.1f}%"
            ).add_to(m)
    return m


# ==================================================
# 🧭 모드 선택
# ==================================================
mode = st.sidebar.radio("모드 선택", ["📸 작가 모드", "🔍 이용자 모드"], label_visibility="collapsed")

tournaments = {
    "JTBC 마라톤": {"date": "2025-11-02", "icon": "🏃‍♂️"},
    "춘천 마라톤": {"date": "2025-10-26", "icon": "🏔️"},
}

# ==================================================
# 📸 작가 모드
# ==================================================
if mode == "📸 작가 모드":
    st.title("📸 작가 모드")
    selected_tournament = st.selectbox("대회 선택", list(tournaments.keys()))

    coords = load_marathon_course(selected_tournament)
    if not coords:
        st.stop()

    m = folium.Map(location=coords[0], zoom_start=13)
    folium.PolyLine(coords, color="blue", weight=3).add_to(m)
    map_data = st_folium(m, width=700, height=500)

    if map_data and map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]
        st.session_state.selected_latlon = (lat, lon)
        st.success(f"📍 선택된 위치: {lat:.6f}, {lon:.6f}")

    uploaded_files = st.file_uploader("사진 업로드", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_files and st.session_state.selected_latlon:
        lat, lon = st.session_state.selected_latlon
        for file in uploaded_files:
            try:
                image = Image.open(file)
                exif_data = extract_exif_data(image)
                photo_time = safe_parse_time(exif_data.get("DateTime", ""))
                embedding = st.session_state.image_finder.get_image_embedding(image)
                thumbnail = image.copy()
                thumbnail.thumbnail((200, 200))
                thumb_b64 = base64.b64encode(io.BytesIO().getbuffer()).decode()
            except Exception as e:
                st.error(f"{file.name} 처리 실패: {e}")
                continue

            st.session_state.saved_photos.append({
                'tournament': selected_tournament,
                'name': file.name,
                'lat': lat,
                'lon': lon,
                'image_bytes': file.getvalue(),
                'embedding': embedding.tolist(),
                'thumb_base64': thumb_b64,
                'time': photo_time.strftime("%Y-%m-%d %H:%M:%S"),
            })
        st.success(f"{len(uploaded_files)}장 사진 저장 완료!")

# ==================================================
# 🔍 이용자 모드
# ==================================================
else:
    st.title("🔍 이용자 모드")

    tournament = st.selectbox("대회 선택", list(tournaments.keys()))
    coords = load_gpx_coords(tournaments[tournament])
    uploaded_file = st.file_uploader("내 사진 업로드", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        query_image = Image.open(uploaded_file).convert("RGB")
        st.image(query_image, caption="내 사진", width=250)

        if st.button("유사 사진 검색"):
            query_emb = get_image_embedding(query_image, model, processor, device)
            results = []
            for p in st.session_state["photos"]:
                if p["tournament"] != tournament:
                    continue
                sim = cosine_similarity(query_emb, p["embedding"])[0][0] * 100
                p["similarity"] = sim
                p["km"] = estimate_km(p["lat"], p["lon"], coords)
                if sim > 70:
                    results.append(p)
            results.sort(key=lambda x: x["similarity"], reverse=True)

            if not results:
                st.warning("유사 사진이 없습니다.")
            else:
                st.subheader(f"🔎 {len(results)}개의 유사 사진 발견")
                cols = st.columns(4)
                for idx, photo in enumerate(results):
                    with cols[idx % 4]:
                        st.image(
                            base64.b64decode(photo["thumb"]),
                            caption=f"{photo['name']} ({photo['similarity']:.1f}%)",
                            use_container_width=True
                        )
                        if st.button("보기", key=f"view_{photo['name']}_{idx}"):
                            st.session_state["selected_photo"] = photo

                if st.session_state["selected_photo"]:
                    p = st.session_state["selected_photo"]
                    st.divider()
                    st.subheader("📍 사진 상세 정보")
                    st.image(p["bytes"], caption=p["name"], width=400)
                    st.write(f"유사도: {p['similarity']:.1f}%")
                    st.write(f"위치: ({p['lat']:.6f}, {p['lon']:.6f}) / 약 {p['km']} km 지점")
                    m = create_map(coords, [p])
                    st_folium(m, width=800, height=400)

                    if st.download_button("사진 다운로드", data=p["bytes"], file_name=p["name"]):
                        st.success("다운로드 완료!")