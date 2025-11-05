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
from datetime import datetime
import base64
import uuid
import zipfile

# ==================================================
# EXIF 안전 파싱
# ==================================================
def extract_exif_data(image):
    try:
        exif_data = {}
        raw_exif = image._getexif()
        if raw_exif:
            for tag, value in raw_exif.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                exif_data[decoded] = value
        return exif_data
    except Exception:
        return {}

def safe_parse_time(exif_data):
    try:
        time_str = exif_data.get("DateTime", None)
        if time_str:
            return datetime.strptime(time_str, "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return datetime.now()

# ==================================================
# GPX 로드
# ==================================================
def load_gpx_coords(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            gpx = gpxpy.parse(f)
        coords = []
        for track in gpx.tracks:
            for seg in track.segments:
                for point in seg.points:
                    coords.append((point.latitude, point.longitude))
        return coords
    except Exception:
        return None

# ==================================================
# CLIP 모델 로드
# ==================================================
@st.cache_resource
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    return model, processor, device

# ==================================================
# 이미지 임베딩
# ==================================================
def get_image_embedding(image, model, processor, device):
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    return emb.cpu().numpy()

# ==================================================
# 지도 생성 (사진 마커 포함)
# ==================================================
def create_course_map_with_photos(coords, photos):
    if not coords:
        return None
    m = folium.Map(location=coords[0], zoom_start=12, tiles="CartoDB positron")
    folium.PolyLine(coords, color="#FF4444", weight=4).add_to(m)
    for p in photos:
        html = f"<div style='text-align:center'><img src='data:image/jpeg;base64,{p['thumb']}' width='100'><br>{p['name']}<br>{p['similarity']:.1f}%</div>"
        folium.Marker([p["lat"], p["lon"]], popup=html).add_to(m)
    return m

# ==================================================
# 이미지 표시 함수
# ==================================================
def image_bytes_to_st_image(image_bytes, use_container_width=False):
    img = Image.open(io.BytesIO(image_bytes))
    st.image(img, use_container_width=use_container_width)

# ==================================================
# 세션 초기화
# ==================================================
def init_session():
    defaults = {
        "photos": [],
        "show_results": False,
        "show_detail_view": False,
        "selected_photo_id": None,
        "selected_for_download": set(),
        "uploaded_image": None,
        "photo_markers": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ==================================================
# 대회 정보
# ==================================================
tournaments = {
    "JTBC 마라톤": "../data/2025_JTBC.gpx",
    "춘천 마라톤": "../data/chuncheon_marathon.gpx",
}

# ==================================================
# 메인 로직
# ==================================================
mode = st.sidebar.radio("모드 선택", ["📸 작가 모드", "🔍 이용자 모드"], label_visibility="collapsed")
model, processor, device = load_clip_model()

# ==================================================
# 📸 작가 모드
# ==================================================
if mode == "📸 작가 모드":
    st.title("📸 작가 모드")

    tournament = st.selectbox("대회 선택", list(tournaments.keys()))
    coords = load_gpx_coords(tournaments[tournament])

    if not coords:
        st.error("GPX 파일을 불러올 수 없습니다.")
        st.stop()

    m = folium.Map(location=coords[0], zoom_start=13)
    folium.PolyLine(coords, color="blue", weight=3).add_to(m)
    map_data = st_folium(m, width=700, height=500)

    latlon = None
    if map_data.get("last_clicked"):
        latlon = (
            map_data["last_clicked"]["lat"],
            map_data["last_clicked"]["lng"]
        )
        st.info(f"선택된 위치: {latlon}")

    uploaded = st.file_uploader("사진 업로드", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded and latlon:
        for f in uploaded:
            img = Image.open(f).convert("RGB")
            exif = extract_exif_data(img)
            photo_time = safe_parse_time(exif)
            emb = get_image_embedding(img, model, processor, device)
            thumb = img.copy()
            thumb.thumbnail((150, 150))
            buf = io.BytesIO()
            thumb.save(buf, format="JPEG")
            thumb_b64 = base64.b64encode(buf.getvalue()).decode()

            st.session_state["photos"].append({
                "id": uuid.uuid4().hex,
                "name": f.name,
                "lat": latlon[0],
                "lon": latlon[1],
                "tournament": tournament,
                "time": photo_time,
                "embedding": emb,
                "thumb": thumb_b64,
                "bytes": f.getvalue(),
            })
        st.success(f"{len(uploaded)}장 업로드 완료")

# ==================================================
# 🔍 이용자 모드
# ==================================================
else:
    if not st.session_state["show_results"]:
        st.title("🏃 High 러너스 🏃")
        st.caption("AI가 마라톤 코스에서 당신의 사진을 찾아드립니다")
        st.markdown("---")

        selected = st.selectbox(
            "1️⃣ 참가한 마라톤 대회를 선택하세요",
            options=["대회를 선택해주세요"] + list(tournaments.keys()),
            key="tournament_selectbox"
        )

        if selected != "대회를 선택해주세요":
            st.session_state["selected_tournament"] = selected
            uploaded_file = st.file_uploader(
                "2️⃣ 본인 사진 업로드",
                type=["png", "jpg", "jpeg"],
                key="photo_uploader"
            )

            if uploaded_file and st.button("🔍 유사 사진 찾기", type="primary"):
                st.session_state["uploaded_image"] = Image.open(uploaded_file).convert("RGB")
                st.session_state["show_results"] = True
                st.session_state["show_detail_view"] = False
                st.session_state["selected_for_download"] = set()
                st.rerun()
        else:
            st.info("먼저 참가한 대회를 선택해주세요.")

    # ----------------------------------------------------
    # 검색 결과 페이지
    # ----------------------------------------------------
    else:
        tournament_name = st.session_state["selected_tournament"]
        coords = load_gpx_coords(tournaments[tournament_name])

        # 헤더
        col1, col2 = st.columns([1, 9])
        with col1:
            if st.session_state["show_detail_view"]:
                if st.button("⬅️ 목록으로", type="secondary"):
                    st.session_state["show_detail_view"] = False
                    st.rerun()
            else:
                if st.button("◀️ 처음으로", type="secondary"):
                    st.session_state["show_results"] = False
                    st.rerun()

        with col2:
            st.markdown(f"<h2 style='text-align:center'>🏁 {tournament_name}</h2>", unsafe_allow_html=True)

        st.markdown("---")

        # 1️⃣ 지도 영역
        map_col, list_col = st.columns([5, 5])
        with map_col:
            st.markdown("### 🗺️ 마라톤 코스 및 사진 위치")

            query_emb = get_image_embedding(st.session_state["uploaded_image"], model, processor, device)
            photo_markers = []
            for p in st.session_state["photos"]:
                if p["tournament"] != tournament_name:
                    continue
                sim = cosine_similarity(query_emb, p["embedding"])[0][0] * 100
                p["similarity"] = sim
                if sim >= 70:
                    photo_markers.append(p)
            photo_markers.sort(key=lambda x: x["similarity"], reverse=True)
            st.session_state["photo_markers"] = photo_markers

            if not photo_markers:
                st.warning("유사 사진을 찾지 못했습니다.")
            else:
                m = create_course_map_with_photos(coords, photo_markers)
                st_folium(m, width=900, height=500)

        # 2️⃣ 오른쪽: 목록 or 상세보기
        with list_col:
            if st.session_state["show_detail_view"]:
                sel_id = st.session_state["selected_photo_id"]
                photo = next((p for p in st.session_state["photo_markers"] if p["id"] == sel_id), None)
                if photo:
                    st.image(photo["bytes"], use_container_width=True)
                    st.markdown(f"**📅 촬영시간:** {photo['time']}")
                    st.markdown(f"**📍 위치:** {round(photo['lat'],4)}, {round(photo['lon'],4)}")
                    st.metric("가격", "5,000원", "고해상도 다운로드")
                else:
                    st.warning("사진 정보를 불러올 수 없습니다.")
            else:
                st.markdown("#### 🎯 유사 사진 목록")
                cols = st.columns(3)
                for i, p in enumerate(photo_markers):
                    with cols[i % 3]:
                        st.image(p["bytes"], use_container_width=True)
                        st.caption(f"유사도: {p['similarity']:.1f}%")
                        if st.button("보기", key=f"view_{uuid.uuid4().hex[:8]}"):
                            st.session_state["selected_photo_id"] = p["id"]
                            st.session_state["show_detail_view"] = True
                            st.rerun()
                        if st.checkbox("다운로드 선택", key=f"chk_{uuid.uuid4().hex[:8]}"):
                            st.session_state["selected_for_download"].add(p["id"])

                if st.session_state["selected_for_download"]:
                    st.info(f"선택된 {len(st.session_state['selected_for_download'])}장 다운로드 가능")
