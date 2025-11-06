# 중복 저장 수정 & 챗봇기능 탑재

"""
마라톤 사진 검색 플랫폼 - GPX/CLIP 통합 버전 (최종 통합본)
기능: 작가(지도 클릭 위치 지정), 이용자(유사도 검색, 바둑판 목록, 선택적 다운로드)
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
import os
from datetime import datetime
import base64
import uuid
import zipfile
from dotenv import load_dotenv
from openai import OpenAI
import hashlib


load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")



def generate_photo_id(f, latlon, tournament):
    f.seek(0)
    content = f.read()
    f.seek(0)
    key = f"{f.name}_{round(latlon[0],4)}_{round(latlon[1],4)}_{tournament}_{len(content)}"
    return hashlib.md5(key.encode()).hexdigest()


# ==================================================
# Streamlit 세션 초기화
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
        "selected_tournament": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

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
# EXIF 파싱
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
# GPX 좌표 로드
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
# 이미지 임베딩
# ==================================================
def get_image_embedding(image, model, processor, device):
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    return emb.cpu().numpy().astype(np.float32)

# ==================================================
# 지도 생성 함수
# ==================================================
def create_course_map_with_photos(coords, photos):
    if not coords:
        return None
    center = [sum(c[0] for c in coords) / len(coords), sum(c[1] for c in coords) / len(coords)]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")
    folium.PolyLine(coords, color="#FF4444", weight=4).add_to(m)
    for p in photos:
        marker_color = 'blue'
        border_style = '2px solid #4a90e2'
        if p["similarity"] >= 90:
            marker_color = 'red'
            border_style = '4px solid red'
        elif p["similarity"] >= 80:
            marker_color = 'orange'
            border_style = '3px solid orange'
        popup_html = f"""
        <div style='width:250px; font-family:Arial;'>
            <img src='data:image/jpeg;base64,{p['thumb']}' style='width:100%; border-radius:8px; border:{border_style};'>
            <b>{p['name']}</b><br>
            📍 {round(p['lat'],4)}, {round(p['lon'],4)}<br>
            ⏰ {p['time'].strftime('%Y-%m-%d %H:%M:%S')}<br>
            🎯 유사도: <b style='color:{marker_color}'>{p['similarity']:.1f}%</b>
        </div>
        """
        icon_html = f"<div style='width:30px;height:30px;border-radius:50%;border:{border_style};background-image:url(data:image/jpeg;base64,{p['thumb']});background-size:cover;'></div>"
        folium.Marker(
            [p["lat"], p["lon"]],
            popup=folium.Popup(popup_html, max_width=250),
            icon=folium.DivIcon(html=icon_html)
        ).add_to(m)
    return m

# ==================================================
# 이미지 표시 함수
# ==================================================
def image_bytes_to_st_image(image_bytes, use_container_width=False):
    img = Image.open(io.BytesIO(image_bytes))
    st.image(img, use_container_width=use_container_width)

# ==================================================
# 대회 경로
# ==================================================
tournaments = {
    "JTBC 마라톤": "../data/2025_JTBC.gpx",
    "춘천 마라톤": "../data/chuncheon_marathon.gpx",
}

# ==================================================
# 메인 실행
# ==================================================
st.set_page_config(layout="wide")
mode = st.sidebar.radio("모드 선택", ["📸 작가 모드", "🔍 이용자 모드"], label_visibility="collapsed")
model, processor, device = load_clip_model()

# ==================================================
# 📸 작가 모드
# ==================================================
if mode == "📸 작가 모드":
    st.title("📸 작가 모드: 사진 등록")

    tournament = st.selectbox("대회 선택", list(tournaments.keys()))
    coords = load_gpx_coords(tournaments[tournament])
    
    if not coords:
        st.error("GPX 파일을 불러올 수 없습니다.")
        st.stop()
    
    m = folium.Map(location=coords[0], zoom_start=13)
    folium.PolyLine(coords, color="blue", weight=3).add_to(m)
    map_data = st_folium(m, width=700, height=500, key="map_photographer")

    latlon = None
    if map_data.get("last_clicked"):
        latlon = (map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"])
        st.info(f"선택된 위치: {latlon}")
    uploaded = st.file_uploader("사진 업로드", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded and latlon:
        if st.button("💾 사진 저장"):
            new_photos = 0
            for f in uploaded:
                img = Image.open(f).convert("RGB")
                exif = extract_exif_data(img)
                photo_time = safe_parse_time(exif)
                emb = get_image_embedding(img, model, processor, device)

                thumb = img.copy()
                thumb.thumbnail((150, 150))
                buf_thumb = io.BytesIO()
                thumb.save(buf_thumb, format="JPEG")
                thumb_b64 = base64.b64encode(buf_thumb.getvalue()).decode()

                buf_full = io.BytesIO()
                img.save(buf_full, format="JPEG")
                full_bytes = buf_full.getvalue()

                # 🔑 고유 해시 기반 ID
                photo_id = generate_photo_id(f, latlon, tournament)

                if any(p["id"] == photo_id for p in st.session_state["photos"]):
                    continue  # 중복이면 skip

                st.session_state["photos"].append({
                    "id": photo_id,
                    "name": f.name,
                    "lat": latlon[0],
                    "lon": latlon[1],
                    "tournament": tournament,
                    "time": photo_time,
                    "embedding": emb,
                    "thumb": thumb_b64,
                    "bytes": full_bytes,
                })
                new_photos += 1

            st.success(f"{new_photos}장 저장 완료.")

# ==================================================
# 🔍 이용자 모드
# ==================================================
else:
    # ---------------------------
    # 1️⃣ 초기 화면: 대회 선택 + 사진 업로드
    # ---------------------------
    if not st.session_state["show_results"]:
        st.title("🏃 High 러너스")
        st.caption("AI가 마라톤 코스에서 당신의 사진을 찾아드립니다.")
        st.markdown("---")

        selected = st.selectbox(
            "1️⃣ 참가한 마라톤 대회를 선택하세요",
            ["대회를 선택해주세요"] + list(tournaments.keys()),
            key="user_tournament_select"
        )

        # 대회 선택 후, 사진 업로드
        if selected != "대회를 선택해주세요":
            uploaded_file = st.file_uploader(
                "2️⃣ 본인 사진 업로드",
                type=["jpg", "jpeg", "png"],
                key="user_upload_photo"
            )

            if uploaded_file and st.button("🔍 유사 사진 찾기", type="primary"):
                st.session_state["uploaded_image"] = Image.open(uploaded_file).convert("RGB")
                st.session_state["selected_tournament"] = selected
                st.session_state["show_results"] = True
                st.session_state["show_detail_view"] = False
                st.session_state["selected_for_download"] = set()
                st.rerun()

        else:
            st.info("먼저 참가한 대회를 선택해주세요.")

    # ---------------------------
    # 2️⃣ 검색 결과 화면
    # ---------------------------
    else:
        tournament_name = st.session_state["selected_tournament"]
        coords = load_gpx_coords(tournaments[tournament_name])

        # 상단 네비게이션 (뒤로가기 / 타이틀)
        col1, col2 = st.columns([1, 9])
        with col1:
            if st.session_state["show_detail_view"]:
                if st.button("⬅️ 목록으로 돌아가기", type="secondary", key="back_to_list"):
                    st.session_state["show_detail_view"] = False
                    st.rerun()
            else:
                if st.button("◀️ 처음으로", type="secondary", key="back_to_home"):
                    st.session_state["show_results"] = False
                    st.rerun()

        with col2:
            st.markdown(f"<h2 style='text-align:center'>🏁 {tournament_name}</h2>", unsafe_allow_html=True)

        st.markdown("---")

        # ---------------------------
        # 3️⃣ 지도 및 유사사진 탐색
        # ---------------------------
        map_col, list_col = st.columns([5, 5])

        with map_col:
            st.markdown("### 🗺️ 마라톤 코스 및 사진 위치")

            # 쿼리 이미지 임베딩
            query_emb = get_image_embedding(st.session_state["uploaded_image"], model, processor, device)

            # 유사도 계산
            photo_markers = []
            for p in st.session_state["photos"]:
                if p["tournament"] != tournament_name:
                    continue
                sim = cosine_similarity(query_emb, p["embedding"])[0][0] * 100
                p["similarity"] = sim
                if sim >= 70:
                    photo_markers.append(p)

            # 유사도 높은 순 정렬
            photo_markers.sort(key=lambda x: x["similarity"], reverse=True)
            st.session_state["photo_markers"] = photo_markers

            # 결과 없을 시 안내
            if not photo_markers:
                st.warning("유사 사진을 찾지 못했습니다. 다른 사진으로 시도해보세요.")
            else:
                # 지도 표시
                m = create_course_map_with_photos(coords, photo_markers)
                st_folium(m, width=900, height=500, key="user_map_result")

        # ---------------------------
        # 4️⃣ 오른쪽 영역: 목록 or 상세보기
        # ---------------------------
        with list_col:
            if st.session_state["show_detail_view"]:
                # 상세보기
                sel_id = st.session_state["selected_photo_id"]
                photo = next((p for p in st.session_state["photo_markers"] if p["id"] == sel_id), None)

                if photo:
                    st.image(photo["bytes"], use_container_width=True)
                    st.markdown(f"**📅 촬영시간:** {photo['time'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.markdown(f"**📍 위치:** {round(photo['lat'],4)}, {round(photo['lon'],4)}")
                    st.metric("💰 고해상도 다운로드 가격", "₩5,000")

                    # 작가 정보 (임시 placeholder)
                    st.markdown("---")
                    st.markdown("### 👤 작가 정보")
                    st.markdown("**이름:** Marathon Photographer")
                    st.markdown("**소속:** High Runners Studio")
                    st.markdown("**연락:** contact@highrunners.ai")

                    # 다운로드 버튼 (단일)
                    if st.button("📥 사진 다운로드", type="primary", key=f"download_{sel_id}"):
                        st.success("다운로드 요청이 접수되었습니다. (기능 연결 예정)")
                else:
                    st.warning("사진 정보를 불러올 수 없습니다.")

            else:
                # 목록 보기
                st.markdown("#### 🎯 유사 사진 목록")
                if not photo_markers:
                    st.info("검색 결과가 없습니다.")
                else:
                    cols = st.columns(3)
                    for i, p in enumerate(photo_markers):
                        with cols[i % 3]:
                            st.image(p["bytes"], use_container_width=True)
                            st.caption(f"유사도: {p['similarity']:.1f}%")

                            # 상세보기 버튼
                            if st.button("보기", key=f"view_{p['id']}"):
                                st.session_state["selected_photo_id"] = p["id"]
                                st.session_state["show_detail_view"] = True
                                st.rerun()

                            # 다운로드 선택 체크박스
                            if st.checkbox("다운로드 선택", key=f"chk_{p['id']}"):
                                st.session_state["selected_for_download"].add(p["id"])
                            else:
                                st.session_state["selected_for_download"].discard(p["id"])

                    # 선택된 사진들 다운로드
                    if st.session_state["selected_for_download"]:
                        st.info(f"선택된 {len(st.session_state['selected_for_download'])}장 다운로드 가능")
                        if st.button("📦 선택 사진 다운로드", type="primary", key="bulk_download"):
                            st.success("다운로드 요청이 접수되었습니다. (기능 연결 예정)")
