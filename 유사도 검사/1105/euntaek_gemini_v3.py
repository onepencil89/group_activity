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
# GPX 코스 로드
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
# 거리 추정
# ==================================================
def estimate_km(lat, lon, coords):
    if not coords:
        return 0
    dists = [((lat - c[0])**2 + (lon - c[1])**2)**0.5 for c in coords]
    return round(dists.index(min(dists)) / 1000, 2)

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
# 이미지 임베딩 계산
# ==================================================
def get_image_embedding(image, model, processor, device):
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    return emb.cpu().numpy()

# ==================================================
# 지도 생성
# ==================================================
def create_map(coords, photo_markers=None):
    if not coords:
        return None
    m = folium.Map(location=coords[0], zoom_start=12, tiles="CartoDB positron")
    folium.PolyLine(coords, color="#FF4444", weight=4).add_to(m)
    if photo_markers:
        for photo in photo_markers:
            img = photo["thumb"]
            html = f"<div style='text-align:center'><img src='data:image/jpeg;base64,{img}' width='120'><br>{photo['name']}<br>{photo['similarity']:.1f}%</div>"
            folium.Marker(
                [photo["lat"], photo["lon"]],
                popup=folium.Popup(html, max_width=250),
                tooltip=f"{photo['similarity']:.1f}%"
            ).add_to(m)
    return m

# ==================================================
# 세션 초기화
# ==================================================
if "photos" not in st.session_state:
    st.session_state["photos"] = []
if "selected_photo" not in st.session_state:
    st.session_state["selected_photo"] = None

# ==================================================
# 마라톤 정보
# ==================================================
tournaments = {
    "JTBC 마라톤": "../data/2025_JTBC.gpx",
    "춘천 마라톤": "../data/chuncheon_marathon.gpx",
}

# ==================================================
# 모드 선택
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

# ==========================================
# 이용자 모드 (개선 안정형)
# ==========================================
else:
    # 페이지 1️⃣ : 대회 선택 + 사진 업로드
    if not st.session_state.get("show_results", False):
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
                    type=["png", "jpg", "jpeg"],
                    key="photo_uploader"
                )

                if uploaded_file:
                    image = Image.open(uploaded_file).convert("RGB")
                    st.session_state.uploaded_image = image
                    if st.button("🔍 코스 및 추천 사진 보기", type="primary"):
                        st.session_state.show_results = True
                        st.session_state.show_detail_view = False
                        st.session_state.selected_for_download = set()
                        st.session_state.selected_similar_photo_id = None
                        st.rerun()
                else:
                    st.info("👆 대회 선택 후, 검색할 사진을 올려주세요.")
            else:
                st.info("👆 먼저 참가한 대회를 선택해주세요.")

    # 페이지 2️⃣ : 결과 / 상세보기
    else:
        tournament_name = st.session_state.selected_tournament
        tournament_info = tournaments[tournament_name]
        coordinates = load_gpx_coords(tournament_name)

        header_col1, header_col2 = st.columns([1, 9])
        with header_col1:
            if st.session_state.show_detail_view:
                if st.button("⬅️ 유사 사진 목록으로", type="secondary", key="back_to_list"):
                    st.session_state.show_detail_view = False
                    st.session_state.selected_similar_photo_id = None
                    st.rerun()
            else:
                if st.button("◀️ 처음으로", type="secondary", key="back_to_main"):
                    st.session_state.show_results = False
                    st.session_state.selected_tournament = None
                    st.session_state.uploaded_image = None
                    st.rerun()

        with header_col2:
            st.markdown(f"""
            <h1 style='text-align: center; color: #2c3e50;'>
                🏁 {tournament_name}
            </h1>
            """, unsafe_allow_html=True)

        st.markdown("---")

        map_col, content_col = st.columns([5, 5])

        # ----------------------------------------------------
        # 1️⃣ 지도 영역
        # ----------------------------------------------------
        with map_col:
            st.markdown("### 🗺️ 마라톤 코스 및 발견된 사진 위치")

            if coordinates and st.session_state.uploaded_image:
                with st.spinner("🤖 유사한 사진을 검색하고 있습니다..."):
                    try:
                        query_embedding = st.session_state.image_finder.get_image_embedding(
                            st.session_state.uploaded_image
                        )

                        photo_markers = []
                        for saved_photo in st.session_state.saved_photos:
                            if saved_photo["tournament"] != tournament_name:
                                continue

                            sim = cosine_similarity(
                                query_embedding,
                                saved_photo["embedding"]
                            )[0][0] * 100
                            saved_photo["similarity"] = sim
                            saved_photo["id"] = f"{saved_photo['tournament']}_{saved_photo['name']}"

                            if sim >= 70:
                                photo_markers.append(saved_photo)

                        photo_markers.sort(key=lambda x: x["similarity"], reverse=True)
                        st.session_state.photo_markers = photo_markers

                        if not photo_markers:
                            st.warning("유사한 사진을 찾지 못했습니다.")
                        else:
                            st.success(f"✅ {len(photo_markers)}개의 유사 사진을 찾았습니다.")
                            map_obj = create_course_map_with_photos(coordinates, photo_markers)
                            st_folium(map_obj, width=950, height=500)
                    except Exception as e:
                        st.error(f"❌ 오류 발생: {str(e)}")
            else:
                st.info("사진 업로드 후 결과를 확인하세요.")

        # ----------------------------------------------------
        # 2️⃣ 오른쪽 콘텐츠 영역
        # ----------------------------------------------------
        with content_col:
            if st.session_state.show_detail_view:
                selected_id = st.session_state.selected_similar_photo_id
                selected_photo = next((p for p in st.session_state.photo_markers if p["id"] == selected_id), None)

                if selected_photo:
                    st.markdown("#### ✨ 선택된 이미지 상세보기")
                    st.markdown("---")

                    image_bytes_to_st_image(selected_photo["image_bytes"], use_container_width=True)
                    st.markdown("---")
                    st.markdown(f"**📍 위치:** {selected_photo.get('km', 0)} km 지점")
                    st.markdown(f"**📅 촬영시간:** {selected_photo.get('time', '정보 없음')}")

                    st.metric("가격", "5,000원", "고해상도 다운로드")
                    st.markdown(
                        """
                        <a href="https://share.streamlit.io/simple-purchase-page" target="_blank">
                        <button class="purchase-btn-style">🛒 구매하기</button>
                        </a>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("선택된 사진 정보를 불러올 수 없습니다.")

            else:
                st.markdown("#### 🎯 유사한 사진 목록")
                uploaded_img = st.session_state.uploaded_image
                if uploaded_img:
                    st.image(uploaded_img, width=220)
                st.markdown("---")

                photo_markers = st.session_state.get("photo_markers", [])
                if not photo_markers:
                    st.info("유사 사진이 없습니다.")
                else:
                    cols = st.columns(3)
                    for idx, photo in enumerate(photo_markers):
                        with cols[idx % 3]:
                            image_bytes_to_st_image(photo["image_bytes"], use_container_width=True)
                            st.caption(
                                f"📍 {photo.get('km', 0)}km | 유사도: <b>{photo['similarity']:.1f}%</b>",
                                unsafe_allow_html=True
                            )

                            # 고유 키 생성으로 key 충돌 방지
                            view_key = f"view_btn_{uuid.uuid4().hex[:8]}"
                            select_key = f"select_chk_{uuid.uuid4().hex[:8]}"

                            if st.button("보기", key=view_key, use_container_width=True):
                                st.session_state.selected_similar_photo_id = photo["id"]
                                st.session_state.show_detail_view = True
                                st.rerun()

                            if st.checkbox(
                                "다운로드 선택",
                                value=photo["id"] in st.session_state.selected_for_download,
                                key=select_key
                            ):
                                st.session_state.selected_for_download.add(photo["id"])
                            else:
                                st.session_state.selected_for_download.discard(photo["id"])

                    if st.session_state.selected_for_download:
                        st.info(f"선택된 {len(st.session_state.selected_for_download)}장 다운로드 가능")
                    else:
                        st.info("다운로드할 사진을 선택하세요.")
