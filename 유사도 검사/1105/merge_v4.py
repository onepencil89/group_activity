# ==================================================
# 1. 기본 설정 및 초기화
# ==================================================
import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import io, base64, uuid, folium, gpxpy
from streamlit_folium import st_folium
from datetime import datetime
import numpy as np


# ==================================================
# 2. GPX 기반 대회 정보
# ==================================================
tournaments = {
    "JTBC 마라톤": "../data/2025_JTBC.gpx",
    "춘천 마라톤": "../data/chuncheon_marathon.gpx",
}


# ==================================================
# 3. 세션 초기화
# ==================================================
if "photos" not in st.session_state:
    # 구조: {"대회명": { (lat, lon): [사진목록] }}
    st.session_state["photos"] = {}


# ==================================================
# 4. 모델 로드
# ==================================================
@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, processor, device

model, processor, device = load_clip_model()


# ==================================================
# 5. 유틸리티 함수
# ==================================================
def extract_exif_data(img):
    """사진의 EXIF 메타데이터 추출"""
    try:
        exif_data = img.getexif()
        return {Image.ExifTags.TAGS.get(k): v for k, v in exif_data.items() if k in Image.ExifTags.TAGS}
    except Exception:
        return {}

def safe_parse_time(exif):
    """촬영 시간 파싱"""
    t = exif.get("DateTime")
    if not t:
        return None
    try:
        return datetime.strptime(t, "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None

def get_image_embedding(img, model, processor, device):
    """이미지 → CLIP 임베딩"""
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    emb /= emb.norm(p=2, dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten()

def cosine_similarity(a, b):
    """코사인 유사도"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def load_gpx_coords(file_path):
    """GPX 파일을 읽어 위도/경도 리스트 반환"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            gpx = gpxpy.parse(f)
        coords = [(p.latitude, p.longitude) for track in gpx.tracks for seg in track.segments for p in seg.points]
        return coords
    except Exception as e:
        st.error(f"GPX 파일 오류: {e}")
        return None

def create_map_with_course(coords, saved_locations=None):
    """GPX 코스 + 기존 저장 위치 마커를 함께 표시"""
    m = folium.Map(location=coords[0], zoom_start=13)
    folium.PolyLine(coords, color="blue", weight=3, opacity=0.8).add_to(m)
    if saved_locations:
        for (lat, lon), photo_list in saved_locations.items():
            folium.Marker(
                location=[lat, lon],
                popup=f"{len(photo_list)}장 저장됨",
                icon=folium.Icon(color="green", icon="camera")
            ).add_to(m)
    return m


# ==================================================
# 6. 메인 UI
# ==================================================
st.sidebar.title("모드 선택")
mode = st.sidebar.radio("선택하세요:", ["작가 모드", "이용자 모드"])


# ==================================================
# 7. 작가 모드: 대회별 위치 기반 업로드
# ==================================================
if mode == "작가 모드":
    st.header("📸 작가 모드 - 대회별 위치 지정 및 사진 등록")

    tournament = st.selectbox("대회를 선택하세요", list(tournaments.keys()))
    coords = load_gpx_coords(tournaments[tournament])
    if not coords:
        st.stop()

    # 지도 생성
    existing_data = st.session_state["photos"].get(tournament, {})
    m = create_map_with_course(coords, existing_data)
    map_data = st_folium(m, width=700, height=500, key="map_creator")

    latlon = None
    if map_data and map_data.get("last_clicked"):
        latlon = (
            map_data["last_clicked"]["lat"],
            map_data["last_clicked"]["lng"]
        )
        st.info(f"선택된 위치: {latlon}")

    # 이미지 업로드
    uploaded = st.file_uploader("📁 사진 업로드", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded and latlon:
        st.session_state["photos"].setdefault(tournament, {})
        st.session_state["photos"][tournament].setdefault(latlon, [])

        existing_names = {p["name"] for p in st.session_state["photos"][tournament][latlon]}
        new_photos = []

        for f in uploaded:
            if f.name in existing_names:
                continue
            img = Image.open(f).convert("RGB")
            exif = extract_exif_data(img)
            photo_time = safe_parse_time(exif)
            emb = get_image_embedding(img, model, processor, device)

            # 썸네일
            thumb = img.copy()
            thumb.thumbnail((150, 150))
            buf = io.BytesIO()
            thumb.save(buf, format="JPEG")
            thumb_b64 = base64.b64encode(buf.getvalue()).decode()

            new_photos.append({
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

        if new_photos:
            st.session_state["photos"][tournament][latlon].extend(new_photos)
            st.success(f"{len(new_photos)}장 업로드 완료 (총 {len(st.session_state['photos'][tournament][latlon])}장)")
        else:
            st.info("이미 저장된 파일은 제외됨.")
    elif not latlon:
        st.warning("지도에서 위치를 클릭하세요.")


# ==================================================
# 8. 이용자 모드: 유사도 검색 + 상세보기 + 지도 연동
# ==================================================
elif mode == "이용자 모드":
    st.header("🔍 이용자 모드 - 유사사진 검색 및 위치별 결과보기")

    # 대회 선택
    if not st.session_state["photos"]:
        st.warning("등록된 대회가 없습니다. 작가 모드에서 추가해주세요.")
        st.stop()

    tournament = st.selectbox("대회를 선택하세요", list(st.session_state["photos"].keys()))
    all_locations = st.session_state["photos"][tournament]

    # 검색 이미지 업로드
    query_file = st.file_uploader("🔎 검색할 사진을 업로드하세요", type=["jpg", "jpeg", "png"])

    if query_file:
        query_img = Image.open(query_file).convert("RGB")
        query_emb = get_image_embedding(query_img, model, processor, device)
        st.image(query_img, caption="검색 이미지", width=250)

        best_match = None
        best_sim = -1
        best_location = None
        location_results = {}

        # 위치별 최고 유사도 계산
        for loc, photos in all_locations.items():
            sims = [cosine_similarity(query_emb, p["embedding"]) for p in photos]
            max_sim = max(sims)
            location_results[loc] = max_sim
            if max_sim > best_sim:
                best_sim = max_sim
                best_match = photos[np.argmax(sims)]
                best_location = loc

        # 지도 표시 (가장 유사한 위치)
        coords = load_gpx_coords(tournaments[tournament])
        m = create_map_with_course(coords, all_locations)
        if best_location:
            folium.Marker(
                location=best_location,
                popup=f"가장 유사한 위치 (유사도 {best_sim*100:.1f}%)",
                icon=folium.Icon(color="red", icon="star")
            ).add_to(m)
        st_folium(m, width=700, height=500, key="map_user")

        # 0.7 이상 유사 사진 목록
        threshold = 0.7
        matched_photos = [p for loc, photos in all_locations.items() for p in photos if cosine_similarity(query_emb, p["embedding"]) >= threshold]
        matched_photos.sort(key=lambda p: cosine_similarity(query_emb, p["embedding"]), reverse=True)

        if matched_photos:
            st.subheader(f"📷 유사도 {threshold*100:.0f}% 이상 사진 ({len(matched_photos)}장)")
            for p in matched_photos:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(base64.b64decode(p["thumb"]), width=120)
                with col2:
                    st.write(f"**{p['name']}** ({p['tournament']})")
                    st.write(f"위치: {p['lat']:.5f}, {p['lon']:.5f}")
                    st.write(f"유사도: {cosine_similarity(query_emb, p['embedding'])*100:.1f}%")
                    if st.button(f"🛒 {p['name']} 구매하기", key=f"buy_{p['id']}"):
                        st.success(f"{p['name']} 구매 완료")
            st.download_button("📦 모든 유사 사진 저장", data=b"export dummy", file_name="similar_photos.zip")
        else:
            st.info("0.7 이상 유사 사진이 없습니다.")
