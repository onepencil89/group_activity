# ==================================================
# 1. 기본 설정 및 초기화
# ==================================================
import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import io, base64, uuid, folium
from streamlit_folium import st_folium
from datetime import datetime

# 세션 초기화
if "photos" not in st.session_state:
    st.session_state["photos"] = {}  # 구조: {tournament: { (lat,lon): [사진들] }}

# CLIP 모델 불러오기
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, processor, device

model, processor, device = load_model()


# ==================================================
# 2. 유틸리티 함수
# ==================================================
def extract_exif_data(img):
    try:
        exif_data = img.getexif()
        return {Image.ExifTags.TAGS.get(k): v for k, v in exif_data.items() if k in Image.ExifTags.TAGS}
    except:
        return {}

def safe_parse_time(exif):
    t = exif.get("DateTime")
    if not t: return None
    try:
        return datetime.strptime(t, "%Y:%m:%d %H:%M:%S")
    except:
        return None

def get_image_embedding(img, model, processor, device):
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    emb /= emb.norm(p=2, dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten()

def cosine_similarity(a, b):
    return float((a @ b) / ((a**2).sum()**0.5 * (b**2).sum()**0.5))


# ==================================================
# 3. GPX 기반 지도 생성 함수
# ==================================================
import gpxpy

def load_gpx_coords(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            gpx = gpxpy.parse(f)
        coords = []
        for track in gpx.tracks:
            for seg in track.segments:
                for pt in seg.points:
                    coords.append((pt.latitude, pt.longitude))
        return coords
    except:
        return None

def create_map(coords, selected_latlon=None, saved_photos=None):
    m = folium.Map(location=coords[0], zoom_start=13)
    folium.PolyLine(coords, color="blue", weight=3).add_to(m)
    if selected_latlon:
        folium.Marker(selected_latlon, icon=folium.Icon(color="red", icon="camera")).add_to(m)
    if saved_photos:
        for (lat, lon), plist in saved_photos.items():
            folium.Marker(
                [lat, lon], popup=f"{len(plist)}장 저장됨",
                icon=folium.Icon(color="blue", icon="camera")
            ).add_to(m)
    return m


# ==================================================
# 4. 대회 목록
# ==================================================
tournaments = {
    "JTBC 마라톤": "../data/2025_JTBC.gpx",
    "춘천 마라톤": "../data/chuncheon_marathon.gpx",
}


# ==================================================
# 5. 모드 선택
# ==================================================
st.sidebar.title("모드 선택")
mode = st.sidebar.radio("선택하세요:", ["작가 모드", "이용자 모드"])


# ==================================================
# 6. 📸 작가 모드
# ==================================================
if mode == "작가 모드":
    st.header("📸 작가 모드")

    tournament = st.selectbox("대회 선택", list(tournaments.keys()))
    coords = load_gpx_coords(tournaments[tournament])
    if not coords:
        st.error("GPX 파일을 불러올 수 없습니다.")
        st.stop()

    # 이전 위치와 다른 경우 임시 업로드 리스트 초기화
    if "selected_location" not in st.session_state or st.session_state.selected_location != tournament:
        st.session_state.selected_location = tournament
        st.session_state.temp_upload = []

    # 지도 표시
    saved_photos = st.session_state["photos"].get(tournament, {})
    m = create_map(coords, saved_photos=saved_photos)
    map_data = st_folium(m, width=700, height=400, key="map_photographer")
    latlon = None
    if map_data and map_data.get("last_clicked"):
        latlon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        st.info(f"선택된 위치: {latlon}")

    # 업로드
    uploaded = st.file_uploader("사진 업로드", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if uploaded and latlon:
        if tournament not in st.session_state["photos"]:
            st.session_state["photos"][tournament] = {}
        if latlon not in st.session_state["photos"][tournament]:
            st.session_state["photos"][tournament][latlon] = []

        # 중복 체크
        existing_names = {p["name"] for p in st.session_state["photos"][tournament][latlon]}
        new_photos = []
        for f in uploaded:
            if f.name in existing_names:
                st.warning(f"{f.name} 이미 저장됨")
                continue
            img = Image.open(f).convert("RGB")
            exif = extract_exif_data(img)
            photo_time = safe_parse_time(exif)
            emb = get_image_embedding(img, model, processor, device)
            thumb = img.copy(); thumb.thumbnail((150,150))
            buf = io.BytesIO(); thumb.save(buf, format="JPEG"); thumb_b64 = base64.b64encode(buf.getvalue()).decode()
            new_photos.append({
                "id": uuid.uuid4().hex,
                "name": f.name,
                "lat": latlon[0],
                "lon": latlon[1],
                "time": photo_time,
                "embedding": emb,
                "thumb": thumb_b64,
                "bytes": f.getvalue(),
            })
        if new_photos:
            st.session_state["photos"][tournament][latlon].extend(new_photos)
            st.success(f"{len(new_photos)}장 업로드 완료")


# ==================================================
# 7. 🔍 이용자 모드
# ==================================================
elif mode == "이용자 모드":
    st.header("🔍 이용자 모드")
    if not st.session_state.get("photos"):
        st.warning("아직 등록된 대회가 없습니다.")
        st.stop()

    tournament = st.selectbox("대회 선택", list(st.session_state["photos"].keys()))
    coords = load_gpx_coords(tournaments[tournament])
    if not coords:
        st.error("GPX 파일을 불러올 수 없습니다.")
        st.stop()

    # 검색 이미지 업로드
    query_file = st.file_uploader("검색할 이미지 업로드", type=["jpg","jpeg","png"])
    if query_file:
        query_img = Image.open(query_file).convert("RGB")
        query_emb = get_image_embedding(query_img, model, processor, device)
        st.image(query_img, caption="검색 이미지", width=250)

        # 위치별 최대 유사도 사진만 선택
        markers = []
        for loc, plist in st.session_state["photos"][tournament].items():
            best_photo = max(plist, key=lambda p: cosine_similarity(query_emb, p["embedding"]))
            sim = cosine_similarity(query_emb, best_photo["embedding"])
            if sim >= 0.7:
                best_photo["similarity"] = sim
                markers.append(best_photo)

        # 지도 표시
        m = create_map(coords, saved_photos={ (p["lat"],p["lon"]):[p] for p in markers })
        st_folium(m, width=700, height=400)

        # 클릭 시 해당 위치 사진 모두 표시
        st.subheader("해당 위치 사진 목록 (유사도 0.7 이상)")
        for p in markers:
            st.image(base64.b64decode(p["thumb"]), caption=f"{p['name']} ({p['similarity']*100:.1f}%)", width=150)
            if st.button("구매/저장", key=p["id"]):
                st.success(f"{p['name']} 선택 완료. 다운로드 가능.")
