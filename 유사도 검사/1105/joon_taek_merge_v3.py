# 중복 저장 수정 & 챗봇기능 탑재

"""
마라톤 사진 검색 플랫폼 - GPX/CLIP 통합 버전 (최종 통합본)
기능: 작가(지도 클릭 위치 지정), 이용자(유사도 검색, 바둑판 목록, 선택적 다운로드)
"""

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
    st.session_state["photos"] = {}  # 구조 변경: {tournament: { (lat,lon): [사진들] }}

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
    """사진에서 EXIF 메타데이터 추출"""
    try:
        exif_data = img.getexif()
        return {Image.ExifTags.TAGS.get(k): v for k, v in exif_data.items() if k in Image.ExifTags.TAGS}
    except Exception:
        return {}

def safe_parse_time(exif):
    """EXIF에서 시간 정보 안전하게 파싱"""
    t = exif.get("DateTime")
    if not t:
        return None
    try:
        return datetime.strptime(t, "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None

def get_image_embedding(img, model, processor, device):
    """이미지를 벡터로 임베딩"""
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    emb /= emb.norm(p=2, dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten()

def cosine_similarity(a, b):
    """코사인 유사도 계산"""
    return float((a @ b) / ((a**2).sum()**0.5 * (b**2).sum()**0.5))


# ==================================================
# 3. 작가 모드 - 위치별 저장 + 지도 마커 표시
# ==================================================
st.sidebar.title("모드 선택")
mode = st.sidebar.radio("선택하세요:", ["작가 모드", "이용자 모드"])

if mode == "작가 모드":
    st.header("📸 작가 모드 - 위치별 저장 및 마커 표시")

    tournament = st.text_input("대회 이름을 입력하세요:")
    if not tournament:
        st.warning("대회명을 입력해주세요.")
        st.stop()

    # 지도 표시 및 위치 선택
    st.subheader("📍 촬영 위치 지정")
    m = folium.Map(location=[37.5665, 126.9780], zoom_start=12)
    
    # 저장된 위치별 마커 표시
    if tournament in st.session_state["photos"]:
        for (lat, lon), photo_list in st.session_state["photos"][tournament].items():
            folium.Marker(
                location=[lat, lon],
                popup=f"{len(photo_list)}장 저장됨",
                icon=folium.Icon(color="blue", icon="camera")
            ).add_to(m)

    map_data = st_folium(m, height=350, width=700)
    latlon = None
    if map_data and map_data.get("last_clicked"):
        latlon = (
            map_data["last_clicked"]["lat"],
            map_data["last_clicked"]["lng"]
        )
        st.info(f"선택된 위치: {latlon}")

    # 이미지 업로드 및 저장
    uploaded = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded and latlon:
        if tournament not in st.session_state["photos"]:
            st.session_state["photos"][tournament] = {}
        if latlon not in st.session_state["photos"][tournament]:
            st.session_state["photos"][tournament][latlon] = []

        existing_names = {p["name"] for p in st.session_state["photos"][tournament][latlon]}
        new_photos = []

        for f in uploaded:
            if f.name in existing_names:
                continue

            img = Image.open(f).convert("RGB")
            exif = extract_exif_data(img)
            photo_time = safe_parse_time(exif)
            emb = get_image_embedding(img, model, processor, device)

            # 썸네일 변환
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
            st.success(f"{len(new_photos)}장 업로드 완료 (현재 위치 총 {len(st.session_state['photos'][tournament][latlon])}장)")
        else:
            st.info("이미 저장된 파일은 건너뜁니다.")

    elif not latlon:
        st.warning("지도를 클릭해 위치를 먼저 지정해주세요.")


# ==================================================
# 4. 이용자 모드 - 유사도 검색 + 상세보기 + 지도 표시
# ==================================================
elif mode == "이용자 모드":
    st.header("🔍 이용자 모드 - 유사사진 검색 및 상세보기")

    # 대회 선택
    if not st.session_state["photos"]:
        st.warning("아직 등록된 대회가 없습니다. 작가 모드에서 사진을 추가하세요.")
        st.stop()

    tournament = st.selectbox("대회를 선택하세요", list(st.session_state["photos"].keys()))

    # 검색 이미지 업로드
    query_img_file = st.file_uploader("검색할 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

    if query_img_file:
        query_img = Image.open(query_img_file).convert("RGB")
        query_emb = get_image_embedding(query_img, model, processor, device)

        st.image(query_img, caption="검색 이미지", width=250)

        # 유사도 계산
        all_photos = []
        for loc, photos in st.session_state["photos"][tournament].items():
            for p in photos:
                sim = cosine_similarity(query_emb, p["embedding"])
                all_photos.append((p, sim))

        if not all_photos:
            st.info("해당 대회에 저장된 사진이 없습니다.")
            st.stop()

        sorted_photos = sorted(all_photos, key=lambda x: x[1], reverse=True)
        st.subheader("🔎 유사 사진 결과 (상위 5장)")
        cols = st.columns(5)

        for i, (p, sim) in enumerate(sorted_photos[:5]):
            with cols[i % 5]:
                st.image(base64.b64decode(p["thumb"]), caption=f"{p['name']} ({sim*100:.1f}%)", width=150)
                if st.button("📄 상세보기", key=p["id"]):
                    st.session_state["selected_photo"] = p

    # 상세보기 영역
    if "selected_photo" in st.session_state:
        p = st.session_state["selected_photo"]
        st.subheader("📋 사진 상세 정보")
        st.image(base64.b64decode(p["thumb"]), width=300)
        st.write(f"**파일명:** {p['name']}")
        st.write(f"**촬영 위치:** {p['lat']:.5f}, {p['lon']:.5f}")
        st.write(f"**촬영 시각:** {p['time'] if p['time'] else '정보 없음'}")

        # 지도 표시
        st.subheader("📍 촬영 위치 지도")
        m = folium.Map(location=[p["lat"], p["lon"]], zoom_start=15)
        folium.Marker(location=[p["lat"], p["lon"]], popup=p["name"], icon=folium.Icon(color="green")).add_to(m)
        st_folium(m, height=300, width=700)
