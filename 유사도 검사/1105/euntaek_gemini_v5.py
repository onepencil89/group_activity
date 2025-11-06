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

load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 세션 상태 초기화
if 'chat_open' not in st.session_state:
    st.session_state.chat_open = False
if 'messages' not in st.session_state:
    st.session_state.messages = []

# API 호출 함수
def call_api(user_message):
    # Open API 인증 및 객체생성
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        completion = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {
                # 시스템 프롬프트
                    "role" : "system",
                    "content" : "너는 달리기 강습 전문가야."
                },
                # 사용자 프롬프트
                {
                    "role":"user",
                    "content" : user_message
                }
            ]
        )
        return completion.choices[0].message.content
     
    except Exception as e:
        return f"오류 발생: {str(e)}"
    

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
        existing_names = {p["name"] for p in st.session_state["photos"] if p["tournament"] == tournament}
        new_photos = []

        for f in uploaded:
            if f.name in existing_names:
                continue  # ✅ 중복 방지

            img = Image.open(f).convert("RGB")
            exif = extract_exif_data(img)
            photo_time = safe_parse_time(exif)
            emb = get_image_embedding(img, model, processor, device)

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
            st.session_state["photos"].extend(new_photos)
            st.success(f"{len(new_photos)}장 업로드 완료 (총 {len(st.session_state['photos'])}장 저장됨)")
        else:
            st.info("이미 저장된 파일이므로 건너뜁니다.")

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

        # 지도 + 결과 표시
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

        # 상세보기 or 목록
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


    st.markdown("---")
    col1, col2, col3 = st.columns([5, 10, 5])
    with col2:
        if st.button("달리기에 관해 무엇이든 물어보세요!😎", key="chatbot_btn", use_container_width=True):
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
