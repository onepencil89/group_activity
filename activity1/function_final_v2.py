"""
마라톤 사진 검색 플랫폼 - GPX/CLIP 통합 버전 (최종 통합본)
기능: 작가(지도 클릭 위치 지정), 이용자(유사도 검색, 바둑판 목록, 선택적 다운로드)
"""

from transformers import CLIPProcessor, CLIPModel
from openai import OpenAI
import folium
from PIL import Image, ExifTags
import gpxpy
import streamlit as st
from streamlit_folium import st_folium
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import io
from datetime import datetime, timedelta # timedelta는 시간 계산 호환을 위해 추가
import base64
import uuid
import zipfile
from dotenv import load_dotenv
import os

load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 세션 상태 초기화
if 'chat_open' not in st.session_state:
    st.session_state.chat_open = False
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'selected_location' not in st.session_state:
    st.session_state.selected_location = None

# ==================================================
# API 호출 함수
# ==================================================
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

# ==================================================
# ⚙️ Streamlit 초기 설정 및 CSS
# ==================================================
st.markdown("""
<style>
    /* st.image 기본 풀스크린 버튼 숨기기 */
    div.stImage > button {
        display: none !important;
    }
    /* 구매 버튼 스타일 */
    .purchase-btn-style {
        background-color: #e35050; 
        color: white; 
        border: none; 
        padding: 10px; 
        border-radius: 5px; 
        width: 100%; 
        font-weight: bold; 
        cursor: pointer; 
        height: 50px;
        text-align: center;
        display: block;
        line-height: 30px;
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)
st.set_page_config(layout="wide")

# ==================================================
# EXIF 안전 파싱 (작가 모드 사용)
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
    # image가 PIL Image 객체라고 가정 (작가 모드에서 변환 완료)
    inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(device) 
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    return emb.cpu().numpy()

# ==================================================
# 🖼️ 이미지 표시
# ==================================================
def image_bytes_to_st_image(image_bytes, **kwargs):
    """
    이미지 바이트 데이터를 st.image에 안전하게 표시합니다.
    """
    st.image(io.BytesIO(image_bytes), **kwargs)


# ==================================================
# 지도 생성 (사진 마커 포함) - 이용자 모드 디테일 복구
# ==================================================
def create_course_map_with_photos(coords, photos):
    if not coords:
        return None
        
    center = [sum(c[0] for c in coords) / len(coords), sum(c[1] for c in coords) / len(coords)]

    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")
    folium.PolyLine(coords, color="#FF4444", weight=4).add_to(m)
    
    for p in photos:
        similarity_percent = p["similarity"]
        
        # 유사도에 따른 테두리 색상 설정
        if similarity_percent >= 90:
            border_style = '4px solid #FF0000'
            marker_color = 'red'
        elif similarity_percent >= 80:
            border_style = '2px solid #FFA500'
            marker_color = 'orange'
        else:
            border_style = '1px solid #4a90e2'
            marker_color = 'blue'

        # 팝업 HTML (상세 보기 JS 트리거 포함)
        popup_html = f"""
        <div style='width: 250px; font-family: Arial;'>
            <img src='data:image/jpeg;base64,{p['thumb']}'  
                  style='width: 100%; border-radius: 8px; margin-bottom: 10px; border: {border_style};'>
            <div style='background: #f0f7ff; padding: 10px; border-radius: 8px;'>
                <b style='color: #2c3e50; font-size: 16px;'>📸 {p['name']}</b><br>
                <hr style='margin: 8px 0; border: none; border-top: 1px solid #ddd;'>
                <small style='color: #666;'>
                    📍 <b>위치:</b> {round(p['lat'], 4)}, {round(p['lon'], 4)}<br>
                    📅 <b>시간:</b> {p['time'].strftime('%Y-%m-%d %H:%M:%S')}<br>
                    🎯 <b>유사도:</b> <span style='color: {marker_color}; font-weight: bold;'>{p['similarity']:.1f}%</span>
                </small>
                <button id='detail_btn_{p['id']}' 
                        onclick="window.parent.postMessage({{
                            type: 'streamlit:setSessionState', 
                            key: 'selected_photo_id', 
                            value: '{p['id']}'
                        }}, '*'); window.parent.postMessage({{type: 'streamlit:setSessionState', key: 'show_detail_view', value: true}}, '*'); window.parent.postMessage({{type: 'streamlit:rerun'}}, '*')"
                        style='background-color: #4a90e2; color: white; border: none; padding: 10px; border-radius: 5px; width: 100%; cursor: pointer; margin-top: 10px;'>
                        🔍 상세 보기 및 구매
                </button>
            </div>
        </div>
        """
        
        # 썸네일 아이콘 (DivIcon)
        icon_html = f"""<div style="width: 30px; height: 30px; border-radius: 50%; overflow: hidden; border: {border_style}; box-shadow: 0 0 5px rgba(0,0,0,0.4); background-image: url('data:image/jpeg;base64,{p['thumb']}'); background-size: cover; background-position: center; cursor: pointer;"></div>"""
        custom_icon = folium.DivIcon(icon_size=(30, 30), icon_anchor=(15, 15), html=icon_html)
        
        folium.Marker(
            [p["lat"], p["lon"]], 
            popup=folium.Popup(popup_html, max_width=270),
            icon=custom_icon,
            tooltip=f"{p['similarity']:.1f}% 유사"
        ).add_to(m)
        
    return m

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
        "selected_tournament": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ==================================================
# 대회 정보
# ==================================================
tournaments = {
    "JTBC 마라톤": "data/2025_JTBC.gpx", # 경로 수정
    "춘천 마라톤": "data/chuncheon_marathon.gpx", # 경로 수정
}

# ==================================================
# 메인 로직
# ==================================================
mode = st.sidebar.radio("모드 선택", ["📸 작가 모드", "🔍 이용자 모드"], label_visibility="collapsed")
model, processor, device = load_clip_model()

# ==================================================
# 📸 작가 모드 - (통합된 새 로직)
# ==================================================
if mode == "📸 작가 모드":
    st.title("📸 작가 모드: 사진 등록")

    col_info, col_map = st.columns([1, 1])

    with col_info:
        tournament = st.selectbox("1️⃣ 대회 선택", list(tournaments.keys()))
        
        st.markdown("---")
        st.markdown("2️⃣ **위치 지정:** 아래 지도에서 사진을 촬영한 **지점**을 클릭하세요.")
        
        latlon = None
        if st.session_state.get("last_clicked_lat"):
            latlon = (
                st.session_state["last_clicked_lat"],
                st.session_state["last_clicked_lng"]
            )
            st.info(f"선택된 위치: 위도 {latlon[0]:.4f}, 경도 {latlon[1]:.4f}")
        else:
            st.warning("지도에서 위치를 클릭해주세요.")
            
    
    with col_map:
        coords = load_gpx_coords(tournaments[tournament])
        if not coords:
            st.error("GPX 파일을 불러올 수 없습니다.")
            st.stop()
            
        # 지도 생성 및 클릭 이벤트 처리
        m = folium.Map(location=coords[0], zoom_start=13)
        folium.PolyLine(coords, color="blue", weight=3).add_to(m)
        
        # 이전 클릭 마커 표시
        if latlon:
             folium.Marker(latlon, icon=folium.Icon(color='red', icon='camera', prefix='fa')).add_to(m)

        map_data = st_folium(m, width=700, height=500, key="photographer_map")
        
        # 맵 클릭 시 세션 상태에 위치 저장 (Streamlit 맵 클릭 처리)
        if map_data.get("last_clicked"):
            st.session_state["last_clicked_lat"] = map_data["last_clicked"]["lat"]
            st.session_state["last_clicked_lng"] = map_data["last_clicked"]["lng"]
            st.rerun() # 위치가 바뀌면 재실행하여 반영
    
    st.markdown("---")
    
    uploaded = st.file_uploader("3️⃣ 사진 업로드", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded and latlon:
        if st.button(f"💾 {len(uploaded)}장 DB에 저장하기", type="primary"):
            progress_bar = st.progress(0, text="AI 처리 및 저장 중...")
            
            for idx, f in enumerate(uploaded):
                img = Image.open(f).convert("RGB")
                exif = extract_exif_data(img)
                photo_time = safe_parse_time(exif)
                
                # 1. 임베딩 생성 (AI)
                emb = get_image_embedding(img, model, processor, device)
                
                # 2. 썸네일 생성 및 Base64 인코딩 (지도/목록 표시용)
                thumb = img.copy()
                thumb.thumbnail((150, 150))
                buf_thumb = io.BytesIO()
                thumb.save(buf_thumb, format="JPEG", quality=70) # 용량 최적화
                thumb_b64 = base64.b64encode(buf_thumb.getvalue()).decode()

                # 3. 원본 이미지 바이트 저장 (상세 보기/다운로드용)
                buf_full = io.BytesIO()
                img.save(buf_full, format="JPEG", quality=90)
                full_bytes = buf_full.getvalue()
                
                # 4. 세션에 저장
                st.session_state["photos"].append({
                    "id": uuid.uuid4().hex,
                    "name": f.name,
                    "lat": latlon[0],
                    "lon": latlon[1],
                    "tournament": tournament,
                    "time": photo_time,
                    "embedding": emb,
                    "thumb": thumb_b64, # 썸네일 Base64
                    "bytes": full_bytes, # 원본 바이트 데이터
                })
                progress_bar.progress((idx + 1) / len(uploaded), text=f"{f.name} 처리 완료")
                
            st.success(f"🎉 {len(uploaded)}장 업로드 및 AI 분석 완료!")
            progress_bar.empty()
            st.balloons()
            st.session_state["last_clicked_lat"] = None # 위치 초기화
            st.session_state["last_clicked_lng"] = None
            st.rerun()

# ==================================================
# 🔍 이용자 모드
# ==================================================
else:
    if not st.session_state["show_results"]:
        # 페이지 1: 대회 선택 + 사진 업로드 (생략)
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
            elif uploaded_file:
                 st.image(uploaded_file, caption="업로드된 사진", width=200) # 업로드 미리보기
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
                # 뒤로가기 버튼
                if st.button("⬅️ 목록으로", type="secondary"):
                    st.session_state["show_detail_view"] = False
                    st.session_state["selected_photo_id"] = None
                    st.rerun()
            else:
                if st.button("◀️ 처음으로", type="secondary"):
                    st.session_state["show_results"] = False
                    st.session_state["selected_tournament"] = None
                    st.session_state["uploaded_image"] = None
                    st.rerun()

        with col2:
            st.markdown(f"<h2 style='text-align:center'>🏁 {tournament_name}</h2>", unsafe_allow_html=True)

        st.markdown("---")

        map_col, content_col = st.columns([5, 5])
        
        # 1. 유사도 계산 및 마커 데이터 준비
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
        st.session_state["photo_markers"] = photo_markers # 세션 상태에 저장

        # ----------------------------------------------------
        # 1. 지도 영역
        # ----------------------------------------------------
        with map_col:
            st.markdown("### 🗺️ 마라톤 코스 및 사진 위치")

            if not photo_markers:
                st.warning("유사 사진을 찾지 못했습니다.")
            else:
                m = create_course_map_with_photos(coords, photo_markers)
                st_folium(m, width=900, height=500)

        # ----------------------------------------------------
        # 2. 오른쪽: 목록 or 상세보기
        # ----------------------------------------------------
        with content_col:
            
            # --- 상세 보기 화면 ---
            if st.session_state["show_detail_view"]:
                sel_id = st.session_state["selected_photo_id"]
                photo = next((p for p in st.session_state["photo_markers"] if p["id"] == sel_id), None)
                
                if photo:
                    st.markdown("#### ✨ 선택된 이미지 상세")
                    
                    # 이미지 표시
                    image_bytes_to_st_image(photo["bytes"], use_container_width=True)
                    st.markdown("---")
                    
                    # 위치 및 시간 정보
                    st.markdown("##### 📍 위치 및 시간 정보")
                    st.markdown(f"**📍 위치:** {round(photo['lat'], 4)}, {round(photo['lon'], 4)}")
                    st.markdown(f"**📅 시간:** {photo['time'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.markdown("---")
                    
                    # 작가 정보
                    st.markdown("##### 👤 촬영자 정보")
                    col_prof1, col_prof2 = st.columns([1, 3])
                    with col_prof1:
                        st.markdown("", unsafe_allow_html=True)
                    with col_prof2:
                        st.markdown(f"**{photo.get('photographer', '작가')}**")
                        st.caption("마라톤 전문 포토그래퍼")

                    st.markdown("---")
                    
                    # 구매 버튼 구현 
                    st.metric("가격", "5,000원", "고해상도 다운로드")
                    purchase_url = "https://share.streamlit.io/simple-purchase-page" 
                    st.markdown(f'<a href="{purchase_url}" target="_blank">'
                                f'<button class="purchase-btn-style">'
                                f'🛒 구매하기 (새 창 열림)'
                                f'</button></a>', unsafe_allow_html=True)
                else:
                    st.warning("사진 정보를 불러올 수 없습니다.")

            # --- 유사 사진 목록 화면 ---
            else:
                st.markdown("#### 🖼️ 검색한 사진")
                if st.session_state["uploaded_image"]:
                    st.image(st.session_state["uploaded_image"], width=200) 
                
                st.markdown("---")
                st.markdown("#### 🎯 유사한 사진 목록")

                # ----------------------------------------------------------------------------------
                # 선택적 다운로드 버튼
                # ----------------------------------------------------------------------------------
                if st.session_state["selected_for_download"]:
                    st.info(f"선택된 사진 {len(st.session_state['selected_for_download'])}장에 대해 다운로드 페이지를 열 수 있습니다.")
                    download_url = "https://share.streamlit.io/download-selection"
                    
                    st.markdown(f'<a href="{download_url}" target="_blank">'
                                f'<button class="purchase-btn-style" style="background-color: #50e3c2;">'
                                f'⬇️ 선택된 사진 다운로드 페이지 열기 (새 창)'
                                f'</button></a>', unsafe_allow_html=True)
                else:
                    st.info("다운로드/구매를 위해 사진을 선택해주세요. (각 사진 아래 체크박스 사용)")
                
                st.markdown("---")
                
                # 바둑판식 목록 표시 (3열)
                cols = st.columns(3)
                
                for i, p in enumerate(photo_markers):
                    with cols[i % 3]: 
                        
                        def set_selected_photo_and_show_detail(photo_id):
                            st.session_state["selected_photo_id"] = photo_id
                            st.session_state["show_detail_view"] = True 
                        
                        # 체크박스 상태 업데이트 함수 (깜빡임 제거)
                        def update_download_selection(photo_id):
                            if st.session_state[f"select_list_{photo_id}"]:
                                st.session_state["selected_for_download"].add(photo_id)
                            else:
                                st.session_state["selected_for_download"].discard(photo_id)

                        # 이미지 표시 (바둑판식)
                        image_bytes_to_st_image(p["bytes"], use_container_width=True) 

                        st.caption(f"📍 {p['time'].strftime('%H:%M')} | 유사도: **<span style='color:red;'>{p['similarity']:.1f}%</span>**", unsafe_allow_html=True)

                        col_view, col_select = st.columns([1, 4])

                        with col_view:
                            # '보기' 버튼 (상세 보기 전환)
                            if st.button("보기", key=f"list_btn_{p['id']}", help="클릭 시 상세 화면으로 이동", type="secondary", use_container_width=True):
                                set_selected_photo_and_show_detail(p["id"])
                                st.rerun()

                        with col_select:
                            # 체크박스 (선택 기능)
                            st.checkbox(
                                "저장 목록에 추가",
                                value=p["id"] in st.session_state["selected_for_download"],
                                key=f"select_list_{p['id']}",
                                on_change=update_download_selection,
                                args=(p["id"],)
                            )
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