"""
마라톤 사진 검색 플랫폼 - GPX 통합 버전 (정리·수정 완료)
"""

import streamlit as st
from PIL import Image
import gpxpy, folium, base64, io
from streamlit_folium import folium_static
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta

st.set_page_config(layout="wide")

# ==========================================
# CLIP 모델 캐싱
# ==========================================
@st.cache_resource
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    return model, processor

# ==========================================
# ImageSimilarityFinder
# ==========================================
class ImageSimilarityFinder:
    def __init__(self):
        self.model, self.processor = load_clip_model()
        self.device = self.model.device

    def get_image_embedding(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.model.get_image_features(**inputs)
        return emb.cpu().numpy()

# ==========================================
# GPX & Map
# ==========================================
def load_marathon_course(name):
    files = {
        "JTBC 마라톤": "../data/2025_JTBC.gpx",
        "춘천 마라톤": "../data/chuncheon_marathon.gpx",
    }
    path = files.get(name)
    if not path:
        return None
    try:
        with open(path, "r") as f:
            gpx = gpxpy.parse(f)
        coords = [[p.latitude, p.longitude] for t in gpx.tracks for s in t.segments for p in s.points]
        return coords
    except FileNotFoundError:
        st.error(f"GPX 파일 없음: {path}")
        return None

def assign_photo_locations(n, coords, start_dt):
    if not coords:
        return []
    total = len(coords)
    locs = []
    for i in range(n):
        idx = int((i / n) * total) or 0
        lat, lon = coords[idx]
        km = (idx / total) * 42.195
        mins = int(km * 6)
        tm = start_dt + timedelta(minutes=mins)
        locs.append({
            "lat": lat, "lon": lon, "km": round(km, 2),
            "time": tm.strftime("%Y-%m-%d %H:%M:%S")
        })
    return locs

def create_course_map_with_photos(coords, markers):
    if not coords:
        return None
    center = [sum(c[0] for c in coords)/len(coords), sum(c[1] for c in coords)/len(coords)]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

    # 코스 라인
    folium.PolyLine(coords, color="#FF4444", weight=5, opacity=0.8).add_to(m)
    folium.Marker(coords[0],  popup="출발", icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(m)
    folium.Marker(coords[-1], popup="도착", icon=folium.Icon(color="red",   icon="stop", prefix="fa")).add_to(m)

    # 사진 마커
    for p in markers:
        img_b64 = p["image_base64"]
        sim = p["similarity"]
        border = "4px solid #FF0000" if sim >= 90 else "2px solid #FFA500" if sim >= 80 else "1px solid #4a90e2"
        icon_html = f"""<div style="width:30px;height:30px;border-radius:50%;overflow:hidden;
                              border:{border};background:url(data:image/png;base64,{img_b64}) center/cover;">
                        </div>"""
        tooltip_html = f"""<div style="width:150px;text-align:center;">
                            <img src="data:image/png;base64,{img_b64}" style="width:100%;border-radius:8px;border:{border};cursor:pointer;"
                                 onclick="window.open(this.src,'_blank','fullscreen=yes');">
                            <b>{p['name']}</b><br>{p['km']}km | <b style="color:red;">{sim:.1f}%</b>
                           </div>"""
        popup_html = f"""<div style="width:250px;">
                          <img src="data:image/png;base64,{img_b64}" style="width:100%;border-radius:8px;border:{border};">
                          <div style="background:#f0f7ff;padding:10px;border-radius:8px;">
                            <b>📸 {p['name']}</b><hr style="margin:4px 0;">
                            <small>📍 {p['km']}km<br>📅 {p['time']}<br>
                                   🎯 유사도 <b style="color:red;">{sim:.1f}%</b></small>
                            <button onclick="window.parent.postMessage(
                                {{type:'streamlit:setSessionState',key:'detailed_photo_id',value:'{p['id']}'}},'*');
                                window.parent.postMessage({{type:'streamlit:rerun'}},'*');"
                                style="margin-top:8px;width:100%;padding:8px;background:#4a90e2;color:#fff;border:none;border-radius:5px;cursor:pointer;">
                              🔍 상세 보기
                            </button>
                          </div>
                         </div>"""
        folium.Marker(
            [p["lat"], p["lon"]],
            icon=folium.DivIcon(html=icon_html, icon_size=(30,30), icon_anchor=(15,15)),
            tooltip=folium.Tooltip(tooltip_html, max_width=200),
            popup=folium.Popup(popup_html, max_width=270)
        ).add_to(m)
    return m

# ==========================================
# 세션 초기화
# ==========================================
def init():
    defaults = {
        "saved_photos": [], "finder": ImageSimilarityFinder(),
        "selected_tournament": None, "uploaded_image": None,
        "show_results": False, "detailed_photo_id": None,
        "show_detail_view": False, "selected_similar_photo_id": None,
        "photo_data": {}          # 작가 모드용 임시 위치
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
init()

# ==========================================
# CSS
# ==========================================
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #f5f7fa 0%, #fff 100%);}
    .stButton>button {background: linear-gradient(90deg, #4a90e2, #50e3c2);
                      color:#fff; font-weight:bold; padding:12px 24px; border-radius:12px;}
    .purchase-btn-style {background:#e35050; color:#fff; border:none; padding:12px;
                         border-radius:8px; width:100%; font-weight:bold; text-align:center;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 대회 정보
# ==========================================
tournaments = {
    "JTBC 마라톤": {"date":"2025년 11월 2일", "start":"08:00:00", "icon":"🏃‍♂️"},
    "춘천 마라톤": {"date":"2025년 10월 26일", "start":"09:00:00", "icon":"🏔️"}
}

mode = st.sidebar.radio("모드", ["📸 작가 모드", "🔍 이용자 모드"], label_visibility="collapsed")

# ==============================================================
# 📸 작가 모드
# ==============================================================
if mode == "📸 작가 모드":
    st.title("📸 작가 모드")
    tournament = st.selectbox("대회", list(tournaments.keys()))
    coords = load_marathon_course(tournament)

    uploaded = st.file_uploader(
        "사진 업로드 (최대 8장)",
        type=["png","jpg","jpeg"],
        accept_multiple_files=True,
        key="author_upload"
    )

    if uploaded and coords:
        # ---- 클릭으로 위치 지정 (옵션) ----
        if st.checkbox("지도 클릭으로 위치 지정"):
            m = folium.Map(location=[coords[len(coords)//2][0], coords[len(coords)//2][1]], zoom_start=12)
            folium.PolyLine(coords, color="#FF4444", weight=5).add_to(m)
            m.add_child(folium.LatLngPopup())
            clicked = folium_static(m, width=800, height=500)
            # 기존 (오류 나는 코드)
            # if clicked and clicked.get("last_clicked"):

            # 새로 바꾸기 (복사-붙여넣기)
            if st.session_state.get("last_clicked"):
                lat = st.session_state.last_clicked["lat"]
                lon = st.session_state.last_clicked["lng"]
                sel = st.selectbox("위치 지정할 사진", [f.name for f in uploaded])
                st.session_state.photo_data[sel] = {"lat": lat, "lon": lon}
                st.success(f"{sel} → ({lat:.5f}, {lon:.5f})")
                del st.session_state.last_clicked  # 다음 클릭 대비 초기화
                st.rerun()
                
        # ---- DB 저장 ----
        if st.button("💾 DB에 저장", type="primary"):
            start_dt = datetime.strptime(
                f"{tournaments[tournament]['date']} {tournaments[tournament]['start']}",
                "%Y년 %m월 %d일 %H:%M:%S"
            )
            locs = assign_photo_locations(len(uploaded[:8]), coords, start_dt)

            prog = st.progress(0)
            for i, (file, loc) in enumerate(zip(uploaded[:8], locs)):
                prog.progress((i+1)/len(uploaded[:8]))
                img = Image.open(file).convert("RGB")
                emb = st.session_state.finder.get_image_embedding(img)
                b64 = base64.b64encode(io.BytesIO().getvalue()).decode()
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()

                st.session_state.saved_photos.append({
                    "name": file.name,
                    "image_bytes": buf.getvalue(),
                    "image_base64": b64,
                    "embedding": emb,
                    "lat": loc["lat"], "lon": loc["lon"],
                    "km": loc["km"], "time": loc["time"],
                    "tournament": tournament,
                    "photographer": "작가",
                    "id": f"{tournament}_{file.name}"
                })
            st.success(f"🎉 {len(uploaded[:8])}장 저장 완료!")
            st.balloons()

# ==============================================================
# 🔍 이용자 모드
# ==============================================================
else:
    if not st.session_state.show_results:
        st.title("🏃 High 러너스")
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            sel = st.selectbox("대회 선택", ["선택"] + list(tournaments.keys()))
            if sel != "선택":
                st.session_state.selected_tournament = sel
                up = st.file_uploader("사진 업로드", type=["png","jpg","jpeg"])
                if up:
                    st.session_state.uploaded_image = Image.open(up).convert("RGB")
                    if st.button("🔍 검색 시작", type="primary"):
                        st.session_state.show_results = True
                        st.rerun()
    else:
        tn = st.session_state.selected_tournament
        coords = load_marathon_course(tn)
        # ---------- 헤더 ----------
        c1, c2 = st.columns([1, 9])
        with c1:
            if st.session_state.show_detail_view:
                if st.button("⬅️ 목록"):
                    st.session_state.show_detail_view = False
                    st.rerun()
            else:
                if st.button("◀️ 처음으로"):
                    st.session_state.show_results = False
                    st.rerun()
        with c2:
            st.markdown(f"<h1 style='text-align:center;'>{tournaments[tn]['icon']} {tn}</h1>", unsafe_allow_html=True)

        map_col, cont_col = st.columns([5, 5])

        # ---------- 지도 ----------
        with map_col:
            st.subheader("🗺️ 코스 & 사진 위치")
            markers = []
            if coords and st.session_state.uploaded_image:
                q_emb = st.session_state.finder.get_image_embedding(st.session_state.uploaded_image)
                for p in st.session_state.saved_photos:
                    if p["tournament"] != tn:
                        continue
                    sim = cosine_similarity(q_emb, p["embedding"])[0][0] * 100
                    if sim < 70:
                        continue
                    p = p.copy()
                    p["similarity"] = sim
                    markers.append(p)
                markers.sort(key=lambda x: x["similarity"], reverse=True)

                m = create_course_map_with_photos(coords, markers)
                if m:
                    st.success(f"✅ {len(markers)}장 발견")
                    folium_static(m, width=950, height=550)

        # ---------- 상세 / 리스트 ----------
        with cont_col:
            # 지도 마커 클릭 → 상세 보기
            if st.session_state.detailed_photo_id:
                st.session_state.selected_similar_photo_id = st.session_state.detailed_photo_id
                st.session_state.show_detail_view = True
                st.session_state.detailed_photo_id = None

            sel_id = st.session_state.selected_similar_photo_id
            sel_photo = next((p for p in markers if p["id"] == sel_id), None)

            if st.session_state.show_detail_view and sel_photo:
                st.subheader("✨ 선택 사진 상세")
                st.image(sel_photo["image_bytes"], use_container_width=True)
                st.markdown(f"**{sel_photo['km']}km** | **{sel_photo['time']}**")
                st.metric("가격", "5,000원")
                st.markdown(
                    '<a href="https://your-purchase-page.com" target="_blank">'
                    '<button class="purchase-btn-style">🛒 구매하기 (새 창)</button></a>',
                    unsafe_allow_html=True
                )
            else:
                st.subheader("🖼️ 내 사진")
                if st.session_state.uploaded_image:
                    st.image(st.session_state.uploaded_image, width=200)
                st.markdown("---")
                st.subheader("🎯 발견된 사진")
                if markers:
                    for p in markers:
                        c1, c2 = st.columns([1, 3])
                        with c1:
                            st.image(p["image_bytes"], width=80)
                        with c2:
                            st.write(f"**{p['km']}km**")
                            st.markdown(f"<span style='color:red;font-weight:bold'>{p['similarity']:.1f}%</span>", unsafe_allow_html=True)
                            if st.button("보기", key=p["id"]):
                                st.session_state.selected_similar_photo_id = p["id"]
                                st.session_state.show_detail_view = True
                                st.rerun()
                else:
                    st.info("아직 사진이 없습니다.")