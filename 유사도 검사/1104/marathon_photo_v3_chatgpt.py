import streamlit as st
from PIL import Image, ExifTags
import gpxpy, folium, io, os, torch
from streamlit_folium import st_folium, folium_static
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ----------------- 세션 초기화 -----------------
for key, default in {
    "chat_open": False, "messages": [],
    "selected_location": None, "saved_photos": {},
    "saved_count": 0, "image_finder": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ----------------- API 호출 -----------------
def call_api(msg):
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        res = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": "너는 달리기 강습 전문가야."},
                {"role": "user", "content": msg}
            ]
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"오류: {str(e)}"

# ----------------- EXIF 시간 / GPS 추출 -----------------
def get_exif_datetime(image):
    try:
        exif = image._getexif()
        if not exif:
            return None
        for tag, val in exif.items():
            tag_name = ExifTags.TAGS.get(tag, tag)
            if tag_name in ['DateTimeOriginal', 'DateTime', 'DateTimeDigitized']:
                try:
                    return datetime.strptime(str(val), "%Y:%m:%d %H:%M:%S")
                except:
                    continue
        return None
    except:
        return None

def get_gps_from_image(image):
    try:
        exif = image._getexif()
        if not exif:
            return None, None

        gps_info = {}
        for tag, val in exif.items():
            tag_name = ExifTags.TAGS.get(tag, tag)
            if tag_name == 'GPSInfo':
                for gps_tag in val:
                    gps_info[ExifTags.GPSTAGS.get(gps_tag, gps_tag)] = val[gps_tag]

        def to_deg(v):
            d, m, s = v
            return float(d) + float(m)/60 + float(s)/3600

        lat = lon = None
        if 'GPSLatitude' in gps_info and 'GPSLatitudeRef' in gps_info:
            lat = to_deg(gps_info['GPSLatitude'])
            if gps_info['GPSLatitudeRef'] == 'S': lat = -lat
        if 'GPSLongitude' in gps_info and 'GPSLongitudeRef' in gps_info:
            lon = to_deg(gps_info['GPSLongitude'])
            if gps_info['GPSLongitudeRef'] == 'W': lon = -lon
        return lat, lon
    except:
        return None, None

# ----------------- 이미지 임베딩 -----------------
class ImageSimilarityFinder:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @st.cache_resource
    def load_model(_self):
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.to(_self.device)
        return model, proc

    def get_image_embedding(self, image):
        if self.model is None or self.processor is None:
            self.model, self.processor = self.load_model()
        image = image.convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.model.get_image_features(**inputs)
        return emb.cpu().numpy()

if st.session_state.image_finder is None:
    st.session_state.image_finder = ImageSimilarityFinder()

# ----------------- GPX 로드 -----------------
def load_marathon_course(name):
    files = {
        "JTBC 마라톤": "../data/2025_JTBC.gpx",
        "춘천 마라톤": "../data/chuncheon_marathon.gpx",
    }
    if name not in files: return None
    try:
        with open(files[name], 'r') as f:
            gpx = gpxpy.parse(f)
        coords = [[p.latitude, p.longitude] for t in gpx.tracks for s in t.segments for p in s.points]
        return coords
    except:
        return None

# ----------------- 지도 생성 -----------------
def create_clickable_course_map(coords):
    if not coords: return None
    c_lat, c_lon = np.mean([c[0] for c in coords]), np.mean([c[1] for c in coords])
    m = folium.Map(location=[c_lat, c_lon], zoom_start=12, tiles='CartoDB positron')
    folium.PolyLine(coords, color='#FF4444', weight=5).add_to(m)
    folium.Marker(coords[0], popup='🏁 출발', icon=folium.Icon(color='green')).add_to(m)
    folium.Marker(coords[-1], popup='🎯 도착', icon=folium.Icon(color='red')).add_to(m)
    return m

def create_course_map(coords, photos=None):
    if not coords: return None
    c_lat, c_lon = np.mean([c[0] for c in coords]), np.mean([c[1] for c in coords])
    m = folium.Map(location=[c_lat, c_lon], zoom_start=12, tiles='CartoDB positron')
    folium.PolyLine(coords, color='#FF4444', weight=5).add_to(m)
    if photos:
        for p in photos:
            folium.Marker([p['lat'], p['lon']], icon=folium.Icon(color='orange', icon='camera')).add_to(m)
    return m

# ----------------- 페이지 모드 -----------------
mode = st.radio("모드 선택", ["📸 작가 모드", "🔍 이용자 모드"], label_visibility="collapsed")

# ----------------- 작가 모드 -----------------
if mode == "📸 작가 모드":
    st.title("📸 작가 모드")
    tournaments = {
        "JTBC 마라톤": {"date": "2025-11-02"},
        "춘천 마라톤": {"date": "2025-10-26"}
    }

    selected = st.selectbox("대회 선택", ["선택하세요"] + list(tournaments.keys()))
    if selected == "선택하세요": st.stop()

    coords = load_marathon_course(selected)
    if not coords: st.error("코스 데이터를 불러올 수 없습니다."); st.stop()
    m = create_clickable_course_map(coords)
    map_data = st_folium(m, width=700, height=500)
    if map_data and map_data.get('last_clicked'):
        lat, lon = map_data['last_clicked']['lat'], map_data['last_clicked']['lng']
        st.session_state.selected_location = {'lat': lat, 'lon': lon}
        st.success(f"📍 위도 {lat:.6f}, 경도 {lon:.6f}")

    if not st.session_state.selected_location: st.stop()
    files = st.file_uploader("사진 업로드", type=['png','jpg','jpeg'], accept_multiple_files=True)
    if not files: st.stop()

    photo_data = []
    for f in files:
        img = Image.open(f)
        dt = get_exif_datetime(img)
        lat, lon = get_gps_from_image(img)
        lat = lat or st.session_state.selected_location['lat']
        lon = lon or st.session_state.selected_location['lon']
        photo_data.append({
            'image': img, 'name': f.name, 'photo_datetime': dt,
            'latitude': lat, 'longitude': lon
        })
        st.image(img, caption=f.name, width=150)

    if st.button("💾 저장"):
        for p in photo_data:
            emb = st.session_state.image_finder.get_image_embedding(p['image'])
            p['embedding'] = emb
            p['tournament'] = selected
            img_bytes = io.BytesIO(); p['image'].save(img_bytes, format='PNG')
            p['image_bytes'] = img_bytes.getvalue()
        st.session_state.saved_photos.setdefault(selected, []).extend(photo_data)
        st.success(f"{len(photo_data)}장 저장 완료")

# ----------------- 이용자 모드 -----------------
else:
    st.title("🔍 이용자 모드")
    selected = st.selectbox("대회 선택", ["선택하세요", "JTBC 마라톤", "춘천 마라톤"])
    if selected == "선택하세요": st.stop()

    uploaded = st.file_uploader("본인 사진 업로드", type=['png','jpg','jpeg'])
    if not uploaded: st.stop()

    image = Image.open(uploaded)
    st.image(image, caption="검색할 사진", width=300)
    if st.button("검색"):
        if selected not in st.session_state.saved_photos:
            st.warning("저장된 사진이 없습니다."); st.stop()

        query_emb = st.session_state.image_finder.get_image_embedding(image)
        results = []
        for p in st.session_state.saved_photos[selected]:
            sim = cosine_similarity(query_emb, p['embedding'])[0][0] * 100
            results.append({'photo': p, 'similarity': sim})
        results.sort(key=lambda x: x['similarity'], reverse=True)
        st.markdown("---")
        for idx, r in enumerate(results[:5]):
            st.image(Image.open(io.BytesIO(r['photo']['image_bytes'])), caption=f"{idx+1}. {r['similarity']:.2f}% 유사")
