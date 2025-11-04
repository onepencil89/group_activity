"""
마라톤 사진 검색 플랫폼
대회 선택 → 사진 업로드 → 새 화면에서 코스 지도 + 유사 사진 표시
"""

import streamlit as st
from PIL import Image
import gpxpy
import folium
from streamlit_folium import folium_static
import os
import glob
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import io
from datetime import datetime

# ==========================================
# ImageSimilarityFinder 클래스
# ==========================================
class ImageSimilarityFinder:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    @st.cache_resource
    def load_model(_self):
        """모델 로드 (캐싱)"""
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.to(_self.device)
        return model, processor
    
    def get_image_embedding(self, image):
        """이미지의 임베딩 벡터 생성"""
        if self.model is None or self.processor is None:
            self.model, self.processor = self.load_model()
        
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        else:
            image = image.convert('RGB')
            
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            embedding = self.model.get_image_features(**inputs)
        
        return embedding.cpu().numpy()

# ==========================================
# 세션 스테이트 초기화
# ==========================================
if 'saved_photos' not in st.session_state:
    st.session_state.saved_photos = []
if 'saved_count' not in st.session_state:
    st.session_state.saved_count = 0
if 'image_finder' not in st.session_state:
    st.session_state.image_finder = ImageSimilarityFinder()


# ==========================================
# GPX지도 설정
# ==========================================

def load_marathon_course(tournament_name):
    """
    대회 이름에 따라 GPX 파일 로드
    """
    gpx_files = {
        "JTBC 마라톤": "../data/2025_JTBC.gpx",
        "춘천 마라톤": "../data/chuncheon_marathon.gpx",
    }
    
    if tournament_name in gpx_files:
        try:
            with open(gpx_files[tournament_name], 'r') as f:
                gpx = gpxpy.parse(f)
            
            coordinates = []
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        coordinates.append([point.latitude, point.longitude])
            
            return coordinates
        except FileNotFoundError:
            return None
    return None

def create_course_map(coordinates, photo_locations=None):
    """
    코스 지도 + 사진 위치 표시
    """
    if not coordinates:
        return None
    
    # 중심점 계산
    center_lat = sum([c[0] for c in coordinates]) / len(coordinates)
    center_lon = sum([c[1] for c in coordinates]) / len(coordinates)
    
    # 지도 생성
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='CartoDB positron'
    )
    
    # 코스 라인
    folium.PolyLine(
        coordinates,
        color='#FF4444',
        weight=5,
        opacity=0.8,
        popup='마라톤 코스'
    ).add_to(m)
    
    # 출발/도착 마커
    folium.Marker(
        coordinates[0],
        popup='🏁 출발',
        icon=folium.Icon(color='green', icon='play')
    ).add_to(m)
    
    folium.Marker(
        coordinates[-1],
        popup='🎯 도착',
        icon=folium.Icon(color='red', icon='stop')
    ).add_to(m)

     # 7단계: 10km마다 거리 마커 추가
    total_points = len(coordinates)
    for km in [10, 20, 21.0975, 30, 40]:
        # 포인트 인덱스 계산 (비율로)
        idx = int((km / 42.195) * total_points)
        if idx < total_points:
            folium.CircleMarker(
                location=coordinates[idx],
                radius=8,
                popup=f'{km}km 지점',
                color='blue',
                fill=True,
                fillColor='lightblue',
                fillOpacity=0.7
            ).add_to(m)
    
    # 사진 위치 표시
    if photo_locations:
        for photo in photo_locations:
            folium.Marker(
                [photo['lat'], photo['lon']],
                popup=folium.Popup(
                    f"""
                    <div style='width: 200px;'>
                        <img src='{photo['thumbnail']}' style='width: 100%;'><br>
                        <b>{photo['name']}</b><br>
                        <small>{photo['distance']:.1f}km 지점</small>
                    </div>
                    """,
                    max_width=220
                ),
                icon=folium.Icon(color='orange', icon='camera')
            ).add_to(m)
    
    return m

# ==========================================
# 페이지 설정
# ==========================================
# 모드 선택

mode = st.radio(
    "모드 선택",
    ["📸 작가 모드", "🔍 이용자 모드"],
    label_visibility="collapsed"
)

# ==========================================
# 작가 모드
# ==========================================
if mode == "📸 작가 모드":
    st.markdown("### 📸 사진 업로드 및 AI 분류")
    st.info("💡 여러 장의 사진을 한 번에 업로드하고 위치를 입력하세요. AI가 자동으로 임베딩을 생성합니다.")
    
    # 파일 업로드
    uploaded_files = st.file_uploader(
        "사진을 선택하세요 (여러 장 가능)",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        key="photographer_upload"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)}장의 사진이 업로드되었습니다!")
        
        # 업로드된 사진 표시
        st.markdown("### 📷 업로드된 사진")
        
        # 사진을 4개씩 나눠서 표시
        cols = st.columns(4)
        photo_data = []
        
        for idx, uploaded_file in enumerate(uploaded_files):
            col = cols[idx % 4]
            
            with col:
                # 이미지 표시
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
                
                # 위치 입력
                location = st.text_input(
                    "위치",
                    placeholder="예: 서울역",
                    key=f"location_{idx}"
                )
                
                photo_data.append({
                    'image': image,
                    'name': uploaded_file.name,
                    'location': location,
                    'uploaded_file': uploaded_file
                })
        
        st.markdown("---")
        
        # 저장 버튼
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("💾 DB에 저장하기", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 각 이미지의 임베딩 생성
                for idx, photo in enumerate(photo_data):
                    status_text.text(f"🤖 AI 처리 중... ({idx + 1}/{len(photo_data)})")
                    
                    try:
                        # 임베딩 생성
                        embedding = st.session_state.image_finder.get_image_embedding(photo['image'])
                        photo['embedding'] = embedding
                        photo['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # 이미지를 바이트로 변환하여 저장
                        img_byte_arr = io.BytesIO()
                        photo['image'].save(img_byte_arr, format='PNG')
                        photo['image_bytes'] = img_byte_arr.getvalue()
                        
                    except Exception as e:
                        st.error(f"❌ {photo['name']} 처리 중 오류: {str(e)}")
                        continue
                    
                    progress_bar.progress((idx + 1) / len(photo_data))
                
                # 데이터 저장
                st.session_state.saved_photos.extend(photo_data)
                st.session_state.saved_count += len(photo_data)
                
                status_text.empty()
                progress_bar.empty()
                
                # 성공 메시지
                st.success(f"✅ {len(photo_data)}장의 사진이 저장되었습니다!")
                st.balloons()
                
                # 페이지 새로고침
                st.rerun()

else:
    st.set_page_config(
        page_title="마라톤 사진 검색",
        page_icon="🏃‍♂️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # ==========================================
    # 간결한 CSS 스타일
    # ==========================================
    st.markdown("""
    <style>
        /* 전체 배경 */
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%);
        }
        
        /* 드롭다운 스타일 */
        .stSelectbox {
            font-size: 18px;
        }
        
        /* 버튼 스타일 */
        .stButton>button {
            background: linear-gradient(90deg, #4a90e2 0%, #50e3c2 100%);
            color: white;
            font-size: 18px;
            font-weight: bold;
            padding: 15px 30px;
            border-radius: 12px;
            border: none;
            width: 100%;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(74, 144, 226, 0.4);
        }
        
        /* 업로드 영역 */
        .stFileUploader {
            border: 2px dashed #4a90e2;
            border-radius: 12px;
            padding: 30px;
            background: white;
        }
        
        /* 카드 스타일 */
        .info-card {
            background: white;
            padding: 20px;
            border-radius: 12px;
            border-left: 4px solid #4a90e2;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        /* 사진 카드 */
        .photo-card {
            background: white;
            padding: 15px;
            border-radius: 10px;
            border: 2px solid #e0e7ff;
            text-align: center;
            transition: all 0.3s;
            cursor: pointer;
        }
        
        .photo-card:hover {
            transform: scale(1.05);
            border-color: #4a90e2;
            box-shadow: 0 4px 12px rgba(74, 144, 226, 0.3);
        }
        
        /* 제목 */
        h1 {
            color: #2c3e50;
            text-align: center;
            font-size: 48px;
            margin-bottom: 30px;
        }
        
        h2 {
            color: #34495e;
            font-size: 28px;
        }
        
        h3 {
            color: #4a90e2;
            font-size: 22px;
        }
    </style>
    """, unsafe_allow_html=True)

    # ==========================================
    # 세션 스테이트 초기화
    # ==========================================
    if 'selected_tournament' not in st.session_state:
        st.session_state.selected_tournament = None

    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None

    if 'show_results' not in st.session_state:
        st.session_state.show_results = False

    # ==========================================
    # 대회 데이터
    # ==========================================
    tournaments = {
        "JTBC 마라톤": {
            "date": "2025년 11월 2일",
            "distance": "42.195km",
            "participants": "30,000명",
            "course": "잠실종합운동장 → 광화문 → 남산 → 한강 → 잠실",
            "icon": "🏃‍♂️",
            "color": "#FF6B6B"
        },
        "춘천 마라톤": {
            "date": "2025년 10월 26일",
            "distance": "42.195km",
            "participants": "15,000명",
            "course": "의암호 → 소양강 → 춘천시가지 → 의암호",
            "icon": "🏔️",
            "color": "#4ECDC4"
        }
        # },
        # "제주 국제 마라톤": {
        #     "date": "2024년 11월 5일",
        #     "distance": "42.195km",
        #     "participants": "12,000명",
        #     "course": "제주시 → 애월 → 한림 → 제주시",
        #     "icon": "🌊",
        #     "color": "#45B7D1"
        # },
        # "부산 국제 마라톤": {
        #     "date": "2024년 4월 14일",
        #     "distance": "42.195km",
        #     "participants": "25,000명",
        #     "course": "광안리 → 해운대 → 마린시티 → 광안리",
        #     "icon": "🌉",
        #     "color": "#FFA07A"
        # }
    }

    # ==========================================
    # 페이지 1: 대회 선택 및 사진 업로드
    # ==========================================
    if not st.session_state.show_results:
        
        # 타이틀
        st.title("🏃 High 러너스 🏃")
        st.caption("AI가 마라톤 코스에서 당신의 사진을 찾아드립니다")
        st.markdown("---")
        
        # 중앙 정렬 레이아웃
        col1, col2, col3 = st.columns([1, 80, 1])
        
        with col2:
            # 1단계: 대회 선택
            st.markdown("### 1️⃣ 대회 선택")
            selected = st.selectbox(
                "참가한 마라톤 대회를 선택하세요",
                options=["대회를 선택해주세요"] + list(tournaments.keys()),
                key="tournament_selectbox"
            )
            
            # 대회가 선택되면 세션에 저장
            if selected != "대회를 선택해주세요":
                st.session_state.selected_tournament = selected
                
                # # 선택된 대회 정보 표시
                # info = tournaments[selected]
                # st.markdown(f"""
                # <div class="info-card">
                #     <h3>{info['icon']} {selected}</h3>
                #     <p style='margin: 5px 0; color: #666;'>
                #         📅 <b>일시:</b> {info['date']}<br>
                #         📏 <b>거리:</b> {info['distance']}<br>
                #         👥 <b>참가자:</b> {info['participants']}<br>
                #         📍 <b>코스:</b> {info['course']}
                #     </p>
                # </div>
                # """, unsafe_allow_html=True)
                
                # st.markdown("---")
                
                # 2단계: 사진 업로드
                st.markdown("### 2️⃣ 사진 업로드")
                uploaded_file = st.file_uploader(
                    "Drag and drop file here",
                    type=['png', 'jpg', 'jpeg'],
                    key="photo_uploader",
                    help="마라톤 사진을 업로드하세요 (최대 200MB)"
                )
                
                # 사진이 업로드되면
                if uploaded_file:
                    # 이미지 읽기 및 세션에 저장
                    image = Image.open(uploaded_file)
                    st.session_state.uploaded_image = image
                    
                    # # 미리보기 표시
                    # st.success(f"✅ {uploaded_file.name} 업로드 완료!")
                    # st.image(image, caption="업로드된 사진", use_container_width=True)
                    
                    # st.markdown("---")
                    
                    # 검색 버튼
                    if st.button("🔍 코스 및 추천 사진 보기", type="primary"):
                        st.session_state.show_results = True
                        st.rerun()
            
            else:
                st.info("👆 위에서 대회를 먼저 선택해주세요")

# ==========================================
# 페이지 2: 코스 지도 + 유사 사진
# ==========================================
    else:
        # 선택된 대회 정보 가져오기
        tournament_name = st.session_state.selected_tournament
        tournament_info = tournaments[tournament_name]
        
        # 상단 헤더
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; background: white; border-radius: 12px; margin-bottom: 30px;'>
            <h1 style='margin: 0; font-size: 36px;'>{tournament_info['icon']} {tournament_name}</h1>
            
        </div>
        """, unsafe_allow_html=True)
        
        # 좌우 분할 (6:4 비율)
        left_col, right_col = st.columns([6, 4])
        
        # ==========================================
        # 왼쪽: 코스 지도 영역
        # ==========================================
        with left_col:
            st.markdown("### 🗺️ 마라톤 코스")
            
            # 대회 정보 카드
            st.markdown(f"""
            <div class="info-card">
                <p style='margin: 0; line-height: 1.8;'>
                    📅 <b>일시:</b> {tournament_info['date']}<br>
                    📏 <b>거리:</b> {tournament_info['distance']}<br>
                    📍 <b>코스:</b> {tournament_info['course']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # 코스 지도 영역 (플레이스홀더)
            coordinates = load_marathon_course(tournament_name)
        
            if coordinates:
                st.success(f"✅ {tournament_name} 코스를 불러왔습니다!")
                
                # 지도 생성 및 표시
                m = create_course_map(coordinates)
                
                if m:
                    folium_static(m, width=1300, height=600)
            else:
                st.error("❌ 코스 데이터를 찾을 수 없습니다.")


            # st.markdown("""
            # <div style='background: white; border-radius: 12px; padding: 40px; text-align: center; min-height: 500px; border: 2px solid #e0e7ff;'>
            #     <div style='padding-top: 100px;'>
            #         <h2 style='color: #4a90e2; font-size: 64px; margin-bottom: 20px;'>🗺️</h2>
            #         <h3 style='color: #666;'>마라톤 코스 지도</h3>
            #         <p style='color: #999; margin-top: 20px;'>실제 구현시 Google Maps API 또는 Folium 사용</p>
            #         <br><br>
            #         <div style='display: flex; justify-content: space-around; margin-top: 60px;'>
            #             <div>
            #                 <div style='width: 80px; height: 80px; background: #e8f5e9; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 36px;'>🏁</div>
            #                 <p style='margin-top: 15px; color: #666; font-weight: bold;'>출발</p>
            #             </div>
            #             <div>
            #                 <div style='width: 80px; height: 80px; background: #fff3e0; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 36px;'>📸</div>
            #                 <p style='margin-top: 15px; color: #666; font-weight: bold;'>중간</p>
            #             </div>
            #             <div>
            #                 <div style='width: 80px; height: 80px; background: #fce4ec; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 36px;'>🎯</div>
            #                 <p style='margin-top: 15px; color: #666; font-weight: bold;'>도착</p>
            #             </div>
            #         </div>
            #     </div>
            # </div>
            # """, unsafe_allow_html=True)
        
        # ==========================================
        # 오른쪽: 유사한 사진들
        # ==========================================
        with right_col:
            # st.markdown("### 📍 코스 상 유사한 사진들")
            
            # 업로드한 사진 미리보기
            if st.session_state.uploaded_image:
                st.markdown("#### 🖼️ 검색한 사진")
                image = st.session_state.uploaded_image
                st.image(image, width=400)
                st.markdown("---")

                # 검색 옵션
                st.markdown("#### ⚙️ 검색 옵션")
                
                # 코스 구간 선택
                course_section = st.selectbox(
                    "📍 코스 구간 (선택사항)",
                    ["전체 코스", "0-10km", "10-20km", "20-30km", "30-42km"]
                )
                
                # 결과 개수
                top_k = st.slider(
                    "📊 표시할 결과 개수",
                    min_value=1,
                    max_value=20,
                    value=5
                )

                # 유사도 임계값
                similarity_threshold = st.slider(
                    "🎯 최소 유사도 (%)",
                    min_value=0,
                    max_value=100,
                    value=70,
                    help="높을수록 더 비슷한 사진만 표시됩니다"
                )
                
                st.markdown("---")
                
                # 검색 버튼
                search_button = st.button("🔍 유사 사진 검색", type="primary", use_container_width=True)
                
                if search_button:
                    if len(st.session_state.saved_photos) == 0:
                        st.warning("⚠️ 저장된 사진이 없습니다. 먼저 작가 모드에서 사진을 업로드해주세요.")
                    else:
                        with st.spinner("🤖 AI가 코스 위에서 유사한 사진을 찾고 있습니다..."):
                            try:
                                # 검색 이미지의 임베딩 생성
                                query_image = st.session_state.uploaded_image
                                query_embedding = st.session_state.image_finder.get_image_embedding(query_image)
                                
                                # 저장된 모든 이미지와 유사도 계산
                                results = []
                                for saved_photo in st.session_state.saved_photos:
                                    if 'embedding' not in saved_photo:
                                        continue
                                    
                                    # 유사도 계산
                                    similarity = cosine_similarity(query_embedding, saved_photo['embedding'])[0][0]
                                    similarity_percent = float(similarity * 100)
                                    
                                    # 임계값 필터
                                    if similarity_percent >= similarity_threshold:
                                        results.append({
                                            'photo': saved_photo,
                                            'similarity': similarity_percent
                                        })
                                
                                # 유사도 순으로 정렬
                                results.sort(key=lambda x: x['similarity'], reverse=True)
                                results = results[:top_k]
                                
                                # 결과 표시
                                if len(results) == 0:
                                    st.warning("😔 조건에 맞는 사진을 찾지 못했습니다.")
                                else:
                                    st.success(f"✅ **{len(results)}장**의 유사한 사진을 찾았습니다!")
                                    st.markdown("---")
                                    
                                    for idx, result in enumerate(results):
                                        col1, col2 = st.columns([1, 2])
                                        
                                        with col1:
                                            # 저장된 이미지 표시
                                            result_image = Image.open(io.BytesIO(result['photo']['image_bytes']))
                                            st.image(result_image, use_container_width=True)
                                        
                                        with col2:
                                            st.markdown(f"**#{idx + 1}**")
                                            st.markdown(f"**📍 {result['photo'].get('location', '위치 미상')}**")
                                            st.markdown(f"**📁 {result['photo']['name']}**")
                                            
                                            # 유사도 표시
                                            similarity_val = float(result['similarity'] / 100)
                                            st.progress(similarity_val)
                                            st.caption(f"유사도: {result['similarity']:.2f}%")
                                            
                                            # 타임스탬프
                                            if 'timestamp' in result['photo']:
                                                st.caption(f"업로드: {result['photo']['timestamp']}")
                                        
                                        st.markdown("---")
                                    
                                    # 세션 스테이트 업데이트
                                    st.session_state.uploaded_photo = image
                                    st.session_state.show_recommendations = True
                            
                            except Exception as e:
                                st.error(f"❌ 검색 중 오류가 발생했습니다: {str(e)}")
            
            else:
                # 업로드 전 안내
                st.markdown("""
                <div class="upload-area">
                    <div style='padding: 50px 20px;'>
                        <div style='font-size: 64px; margin-bottom: 20px;'>📤</div>
                        <h3 style='color: #666; margin-bottom: 10px;'>사진을 업로드하세요</h3>
                        <p style='color: #999;'>JPG, PNG 형식 지원</p>
                        <br>
                        <small style='color: #bbb;'>위 버튼을 클릭하여 파일을 선택하세요</small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
    # ==========================================
    # 푸터
    # ==========================================
    st.markdown("---")
    st.caption("🤖 Powered by OpenAI CLIP Model | 이미지 임베딩 기반 유사도 검색")            
        #     # AI 추천 사진
        #     st.success("✨ AI가 찾은 유사한 사진 5장")
            
        #     # # 5장의 추천 사진 (2열로 배치)
        #     # for i in range(5):
        #     #     st.markdown(f"""
        #     #     <div class="photo-card">
        #     #         <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        #     #                     height: 200px; 
        #     #                     border-radius: 8px; 
        #     #                     display: flex; 
        #     #                     align-items: center; 
        #     #                     justify-content: center; 
        #     #                     margin-bottom: 12px;'>
        #     #             <span style='font-size: 64px;'>🖼️</span>
        #     #         </div>
        #     #         <div style='text-align: left; padding: 5px;'>
        #     #             <p style='margin: 0; font-size: 16px; font-weight: bold; color: #2c3e50;'>
        #     #                 📍 {i*8 + 5}km 지점
        #     #             </p>
        #     #             <p style='margin: 5px 0 0 0; font-size: 14px; color: #4a90e2;'>
        #     #                 유사도: {95 - i*2}%
        #     #             </p>
        #     #         </div>
        #     #     </div>
        #     #     """, unsafe_allow_html=True)
                
        #     #     st.markdown("<br>", unsafe_allow_html=True)
        
        # # ==========================================
        # # 하단: 뒤로 가기 버튼
        # # ==========================================
        # st.markdown("---")
        
    col1, col2, col3 = st.columns([5, 10, 5])
    with col2:
        if st.button("◀️ 처음으로 돌아가기", use_container_width=True):
            # 세션 초기화
            st.session_state.show_results = False
            st.session_state.selected_tournament = None
            st.session_state.uploaded_image = None
            st.rerun()

# ==========================================
# 하단 푸터
# ==========================================
st.markdown("---")
st.caption("💡 Tip: 정확한 검색을 위해 선명한 사진을 업로드해주세요")