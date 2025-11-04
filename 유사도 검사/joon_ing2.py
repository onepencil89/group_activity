"""
마라톤 사진 검색 플랫폼 - UI/UX 프로토타입
이용자가 대회를 선택하고 사진을 업로드하면 코스 위에 유사한 사진을 추천
"""

import streamlit as st
from PIL import Image
import os
import glob
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# ==========================================
# ImageSimilarityFinder 클래스
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

# 세션 스테이트 초기화
if 'saved_photos' not in st.session_state:
    st.session_state.saved_photos = []
if 'saved_count' not in st.session_state:
    st.session_state.saved_count = 0
if 'image_finder' not in st.session_state:
    st.session_state.image_finder = ImageSimilarityFinder()

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
# ==========================================
# 페이지 설정
# ==========================================
else:

    st.set_page_config(
        page_title="마라톤 사진 검색",
        page_icon="🏃‍♂️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ==========================================
    # CSS 스타일
    # ==========================================
    st.markdown("""
    <style>
        /* 전체 배경 */
        .main {
            background-color: #f8f9fa;
        }
        
        /* 사이드바 스타일 */
        [data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 2px solid #e9ecef;
        }
        
        /* 대회 선택 버튼 스타일 */
        .tournament-card {
            background: white;
            padding: 20px;
            border-radius: 12px;
            border: 2px solid #e9ecef;
            margin-bottom: 15px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .tournament-card:hover {
            border-color: #4CAF50;
            box-shadow: 0 4px 12px rgba(76, 175, 80, 0.2);
            transform: translateY(-2px);
        }
        
        .tournament-card.active {
            border-color: #4CAF50;
            background: #f1f8f4;
        }
        
        /* 코스 지도 영역 */
        .course-map {
            background: white;
            border-radius: 12px;
            padding: 20px;
            min-height: 600px;
            border: 2px solid #e9ecef;
        }
        
        /* 업로드 영역 */
        .upload-area {
            background: white;
            border-radius: 12px;
            padding: 30px;
            border: 3px dashed #dee2e6;
            text-align: center;
            min-height: 300px;
            transition: all 0.3s;
        }
        
        .upload-area:hover {
            border-color: #4CAF50;
            background: #f8fff9;
        }
        
        /* 사진 핀 스타일 */
        .photo-pin {
            background: white;
            border: 3px solid #4CAF50;
            border-radius: 12px;
            padding: 10px;
            margin: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .photo-pin:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 16px rgba(0,0,0,0.2);
        }
        
        /* 헤더 */
        h1 {
            color: #2c3e50;
            font-weight: 700;
        }
        
        h2, h3 {
            color: #34495e;
        }
        
        /* 버튼 */
        .stButton>button {
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(76, 175, 80, 0.4);
        }
    </style>
    """, unsafe_allow_html=True)

    # ==========================================
    # 세션 스테이트 초기화
    # ==========================================
    if 'selected_tournament' not in st.session_state:
        st.session_state.selected_tournament = None

    if 'uploaded_photo' not in st.session_state:
        st.session_state.uploaded_photo = None

    if 'show_recommendations' not in st.session_state:
        st.session_state.show_recommendations = False

    # ==========================================
    # 대회 데이터 (예시)
    # ==========================================
    tournaments = {
        "서울 국제 마라톤": {
            "date": "2024년 3월 17일",
            "distance": "42.195km",
            "participants": "30,000명",
            "course": "잠실종합운동장 → 광화문 → 남산 → 한강 → 잠실",
            "icon": "🏃‍♂️"
        },
        "춘천 마라톤": {
            "date": "2024년 10월 20일",
            "distance": "42.195km",
            "participants": "15,000명",
            "course": "의암호 → 소양강 → 춘천시가지 → 의암호",
            "icon": "🏔️"
        },
        "제주 국제 마라톤": {
            "date": "2024년 11월 5일",
            "distance": "42.195km",
            "participants": "12,000명",
            "course": "제주시 → 애월 → 한림 → 제주시",
            "icon": "🌊"
        },
        "부산 국제 마라톤": {
            "date": "2024년 4월 14일",
            "distance": "42.195km",
            "participants": "25,000명",
            "course": "광안리 → 해운대 → 마린시티 → 광안리",
            "icon": "🌉"
        }
    }

    # ==========================================
    # 사이드바: 대회 선택
    # ==========================================
    with st.sidebar:
        st.title("🏃‍♂️ 대회 선택")
        st.markdown("참가한 마라톤 대회를 선택하세요")
        st.markdown("---")
        
        for tournament_name, info in tournaments.items():
            # 대회 카드 생성
            is_selected = st.session_state.selected_tournament == tournament_name
            
            if st.button(
                f"{info['icon']} {tournament_name}",
                key=tournament_name,
                use_container_width=True,
                type="primary" if is_selected else "secondary"
            ):
                st.session_state.selected_tournament = tournament_name
                st.session_state.show_recommendations = False
                st.rerun()
            
            if is_selected:
                st.markdown(f"""
                <div style='background: #f1f8f4; padding: 10px; border-radius: 8px; margin-bottom: 15px;'>
                    <small>
                    📅 <b>일시:</b> {info['date']}<br>
                    📏 <b>거리:</b> {info['distance']}<br>
                    👥 <b>참가자:</b> {info['participants']}
                    </small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.caption("💡 대회를 선택하면 코스 지도가 표시됩니다")

    # ==========================================
    # 메인 화면: 좌우 분할
    # ==========================================

    # 헤더
    st.title("🏃‍♂️ 마라톤 사진 검색 플랫폼")
    st.caption("AI가 당신의 마라톤 사진을 코스 위에서 찾아드립니다")
    st.markdown("---")

    # 좌우 분할 (6:4 비율)
    left_col, right_col = st.columns([6, 4])

    # ==========================================
    # 왼쪽: 코스 지도 + 추천 사진
    # ==========================================
    with left_col:
        st.markdown("### 🗺️ 마라톤 코스")
        
        if st.session_state.selected_tournament:
            selected_info = tournaments[st.session_state.selected_tournament]
            
            # 대회 정보 헤더
            st.info(f"""
            **{selected_info['icon']} {st.session_state.selected_tournament}**  
            📍 코스: {selected_info['course']}
            """)
            
            # 코스 지도 영역 (실제로는 지도 API 사용)
            st.markdown("""
            <div class="course-map">
                <div style='text-align: center; padding: 50px 0;'>
                    <h2 style='color: #95a5a6; margin-bottom: 20px;'>🗺️</h2>
                    <h3 style='color: #95a5a6;'>코스 지도 영역</h3>
                    <p style='color: #bdc3c7;'>(실제 구현시 Google Maps API 또는 Folium 사용)</p>
                    <br><br>
                    <div style='display: flex; justify-content: space-around; margin-top: 40px;'>
                        <div style='text-align: center;'>
                            <div style='width: 60px; height: 60px; background: #e8f5e9; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 24px;'>
                                🏁
                            </div>
                            <p style='margin-top: 10px; color: #666;'>출발점</p>
                        </div>
                        <div style='text-align: center;'>
                            <div style='width: 60px; height: 60px; background: #fff3e0; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 24px;'>
                                📸
                            </div>
                            <p style='margin-top: 10px; color: #666;'>중간 지점</p>
                        </div>
                        <div style='text-align: center;'>
                            <div style='width: 60px; height: 60px; background: #fce4ec; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 24px;'>
                                🎯
                            </div>
                            <p style='margin-top: 10px; color: #666;'>도착점</p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # 추천 사진이 있을 때
            if st.session_state.show_recommendations:
                st.markdown("---")
                st.markdown("#### 📍 코스 상 유사한 사진들")
                st.success("✨ AI가 찾은 유사한 사진 5장")
                
                # 추천 사진 표시 (3개씩)
                rec_cols = st.columns(3)
                
                for i in range(5):
                    col = rec_cols[i % 3]
                    with col:
                        st.markdown(f"""
                        <div class="photo-pin">
                            <div style='background: #f0f0f0; height: 150px; border-radius: 8px; display: flex; align-items: center; justify-content: center; margin-bottom: 10px;'>
                                <span style='font-size: 48px;'>🖼️</span>
                            </div>
                            <p style='margin: 0; font-size: 14px; color: #666;'>
                                <b>📍 {i*8 + 5}km 지점</b><br>
                                유사도: {95 - i*3}%
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        else:
            # 대회 미선택 시
            st.info("👈 왼쪽 사이드바에서 대회를 선택하세요")
            st.markdown("""
            <div style='text-align: center; padding: 100px 50px; color: #95a5a6;'>
                <h1 style='font-size: 80px; margin-bottom: 20px;'>🏃‍♂️</h1>
                <h2>마라톤 대회를 선택해주세요</h2>
                <p>대회를 선택하면 코스 지도가 표시됩니다</p>
            </div>
            """, unsafe_allow_html=True)

    # ==========================================
    # 오른쪽: 사진 업로드
    # ==========================================
    with right_col:
        st.markdown("### 📤 내 사진 업로드")
        
        if st.session_state.selected_tournament:
            st.info("📸 마라톤 사진을 업로드하면 AI가 비슷한 사진을 찾아드립니다")
            
            # 파일 업로드
            uploaded_file = st.file_uploader(
                "사진을 선택하세요",
                type=['png', 'jpg', 'jpeg'],
                key="user_photo_upload",
                label_visibility="collapsed"
            )
            
            if uploaded_file:
                # 업로드된 사진 미리보기
                st.markdown("#### 🖼️ 업로드한 사진")
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True, caption=uploaded_file.name)
                
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
                similarity = st.slider(
                    "🎯 최소 유사도",
                    min_value=70,
                    max_value=100,
                    value=85,
                    help="높을수록 더 비슷한 사진만 표시됩니다"
                )
                
                st.markdown("---")
                
                # 검색 버튼
                if st.button("🔍 유사 사진 검색", type="primary", use_container_width=True):
                    with st.spinner("🤖 AI가 코스 위에서 유사한 사진을 찾고 있습니다..."):
                        import time
                        time.sleep(2)  # 시뮬레이션
                        st.session_state.uploaded_photo = image
                        st.session_state.show_recommendations = True
                        st.success("✅ 5장의 유사한 사진을 찾았습니다!")
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
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
        
        else:
            # 대회 미선택 시
            st.warning("⚠️ 먼저 대회를 선택해주세요")
            st.markdown("""
            <div style='text-align: center; padding: 50px 20px; color: #95a5a6;'>
                <div style='font-size: 48px; margin-bottom: 20px;'>🏃‍♂️</div>
                <p>대회를 먼저 선택하면<br>사진을 업로드할 수 있습니다</p>
            </div>
            """, unsafe_allow_html=True)

    # ==========================================
    # 하단 안내
    # ==========================================
    st.markdown("---")
    # 검색 전 안내
    if not uploaded_file:
        st.info("👈 왼쪽에서 이미지를 업로드하고 검색해보세요")

    elif search_button:
        if len(st.session_state.saved_photos) == 0:
            st.warning("⚠️ 저장된 사진이 없습니다. 먼저 작가 모드에서 사진을 업로드해주세요.")
        
        else:
            with st.spinner("🤖 AI가 유사 이미지를 검색 중입니다..."):
                try:
                    # 검색 이미지의 임베딩 생성
                    query_image = Image.open(uploaded_file)
                    query_embedding = st.session_state.image_finder.get_image_embedding(query_image)
                    
                    # 저장된 모든 이미지와 유사도 계산
                    results = []
                    for saved_photo in st.session_state.saved_photos:
                        if 'embedding' not in saved_photo:
                            continue
                        
                        # 위치 필터 적용
                        if location_filter and location_filter.strip():
                            if location_filter.lower() not in saved_photo.get('location', '').lower():
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
                
                except Exception as e:
                    st.error(f"❌ 검색 중 오류가 발생했습니다: {str(e)}")

    # 푸터
    st.markdown("---")
    st.caption("🤖 Powered by OpenAI CLIP Model | 이미지 임베딩 기반 유사도 검색")