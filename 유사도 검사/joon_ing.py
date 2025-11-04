import streamlit as st
from PIL import Image
import torch
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import io
import base64
from datetime import datetime

# 페이지 설정
st.set_page_config(
    page_title="AI 이미지 유사도 검색",
    page_icon="🔍",
    layout="wide"
)

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

# 헤더
st.title("🔍 AI 이미지 유사도 검색 시스템")
st.markdown("**CLIP 모델 기반 지능형 이미지 매칭**")

# 사이드바
with st.sidebar:
    st.markdown("### 📊 시스템 정보")
    st.metric("저장된 이미지", f"{len(st.session_state.saved_photos)}장")
    
    device_info = "🟢 GPU" if torch.cuda.is_available() else "🔵 CPU"
    st.info(f"연산 장치: {device_info}")
    
    st.markdown("---")
    
    # 모드 선택
    mode = st.radio(
        "모드 선택",
        ["📸 작가 모드", "🔍 이용자 모드"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.caption("💡 작가 모드: 사진 업로드 및 저장")
    st.caption("💡 이용자 모드: 유사 이미지 검색")

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
# 이용자 모드
# ==========================================
else:
    st.markdown("### 🔍 유사 사진 검색")
    st.info("💡 찾고 싶은 사진을 업로드하면 AI가 CLIP 모델로 비슷한 사진을 찾아드립니다")
    
    # 2개 열로 나누기
    left_col, right_col = st.columns([1, 1])
    
    # 왼쪽: 검색 설정
    with left_col:
        st.markdown("#### 🖼️ 검색할 이미지")
        search_image = st.file_uploader(
            "이미지를 선택하세요",
            type=['png', 'jpg', 'jpeg'],
            key="user_upload"
        )
        
        if search_image:
            image = Image.open(search_image)
            st.image(image, caption="검색할 이미지", use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("#### ⚙️ 검색 옵션")
        
        # 위치 필터
        location_filter = st.text_input(
            "📍 위치 필터",
            placeholder="예: 서울역 (비워두면 전체 검색)"
        )
        
        # 결과 개수
        top_k = st.slider(
            "📊 표시할 결과 개수",
            min_value=1,
            max_value=20,
            value=5
        )
        
        # 유사도 슬라이더
        similarity_threshold = st.slider(
            "🎯 최소 유사도 (%)",
            min_value=0,
            max_value=100,
            value=70,
            help="높을수록 비슷한 이미지만 표시됩니다"
        )
        
        st.markdown("---")
        
        # 검색 버튼
        search_button = st.button("🔍 검색 시작", type="primary", disabled=not search_image)
    
    # 오른쪽: 검색 결과
    with right_col:
        st.markdown("#### 📊 검색 결과")
        
        # 검색 전 안내
        if not search_image:
            st.info("👈 왼쪽에서 이미지를 업로드하고 검색해보세요")
        
        elif search_button:
            if len(st.session_state.saved_photos) == 0:
                st.warning("⚠️ 저장된 사진이 없습니다. 먼저 작가 모드에서 사진을 업로드해주세요.")
            
            else:
                with st.spinner("🤖 AI가 유사 이미지를 검색 중입니다..."):
                    try:
                        # 검색 이미지의 임베딩 생성
                        query_image = Image.open(search_image)
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
                            similarity_percent = similarity * 100
                            
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
                            
                            for idx, result in enumerate(results):
                                with st.container():
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
                                        similarity_val = result['similarity'] / 100
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