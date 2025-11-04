"""
AI 사진 검색 앱 - Streamlit 버전
간단하고 직관적인 UI
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

# -----------------------------------
# 유사검색 설정

class ImageSimilarityFinder:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
    def get_image_embedding(self, image_path):
        """이미지의 임베딩 벡터 생성"""
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            embedding = self.model.get_image_features(**inputs)
        
        return embedding.cpu().numpy()
    
    def build_index(self, folder_path):
        """폴더 내 모든 이미지의 임베딩 생성"""
        image_paths = []
        embeddings = []
        
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_paths.extend(glob.glob(f'{folder_path}/**/{ext}', recursive=True))
        
        print(f"총 {len(image_paths)}개 이미지 처리 중...")
        
        for i, path in enumerate(image_paths):
            try:
                embedding = self.get_image_embedding(path)
                embeddings.append(embedding)
                if (i + 1) % 100 == 0:
                    print(f"{i + 1}/{len(image_paths)} 완료")
            except Exception as e:
                print(f"오류 발생 ({path}): {e}")
                continue
        
        # 저장
        data = {
            'paths': image_paths,
            'embeddings': np.vstack(embeddings)
        }
        
        with open('image_index.pkl', 'wb') as f:
            pickle.dump(data, f)
        
        return data
    
    def find_similar(self, query_image_path, top_k=10):
        """유사한 이미지 찾기"""
        # 인덱스 로드
        with open('image_index.pkl', 'rb') as f:
            data = pickle.load(f)
        
        # 쿼리 이미지 임베딩
        query_embedding = self.get_image_embedding(query_image_path)
        
        # 유사도 계산
        similarities = cosine_similarity(query_embedding, data['embeddings'])[0]
        
        # # 상위 k개 인덱스
        # top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'path': data['paths'][idx],
                'similarity': similarities[idx]
            })
        
        return results


# ==========================================
# 페이지 설정
# ==========================================
st.set_page_config(
    page_title="AI 사진 검색",
    page_icon="🖼️",
    layout="wide"
)

# ==========================================
# CSS 스타일 (예쁘게 만들기)
# ==========================================
st.markdown("""
<style>
    /* 전체 배경 */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* 카드 스타일 */
    .card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    /* 헤더 */
    .header {
        background: white;
        padding: 30px;
        border-radius: 20px;
        margin-bottom: 30px;
        text-align: center;
    }
    
    /* 버튼 */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
    }
    
    /* 파일 업로더 */
    .uploadedFile {
        border: 3px dashed #ddd;
        border-radius: 15px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 세션 스테이트 초기화
# ==========================================
if 'saved_photos' not in st.session_state:
    st.session_state.saved_photos = []
if 'saved_count' not in st.session_state:
    st.session_state.saved_count = 0

# ==========================================
# 헤더
# ==========================================
st.markdown("""
<div class="header">
    <h1>🖼️ AI 사진 검색 앱</h1>
    <p>사진작가와 이용자를 위한 간단한 플랫폼</p>
</div>
""", unsafe_allow_html=True)

# DB 개수 표시
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.metric(label="📊 DB에 저장된 사진", value=st.session_state.saved_count)

st.markdown("---")

# ==========================================
# 모드 선택
# ==========================================
mode = st.radio(
    "모드 선택",
    ["📸 작가 모드", "🔍 이용자 모드"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("---")

# ==========================================
# 작가 모드
# ==========================================
if mode == "📸 작가 모드":
    st.markdown("### 📸 사진 업로드 및 AI 분류")
    st.info("💡 여러 장의 사진을 한 번에 업로드하고 위치를 입력하세요")
    
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
                    'location': location
                })
        
        st.markdown("---")
        
        # 저장 버튼
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("💾 DB에 저장하기", type="primary"):
                # 데이터 저장
                st.session_state.saved_photos.extend(photo_data)
                st.session_state.saved_count += len(photo_data)
                
                # 성공 메시지
                st.success(f"✅ {len(photo_data)}장의 사진이 저장되었습니다!")
                st.balloons()
                
                # 페이지 새로고침을 위한 rerun
                st.rerun()

# ==========================================
# 이용자 모드
# ==========================================
else:
    st.markdown("### 🔍 유사 사진 검색")
    st.info("💡 찾고 싶은 사진을 업로드하면 AI가 비슷한 사진을 찾아드립니다")
    
    # 2개 열로 나누기
    left_col, right_col = st.columns(2)
    
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
        
        # 유사도 슬라이더
        similarity_threshold = st.slider(
            "🎯 최소 유사도",
            min_value=50,
            max_value=100,
            value=80,
            help="높을수록 비슷한 이미지만 표시됩니다"
        )
        
        st.markdown("---")
        
        # 검색 버튼
        if st.button("🔍 검색 시작", type="primary", disabled=not search_image):
            with st.spinner("🤖 AI가 검색 중입니다..."):
                # 실제 검색 기능은 나중에 구현
                # 지금은 시뮬레이션
                import time
                time.sleep(1.5)
                st.success("✅ 검색 완료!")
    
    # 오른쪽: 검색 결과
    with right_col:
        st.markdown("#### 📊 검색 결과")
        
        # 검색 전 안내
        if not search_image:
            st.info("👈 왼쪽에서 이미지를 업로드하고 검색해보세요")
        else:
            # 결과가 있을 때 (시뮬레이션)
            st.markdown("**찾은 사진: 4장**")
            
            # 예시 결과 (실제로는 DB에서 가져와야 함)
            results = [
                {"location": "서울역", "similarity": 95},
                {"location": "광화문", "similarity": 87},
                {"location": "남산타워", "similarity": 82},
                {"location": "명동", "similarity": 78}
            ]
            
            for idx, result in enumerate(results):
                with st.container():
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        # 실제로는 저장된 이미지를 보여줘야 함
                        if search_image:
                            st.image(image, width=100)
                    
                    with col2:
                        st.markdown(f"**📍 {result['location']}**")
                        st.progress(result['similarity'] / 100)
                        st.caption(f"유사도: {result['similarity']}%")
                    
                    st.markdown("---")

# ==========================================
# 하단 팁
# ==========================================
st.markdown("---")
st.info("💡 **Tip:** 작가 모드로 사진을 먼저 업로드하면 이용자 모드에서 검색이 가능합니다!")

# ==========================================
# 디버그 정보 (사이드바)
# ==========================================
with st.sidebar:
    st.markdown("### 🔧 설정")
    
    if st.checkbox("디버그 모드"):
        st.json({
            "저장된 사진 수": st.session_state.saved_count,
            "메모리 사진 수": len(st.session_state.saved_photos)
        })
    
    st.markdown("---")
    st.markdown("### 📚 사용 가이드")
    st.markdown("""
    **작가 모드:**
    1. 사진 여러 장 업로드
    2. 각 사진에 위치 입력
    3. DB에 저장 버튼 클릭
    
    **이용자 모드:**
    1. 검색할 사진 업로드
    2. 옵션 설정 (선택사항)
    3. 검색 시작 버튼 클릭
    """)