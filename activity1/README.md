# 프로젝트 구조
high-runners/
│
├── function_final.py      ← 메인 앱
├── data/
│   ├── 2025_JTBC.gpx
│   └── chuncheon_marathon.gpx
├── requirements.txt
├── .env                   ← OPENAI_API_KEY
└── README.md              ← 지금 보고 계신 파일

# requirements
```
streamlit
streamlit-folium
folium
torch
transformers
pillow
gpxpy
scikit-learn
python-dotenv
openai
```

# 설치 라이브러리
```
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
```
# 주요 내용

1. 작가 : 대회 선택 -> 사진 위치 지정 -> 사진 선택 -> 저장
2. 이용자 : 대회 선택 -> 사진 업로드 -> 사진 비교 및 검색(유사도 기준) -> 선택 및 저장
3. AI챗봇을 활용한 코칭 기능

# 실행방법(bash)
```
git clone 
cd project1
pip install -r rquirements.txt
streamlit run function_final.py
```