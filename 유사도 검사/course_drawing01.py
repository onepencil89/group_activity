import streamlit as st
import folium
from streamlit_folium import folium_static
import gpxpy
import io

st.title("🗺️ GPX 파일 업로드 & 지도 표시")

# 파일 업로더
uploaded_file = st.file_uploader(
    "GPX 파일을 업로드하세요",
    type=['gpx']
)

if uploaded_file is not None:
    # GPX 파일 읽기
    gpx = gpxpy.parse(uploaded_file)
    
    # 좌표 추출
    coordinates = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                coordinates.append([point.latitude, point.longitude])
    
    st.success(f"✅ {len(coordinates)}개의 포인트를 읽었습니다!")
    
    # 지도 생성
    center_lat = sum([c[0] for c in coordinates]) / len(coordinates)
    center_lon = sum([c[1] for c in coordinates]) / len(coordinates)
    
    # styles = [
    #     'OpenStreetMap',      # 기본 (무료)
    #     'CartoDB positron',   # 밝은 테마 (추천!)
    #     'CartoDB dark_matter',# 어두운 테마
    #     'Stamen Terrain',     # 지형 강조
    #     'Stamen Watercolor'   # 수채화 느낌
    # ]

    m = folium.Map(
        location=[37.5665, 126.9780],
        zoom_start=13,
        titles = 'CartoDB positron'  # 깔끔한 밝은 테마
    )
    
    # 경로 그리기
    folium.PolyLine(
        locations = coordinates,
        color='red',
        weight=5,
        opacity=0.8
    ).add_to(m)
    
    # 출발/도착 마커
    folium.Marker(
        coordinates[0],
        popup='출발',
        icon=folium.Icon(color='green')
    ).add_to(m)
    
    folium.Marker(
        coordinates[-1],
        popup='도착',
        icon=folium.Icon(color='red')
    ).add_to(m)
    
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


    # 지도 표시
    folium_static(m, width=1300, height=600)
else:
    st.info("👆 GPX 파일을 업로드해주세요")