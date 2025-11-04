import streamlit as st
import folium
from streamlit_folium import folium_static
import gpxpy

def load_marathon_course(tournament_name):
    """
    대회 이름에 따라 GPX 파일 로드
    """
    gpx_files = {
        "JTBC 마라톤": "data/2025_JTBC.gpx",
        "춘천 마라톤": "data/chuncheon_marathon.gpx",
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
        zoom_start=13,
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

# 메인 앱
st.title("🏃‍♂️ 마라톤 사진 검색")

# 대회 선택
selected_tournament = st.selectbox(
    "대회 선택",
    ["JTBC 마라톤", "춘천 마라톤"]
)

if selected_tournament:
    # 코스 로드
    with st.spinner("코스를 불러오는 중..."):
        coordinates = load_marathon_course(selected_tournament)
    
    if coordinates:
        st.success(f"✅ {selected_tournament} 코스를 불러왔습니다!")
        
        # 지도 생성 및 표시
        m = create_course_map(coordinates)
        
        if m:
            folium_static(m, width=1000, height=600)
    else:
        st.error("❌ 코스 데이터를 찾을 수 없습니다.")