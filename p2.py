import streamlit as st
from PIL import Image
import google.generativeai as genai
import json
from openai import OpenAI
import re
import time
import os
import plotly.express as px
import pandas as pd
from pytrends.request import TrendReq
from urllib.parse import quote
import requests
from datetime import datetime, timedelta

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="맞춤 패션 추천 시스템",
    page_icon="👕",
    layout="wide"
)

# --- API 키 설정 ---
try:
    kma_api_key = st.secrets["KMA_API_KEY"]
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception as e:
    st.error(f"🚨 API 키를 설정해주세요! .streamlit/secrets.toml 파일을 확인하세요. 오류: {e}")
    st.stop()

# --- Gemini 모델 초기화 ---
llm_model = genai.GenerativeModel('gemini-1.5-flash')


# --- 1. 기능 함수들 ---

# --- 1.1. 날씨 관련 함수 ---

def recommend_clothing(temp):
    """기온에 따라 적절한 옷차림 추천 문구를 반환하는 함수."""
    try:
        temp = float(temp)
    except (ValueError, TypeError):
        return "온도 정보가 없어 추천할 수 없어요."
    if temp >= 28:
        return "민소매, 반팔, 반바지, 원피스 등 매우 가벼운 옷차림을 추천해요. 🥵"
    elif temp >= 23:
        return "반팔, 얇은 셔츠, 반바지, 면바지로 시원하게 입으세요. 😄"
    elif temp >= 17:
        return "얇은 니트, 가디건, 맨투맨, 청바지가 활동하기 좋은 날씨예요. 👍"
    elif temp >= 10:
        return "자켓, 트렌치코트, 니트, 청바지로 멋과 보온을 둘 다 챙기세요.🧥"
    elif temp >= 5:
        return "두꺼운 코트, 가죽 자켓, 플리스, 기모 옷차림이 필요해요. 🥶"
    else:
        return "패딩, 두꺼운 코트, 목도리, 장갑 등 방한용품으로 따뜻하게 입으세요. 🧤"


def get_weather_data(api_key, base_date, base_time, nx, ny):
    """기상청 단기예보 API로부터 날씨 데이터를 요청하는 함수"""
    endpoint = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
    params = {'serviceKey': api_key, 'pageNo': '1', 'numOfRows': '1000', 'dataType': 'JSON', 'base_date': base_date,
              'base_time': base_time, 'nx': nx, 'ny': ny}
    try:
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API 요청 중 오류가 발생했습니다: {e}")
        return None


def process_weather_data(data):
    """API 응답 데이터를 DataFrame으로 변환하고 가공하는 함수"""
    if not data or data['response']['header']['resultCode'] != '00':
        result_msg = data.get('response', {}).get('header', {}).get('resultMsg', '알 수 없는 오류')
        st.error(f"잘못된 응답을 받았습니다: {result_msg}")
        return pd.DataFrame()
    items = data['response']['body']['items']['item']
    df = pd.DataFrame(items)
    df_pivot = df.pivot_table(index=['fcstDate', 'fcstTime'], columns='category', values='fcstValue',
                              aggfunc='first').reset_index()
    sky_codes = {'1': '맑음 ☀️', '3': '구름많음 ☁️', '4': '흐림 🌥️'}
    pty_codes = {'0': '강수 없음', '1': '비 🌧️', '2': '비/눈 🌨️', '3': '눈 ❄️', '4': '소나기 🌦️'}
    if 'SKY' in df_pivot.columns: df_pivot['SKY_STATUS'] = df_pivot['SKY'].map(sky_codes)
    if 'PTY' in df_pivot.columns: df_pivot['PTY_STATUS'] = df_pivot['PTY'].map(pty_codes).fillna('강수 없음')
    return df_pivot


def get_base_datetime():
    """API 요청에 필요한 base_date와 base_time을 계산하는 함수"""
    now = datetime.now()
    if now.hour < 2 or (now.hour == 2 and now.minute <= 10):
        base_dt = now - timedelta(days=1)
        base_hour = 23
    else:
        base_dt = now
        available_times = [2, 5, 8, 11, 14, 17, 20, 23]
        base_hour = max(t for t in available_times if t <= now.hour)
    base_date = base_dt.strftime('%Y%m%d')
    base_time = f"{base_hour:02d}00"
    return base_date, base_time


# --- 1.2. 패션 추천 관련 함수 ---

def recommend_size(height, weight, gender):
    bmi = weight / ((height / 100) ** 2)
    if gender == "남자":
        if height < 170:
            top_size = "M (95)"
        elif 170 <= height < 180:
            top_size = "L (100)" if bmi < 25 else "XL (105)"
        else:
            top_size = "XL (105) 이상"
        if weight < 65:
            bottom_size = "28-30 inch"
        elif 65 <= weight < 75:
            bottom_size = "31-33 inch"
        elif 75 <= weight < 85:
            bottom_size = "34-36 inch"
        else:
            bottom_size = "37 inch 이상"
    else:
        if height < 160:
            top_size = "S (44-55)"
        elif 160 <= height < 168:
            top_size = "M (55-66)" if bmi < 22 else "L (66-77)"
        else:
            top_size = "L (66-77) 이상"
        if weight < 50:
            bottom_size = "25-26 inch (S)"
        elif 50 <= weight < 58:
            bottom_size = "27-28 inch (M)"
        elif 58 <= weight < 68:
            bottom_size = "29-30 inch (L)"
        else:
            bottom_size = "31 inch (XL) 이상"
    return {"상의": top_size, "하의": bottom_size}


@st.cache_data(ttl=3600)
def get_google_trends_data():
    try:
        pytrends = TrendReq(hl='ko-KR', tz=360)
        keywords = ["미니멀리즘 패션", "스트릿 패션", "Y2K 패션", "고프코어"]
        pytrends.build_payload(keywords, cat=0, timeframe='today 12-m', geo='KR', gprop='')
        df = pytrends.interest_over_time()
        return df.drop(columns=['isPartial']) if not df.empty else pd.DataFrame()
    except Exception as e:
        st.warning(f"Google Trends 데이터를 가져오는 데 실패했습니다: {e}")
        return pd.DataFrame()


def save_image(directory, file):
    if not os.path.exists(directory): os.makedirs(directory)
    filename = f"capture_{int(time.time())}.jpg"
    filepath = os.path.join(directory, filename)
    with open(filepath, 'wb') as f: f.write(file.getbuffer())
    return filepath


def save_image_from_url(directory, url):
    if not os.path.exists(directory): os.makedirs(directory)
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        filename = f"saved_{int(time.time())}.png"
        filepath = os.path.join(directory, filename)
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
        return True, filepath
    except Exception as e:
        st.error(f"이미지 저장 중 오류 발생: {e}")
        return False, None


def generate_image_with_dalle(prompt, retries=3, delay=5):
    for attempt in range(retries):
        try:
            response = openai_client.images.generate(model="dall-e-3", prompt=prompt, size="1024x1024",
                                                     quality="standard", n=1)
            return response.data[0].url
        except Exception as e:
            if "Connection error" in str(e) and attempt < retries - 1:
                st.warning(f"DALL-E 연결 오류. {delay}초 후 재시도합니다... ({attempt + 1}/{retries})")
                time.sleep(delay)
            else:
                st.error(f"DALL-E 이미지 생성 중 오류 발생: {e}")
                return None


def analyze_clothing_image(uploaded_image):
    img = Image.open(uploaded_image)
    prompt = """
    당신은 패션 스타일리스트이자 의류 분석 전문가입니다. 이 이미지에 있는 옷을 분석해서 아래 JSON 형식에 맞춰 답변해주세요. 각 항목에 대해 가장 적절한 단 하나의 값만 선택해주세요. **중요: 답변에는 JSON 코드 외에 어떤 설명이나 인사도 포함하지 말고, 오직 JSON 객체만 응답해야 합니다.**
    {"item_type": "상의, 하의, 아우터, 신발, 액세서리 중 하나", "category": "티셔츠, 셔츠, 청바지 등 구체적인 카테고리", "color": "옷의 가장 주된 색상", "pattern": "솔리드(단색), 스트라이프, 체크 등", "style_tags": ["캐주얼", "미니멀", "스트리트", "포멀", "스포티"]}
    """
    try:
        response = llm_model.generate_content([prompt, img])
        match = re.search(r"\{.*\}", response.text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            st.error("AI 응답에서 JSON을 찾을 수 없습니다.");
            st.code(response.text)
            return None
    except Exception as e:
        st.error(f"이미지 분석 중 오류 발생: {e}");
        return None


def get_cody_recommendation_with_image(user_info, clothing_info, situation):
    prompt = f"""
    당신은 친절하고 스타일리시한 AI 패션 어드바이저입니다. 고객 정보, 의류 아이템, 주어진 상황을 바탕으로 최고의 코디를 추천해주세요. **중요: 답변의 가독성을 높이기 위해 다음 규칙을 반드시 지켜주세요.** 1. 각 코디 제안의 제목은 Markdown의 `##`를 사용하여 크고 굵게 표시해주세요. 2. 설명에 어울리는 이모티콘(👕,👖,👟,✨ 등)을 자유롭게 사용해주세요. 3. 의류 아이템, 색상, 스타일 등 중요한 키워드는 `<span style='color: #87CEEB;'>키워드</span>` 와 같이 HTML 태그를 사용해 색상을 입혀 강조해주세요. 4. 추천된 각 아이템 뒤에는 검색 가능한 키워드를 `(검색 키워드: [키워드])` 형식으로 추가해주세요.
    ## 🧑‍💻 고객 정보:
    - 성별: {user_info['성별']}; - 키: {user_info['키']}cm, 몸무게: {user_info['몸무게']}kg; - 피부 톤: {user_info['피부_톤']}; - 선호 스타일: {', '.join(user_info['선호_스타일'])}
    ## 👚 분석된 의류 아이템:
    - 종류: {clothing_info['item_type']}, 카테고리: {clothing_info['category']}, 색상: {clothing_info['color']}, 패턴: {clothing_info['pattern']}
    ## 🏞️ 주어진 상황:
    - {situation}
    ## 요청 사항:
    1. 위 정보를 종합하여, 총 **두 가지 스타일의 완성된 코디**를 추천하고, 각 코디를 추천한 이유를 친절하게 설명해주세요.
    2. 각 코디 설명 후, DALL-E가 이미지를 생성할 수 있도록, **고객의 성별을 반영**하고 **주어진 상황을 반영**하여 해당 코디를 입은 모델의 모습을 상세하고 사실적으로 묘사하는 **영어 프롬프트**를 다음 형식으로 제공해주세요: `IMAGE_PROMPT_1: [첫 번째 코디에 대한 상세한 영어 묘사]`, `IMAGE_PROMPT_2: [두 번째 코디에 대한 상세한 영어 묘사]`
    """
    try:
        response = llm_model.generate_content(prompt)
        recommendation_text = response.text
        image_prompts = re.findall(r"IMAGE_PROMPT_\d:\s*(.*)", recommendation_text)
        search_keywords = re.findall(r"\(검색 키워드: (.*?)\)", recommendation_text)
        display_text = re.sub(r"\(검색 키워드: .*?\)", "", recommendation_text)
        display_text = re.sub(r"IMAGE_PROMPT_\d:\s*.*", "", display_text).strip()
        return display_text, image_prompts, search_keywords
    except Exception as e:
        st.error(f"코디 추천 중 오류 발생: {e}");
        return None, None, None


def analyze_personal_color(face_image):
    img = Image.open(face_image)
    prompt = """
    당신은 전문 퍼스널 컬러 컨설턴트입니다. 이 인물의 얼굴 사진을 보고, 피부의 언더톤, 머리카락과 눈동자 색의 대비 등을 종합적으로 분석하여 가장 가능성이 높은 퍼스널 컬러를 진단해주세요.
    답변은 아래 형식과 같이 **진단 결과**와 **진단 근거**를 명확히 구분하여 작성해주세요. 진단 근거는 2~3가지 핵심적인 이유를 간결한 불릿 포인트로 설명해야 합니다.
    **진단 결과**: [🌱봄 웜톤, 🌊여름 쿨톤, 🍂가을 웜톤, ❄️겨울 쿨톤 중 하나]
    **진단 근거**:
    * 피부 톤: [피부 톤에 대한 구체적인 분석]
    * 헤어/눈동자 컬러: [헤어와 눈동자 컬러에 대한 구체적인 분석]
    * 전체적인 조화: [전체적인 이미지와 색의 조화에 대한 분석]
    """
    try:
        response = llm_model.generate_content([prompt, img])
        return response.text.strip()
    except Exception as e:
        st.error(f"퍼스널 컬러 분석 중 오류 발생: {e}");
        return None


def make_audio(text_to_speak, filename):
    if not os.path.exists("audio"): os.makedirs("audio")
    clean_text = re.sub('<.*?>', '', text_to_speak)
    try:
        response = openai_client.audio.speech.create(model="tts-1", input=clean_text, voice="echo",
                                                     response_format="mp3", speed=1.2)
        filepath = os.path.join("audio", filename)
        response.stream_to_file(filepath)
        return filepath
    except Exception as e:
        st.error(f"음성 생성 중 오류가 발생했습니다: {e}")
        return None


# --- 2. 사이드바 및 페이지 상태 관리 ---
st.sidebar.title("나만의 맞춤 패션 추천")
if st.sidebar.button("🏠 나의 맞춤 패션 추천", use_container_width=True): st.session_state.page = "main"
if st.sidebar.button("🌤️ 오늘의 날씨", use_container_width=True): st.session_state.page = "weather"
if st.sidebar.button("👚 나의 옷장", use_container_width=True): st.session_state.page = "closet"
if st.sidebar.button("🎨 퍼스널 컬러 분석", use_container_width=True): st.session_state.page = "personal_color"
if st.sidebar.button("📊 패션 데이터 분석", use_container_width=True): st.session_state.page = "analytics"
if st.sidebar.button("🔎 옷 입혀보기 AI", use_container_width=True): st.session_state.page = "vton"

# --- 페이지 상태 초기화 ---
if "page" not in st.session_state: st.session_state.page = "main"
if "face_photo_object" not in st.session_state: st.session_state.face_photo_object = None
if "cloth_photo_object" not in st.session_state: st.session_state.cloth_photo_object = None  # 옷 사진 상태
if "user_activity_log" not in st.session_state: st.session_state.user_activity_log = []
if "my_closet" not in st.session_state: st.session_state.my_closet = []
if "saved_images" not in st.session_state: st.session_state.saved_images = []

personal_color_options = ["봄 웜톤", "여름 쿨톤", "가을 웜톤", "겨울 쿨톤"]

# --- 3. 페이지별 UI 구성 ---

# 3.1. 메인 페이지
if st.session_state.page == "main":
    st.title("👕 나의 맞춤 패션 추천")
    tab1, tab_size, tab2, tab3 = st.tabs(["⚙️ 나의 맞춤 정보", "📏 사이즈 추천", "🧠 옷 분석하기", "✨ 코디 추천받기"])
    with tab1:
        st.subheader("성별");
        성별 = st.radio("성별", ["남자", "여자"], horizontal=True, key="gender_radio")
        st.subheader("체형");
        키 = st.number_input("키(cm)", 100, 250, st.session_state.get("키", 170), 1)
        몸무게 = st.number_input("몸무게(kg)", 30, 200, st.session_state.get("몸무게", 62), 1)
        st.subheader("피부")
        default_index = personal_color_options.index(st.session_state.get("피부_톤", personal_color_options[0]))
        피부_톤 = st.radio("피부 톤", personal_color_options, index=default_index, key="skin_tone_radio")
        st.subheader("패션")
        if os.path.exists("img/img.png"): st.image("img/img.png")
        fashion_styles_options = ["캐주얼", "시크", "시티보이", "클래식", "에스닉", "걸리시", "고프코어", "미니멀", "프레피", "리조트", "레트로", "로맨틱",
                                  "스포티", "스트리트", "워크웨어"]
        선호_스타일 = st.multiselect("선호하는 패션 스타일", fashion_styles_options, default=st.session_state.get("선호_스타일", []))
        if st.button("내 정보 저장하기"):
            st.session_state.user_info = {"성별": 성별, "키": 키, "몸무게": 몸무게, "피부_톤": 피부_톤, "선호_스타일": 선호_스타일}
            st.session_state.성별, st.session_state.키, st.session_state.몸무게, st.session_state.피부_톤, st.session_state.선호_스타일 = 성별, 키, 몸무게, 피부_톤, 선호_스타일
            st.success("정보가 저장되었습니다!")
    with tab_size:
        st.subheader("📏 AI 사이즈 추천")
        if 'user_info' in st.session_state:
            gender, height, weight = st.session_state.user_info['성별'], st.session_state.user_info['키'], \
            st.session_state.user_info['몸무게']
            st.write(f"입력된 정보: **성별 {gender}**, **키 {height}cm**, **몸무게 {weight}kg**")
            if st.button("내 사이즈 추천받기", use_container_width=True):
                sizes = recommend_size(height, weight, gender)
                st.success("사이즈 추천 완료!");
                st.markdown(f"### 추천 상의: **{sizes['상의']}**");
                st.markdown(f"### 추천 하의: **{sizes['하의']}**")
                st.info("※ 위 추천은 일반적인 체형 기준이며, 브랜드나 핏에 따라 다를 수 있습니다.")
        else:
            st.warning("먼저 '나의 맞춤 정보' 탭에서 정보를 저장해주세요.")
    with tab2:
        st.subheader("👚 분석할 옷 사진 업로드")
        uploaded_file = st.file_uploader("코디를 추천받고 싶은 옷 사진", type=["jpg", "jpeg", "png"], key="cloth_uploader")

        if uploaded_file:
            st.session_state.cloth_photo_object = uploaded_file

        if st.session_state.cloth_photo_object:
            col1, col2 = st.columns(2)
            with col1:
                st.image(st.session_state.cloth_photo_object, caption="업로드된 옷 이미지", use_container_width=True)
                if st.button("이미지 삭제", use_container_width=True, key="del_cloth_img"):
                    st.session_state.cloth_photo_object = None
                    st.session_state.analysis_result = None
                    st.rerun()

            with col2:
                if st.button("AI로 옷 분석 시작하기", use_container_width=True):
                    if 'user_info' not in st.session_state:
                        st.error("먼저 '나의 맞춤 정보' 탭에서 정보를 저장해주세요!")
                    else:
                        with st.spinner("AI가 이미지를 분석하고 있습니다... 🧠"):
                            analysis_result = analyze_clothing_image(st.session_state.cloth_photo_object)
                            if analysis_result:
                                st.session_state.analysis_result = analysis_result
                                st.session_state.user_activity_log.append(analysis_result)
                                st.rerun()

                if st.session_state.get("analysis_result"):
                    result = st.session_state.analysis_result
                    st.success("분석 완료!");
                    st.subheader("✅ AI 분석 결과")
                    st.write(f"**의류 종류**: {result.get('item_type', 'N/A')}");
                    st.write(f"**카테고리**: {result.get('category', 'N/A')}")
                    st.write(f"**색상**: {result.get('color', 'N/A')}");
                    st.write(f"**패턴**: {result.get('pattern', 'N/A')}")
                    tags = result.get('style_tags', []);
                    st.write(f"**스타일 태그**: {', '.join(tags) if tags else 'N/A'}")
                    if st.button("👚 옷장에 추가하기", use_container_width=True):
                        # 파일 업로더 객체를 직접 사용하면 새로고침 시 사라지므로, session_state에 저장된 객체를 사용
                        closet_item = {"image": st.session_state.cloth_photo_object.getvalue(),
                                       "name": st.session_state.cloth_photo_object.name, "analysis": result}
                        st.session_state.my_closet.append(closet_item);
                        st.success(f"'{st.session_state.cloth_photo_object.name}'을(를) 옷장에 추가했습니다!")
                    st.info("'코디 추천받기' 탭으로 이동하여 추천을 받아보세요!")
    with tab3:
        st.subheader("✨ AI 코디 추천 결과")
        if 'analysis_result' in st.session_state and st.session_state.get(
                'analysis_result') is not None and 'user_info' in st.session_state:
            situation_input = st.text_input("어떤 상황에서 입을 코디를 추천받을까요?", placeholder="예: 주말 오후 카페에서, 도서관에서 공부할 때")
            if st.button("AI 코디 추천 및 이미지 생성", use_container_width=True):
                situation = situation_input if situation_input else "일상적인 상황"
                with st.spinner("AI 스타일리스트가 코디를 만들고 이미지를 생성합니다... ✨"):
                    recommendation_text, image_prompts, search_keywords = get_cody_recommendation_with_image(
                        st.session_state.user_info, st.session_state.analysis_result, situation)
                    if recommendation_text and image_prompts:
                        image_urls = [generate_image_with_dalle(prompt) for prompt in image_prompts]
                        audio_filepath = make_audio(recommendation_text, "output.mp3")
                        st.session_state.recommendation_output = {"text": recommendation_text,
                                                                  "keywords": search_keywords, "image_urls": image_urls,
                                                                  "audio": audio_filepath}
                    else:
                        st.session_state.recommendation_output = None; st.error("코디 추천에 실패했습니다.")

            if st.session_state.get("recommendation_output"):
                output = st.session_state.recommendation_output
                if output.get("audio"): st.audio(output["audio"], autoplay=True)
                st.subheader("AI 스타일리스트의 추천");
                st.markdown(output["text"], unsafe_allow_html=True)
                st.subheader("🛍️ 추천 아이템 쇼핑하기")
                for keyword in set(output["keywords"]):
                    musinsa_url = f"https://www.musinsa.com/search/musinsa/integration?q={quote(keyword)}"
                    st.markdown(f"- [{keyword} 찾아보기]({musinsa_url})")
                if output["image_urls"]:
                    st.subheader("🎨 추천 코디 시각화")
                    cols = st.columns(len(output["image_urls"]))
                    for i, url in enumerate(output["image_urls"]):
                        with cols[i]:
                            if url:
                                st.image(url, caption=f"추천 코디 {i + 1}", use_container_width=True)
                                save_key = f"save_{url}_{i}"
                                if url in st.session_state.saved_images:
                                    st.success("✅ 저장됨")
                                else:
                                    if st.button("💾 이 코디 저장하기", key=save_key, use_container_width=True):
                                        success, filepath = save_image_from_url("saved_outfits", url)
                                        if success:
                                            st.session_state.saved_images.append(url); st.success(f"저장 완료!"); st.rerun()
                                        else:
                                            st.error("저장에 실패했습니다.")
                            else:
                                st.warning(f"추천 코디 {i + 1} 이미지 생성 실패.")
        else:
            st.warning("먼저 '옷 분석하기'를 완료해주세요.")

# 3.2. 오늘의 날씨 페이지
elif st.session_state.page == "weather":
    st.title("🏞️ 기상청 단기예보 조회 서비스")
    st.write(f"오늘 날짜: {datetime.now().strftime('%Y년 %m월 %d일')}")
    locations = {"서울": (60, 127), "부산": (98, 76), "대구": (89, 90), "인천": (55, 124), "광주": (58, 74), "대전": (67, 100),
                 "울산": (102, 84), "세종": (66, 103), "경기": (60, 120), "강원": (73, 134), "충북": (69, 107), "충남": (68, 100),
                 "전북": (63, 89), "전남": (51, 67), "경북": (89, 91), "경남": (91, 77), "제주": (52, 38)}
    selected_location = st.selectbox("조회할 지역을 선택하세요", list(locations.keys()))
    if st.button("날씨 조회하기 🚀", use_container_width=True):
        with st.spinner('날씨 데이터를 가져오는 중입니다...'):
            nx, ny = locations[selected_location]
            base_date, base_time = get_base_datetime()
            weather_json = get_weather_data(kma_api_key, base_date, base_time, nx, ny)
            if weather_json:
                df = process_weather_data(weather_json)
                if not df.empty:
                    st.success(f"**{selected_location}** 지역의 날씨 예보입니다. (데이터 기준: {base_date} {base_time})")
                    latest_data = df.iloc[0]
                    temp = latest_data.get('TMP', 'N/A');
                    sky = latest_data.get('SKY_STATUS', 'N/A');
                    pty = latest_data.get('PTY_STATUS', 'N/A')
                    pop = latest_data.get('POP', 'N/A');
                    wsd = latest_data.get('WSD', 'N/A');
                    reh = latest_data.get('REH', 'N/A')
                    clothing_recommendation = recommend_clothing(temp)
                    st.info(f"👕 **오늘의 옷차림 추천:** {clothing_recommendation}")
                    st.metric(label="현재 기온", value=f"{temp}°C")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("하늘 상태", sky);
                    col2.metric("강수 형태", pty);
                    col3.metric("강수 확률", f"{pop}%")
                    col4, col5 = st.columns(2)
                    col4.metric("풍속", f"{wsd} m/s");
                    col5.metric("습도", f"{reh}%")
                    with st.expander("시간대별 상세 예보 보기"): st.dataframe(df)

# 3.3. 나의 옷장 페이지
elif st.session_state.page == "closet":
    st.title("👚 나의 옷장")
    st.subheader("내가 분석한 옷")
    if not st.session_state.my_closet:
        st.info("아직 옷장에 저장된 옷이 없습니다.")
    else:
        cols = st.columns(4)
        for i, item in enumerate(st.session_state.my_closet):
            with cols[i % 4]:
                st.image(item["image"], caption=item["name"])
                if st.button("삭제", key=f"delete_closet_{i}", use_container_width=True):
                    st.session_state.my_closet.pop(i);
                    st.rerun()
    st.write("---")
    st.subheader("저장된 추천 코디")
    if not st.session_state.saved_images:
        st.info("아직 저장된 추천 코디가 없습니다.")
    else:
        cols = st.columns(4)
        for i, url in enumerate(st.session_state.saved_images):
            with cols[i % 4]:
                st.image(url, caption=f"저장된 코디 {i + 1}")
                if st.button("삭제", key=f"delete_saved_{i}", use_container_width=True):
                    st.session_state.saved_images.pop(i);
                    st.rerun()

# 3.4. 퍼스널 컬러 분석 페이지
elif st.session_state.page == "personal_color":
    st.title("🎨 AI 퍼스널 컬러 진단")
    st.write("메이크업을 하지 않은 정면 얼굴 사진을 직접 촬영하거나 업로드해주세요.")
    if "show_camera" not in st.session_state: st.session_state.show_camera = False
    if st.button("사진 촬영 📸", use_container_width=True): st.session_state.show_camera = not st.session_state.show_camera
    uploaded_photo = st.file_uploader("또는 얼굴 사진 업로드 📂", type=["jpg", "jpeg", "png"])
    if st.session_state.show_camera:
        camera_photo = st.camera_input("카메라")
        if camera_photo: st.session_state.face_photo_object = camera_photo; st.session_state.show_camera = False; st.rerun()
    else:
        camera_photo = None
    if uploaded_photo: st.session_state.face_photo_object = uploaded_photo
    if st.session_state.face_photo_object:
        st.image(st.session_state.face_photo_object, caption="분석할 얼굴 이미지", width=300)
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("AI로 분석하기", use_container_width=True):
                with st.spinner("AI가 퍼스널 컬러를 분석 중입니다..."):
                    analysis_text = analyze_personal_color(st.session_state.face_photo_object)
                    if analysis_text:
                        st.markdown(analysis_text)
                        match = re.search(r"진단 결과\s*:\s*(.+)", analysis_text)
                        if match and match.group(1).strip() in personal_color_options:
                            st.session_state.analyzed_color = match.group(1).strip()
                    else:
                        st.error("분석에 실패했습니다.")
        with col2:
            if st.button("이미지 삭제", use_container_width=True):
                st.session_state.face_photo_object = None
                if "analyzed_color" in st.session_state: del st.session_state.analyzed_color
                st.rerun()
    if "analyzed_color" in st.session_state and st.session_state.face_photo_object:
        if st.button(f"'{st.session_state.analyzed_color}' 결과를 내 정보에 적용하기"):
            st.session_state.피부_톤 = st.session_state.analyzed_color
            if 'user_info' in st.session_state: st.session_state.user_info['피부_톤'] = st.session_state.analyzed_color
            st.success(f"'{st.session_state.피부_톤}'이 맞춤 정보에 적용되었습니다.")

# 3.5. 패션 데이터 분석 페이지
elif st.session_state.page == "analytics":
    st.title("📊 패션 데이터 분석 대시보드")
    st.markdown("### 📈 나의 패션 프로필 분석")
    st.caption("'옷 분석하기' 데이터를 기반으로 생성됩니다.")
    if not st.session_state.user_activity_log:
        st.info("아직 분석된 옷 데이터가 없습니다.")
    else:
        all_tags = [tag for item in st.session_state.user_activity_log for tag in item.get('style_tags', [])]
        tag_counts = pd.Series(all_tags).value_counts()
        all_colors = [item.get('color', 'N/A') for item in st.session_state.user_activity_log]
        color_counts = pd.Series(all_colors).value_counts()
        col1, col2 = st.columns(2)
        with col1:
            if not tag_counts.empty:
                fig_style = px.bar(tag_counts, y=tag_counts.index, x=tag_counts.values, orientation='h',
                                   title="나의 스타일 선호도 분석", labels={'y': '스타일', 'x': '분석 횟수'}, color=tag_counts.values,
                                   color_continuous_scale='viridis')
                st.plotly_chart(fig_style, use_container_width=True)
            else:
                st.info("스타일 데이터가 부족합니다.")
        with col2:
            if not color_counts.empty:
                fig_color = px.pie(values=color_counts.values, names=color_counts.index, title="분석된 옷 색상 분포",
                                   color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_color, use_container_width=True)
            else:
                st.info("색상 데이터가 부족합니다.")
    st.markdown("### 🌍 최신 패션 트렌드 분석 (Google Trends)")
    with st.spinner("Google Trends에서 최신 데이터를 가져오는 중..."):
        trends_df = get_google_trends_data()
        if not trends_df.empty:
            st.line_chart(trends_df);
            st.caption("지난 1년간의 주요 패션 키워드에 대한 관심도 변화입니다.")
        else:
            st.warning("트렌드 데이터를 가져오는 데 실패했습니다.")

# 3.6. 옷 입혀보기 AI 페이지
elif st.session_state.page == "vton":
    st.title("🔎 옷 입혀보기 AI (Virtual Try-On)")
    site_url = "https://huggingface.co/spaces/Kwai-Kolors/Kolors-Virtual-Try-On"
    st.write(f"아래 링크를 통해 외부 가상 피팅 사이트를 이용할 수 있습니다.")
    st.markdown(f"**[가상 피팅 사이트로 이동하기]({site_url})**", unsafe_allow_html=True)
    st.info("⚠️ 외부 사이트는 안정적이지 않을 수 있으며, 로딩에 시간이 걸릴 수 있습니다")