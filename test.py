import streamlit as st
from PIL import Image
import google.generativeai as genai
import json
from openai import OpenAI
import re
import time
import os
import pandas as pd  # 데이터 시각화를 위한 라이브러리 추가
import plotly.express as px  # 데이터 시각화를 위한 라이브러리 추가

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="맞춤 패션 추천 시스템",
    page_icon="👕",
    layout="wide"
)

# --- API 키 설정 ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("🚨 API 키를 설정해주세요! .streamlit/secrets.toml 파일을 확인하세요.")
    st.stop()

# --- Gemini 모델 초기화 ---
llm_model = genai.GenerativeModel('gemini-1.5-flash')


# --- (이하 모든 함수 정의는 변경 없음) ---
def save_image(directory, file):
    if not os.path.exists(directory): os.makedirs(directory)
    filename = f"capture_{int(time.time())}.jpg"
    filepath = os.path.join(directory, filename)
    with open(filepath, 'wb') as f: f.write(file.getbuffer())
    return filepath


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
    당신은 친절하고 스타일리시한 AI 패션 어드바이저입니다. 고객 정보, 의류 아이템, 주어진 상황을 바탕으로 최고의 코디를 추천해주세요. **중요: 답변의 가독성을 높이기 위해 다음 규칙을 반드시 지켜주세요.** 1. 각 코디 제안의 제목은 Markdown의 `##`를 사용하여 크고 굵게 표시해주세요. 2. 설명에 어울리는 이모티콘(👕,👖,👟,✨ 등)을 자유롭게 사용해주세요. 3. 의류 아이템, 색상, 스타일 등 중요한 키워드는 `<span style='color: #87CEEB;'>키워드</span>` 와 같이 HTML 태그를 사용해 색상을 입혀 강조해주세요.
    ## 🧑‍💻 고객 정보:
    - 키: {user_info['키']}cm, 몸무게: {user_info['몸무게']}kg, 피부 톤: {user_info['피부_톤']}, 선호 스타일: {', '.join(user_info['선호_스타일'])}
    ## 👚 분석된 의류 아이템:
    - 종류: {clothing_info['item_type']}, 카테고리: {clothing_info['category']}, 색상: {clothing_info['color']}, 패턴: {clothing_info['pattern']}
    ## 🏞️ 주어진 상황:
    - {situation}
    ## 요청 사항:
    1. 위 정보를 종합하여, 총 **두 가지 스타일의 완성된 코디**를 추천하고, 각 코디를 추천한 이유를 친절하게 설명해주세요.
    2. 각 코디 설명 후, DALL-E가 이미지를 생성할 수 있도록, **주어진 상황을 반영**하여 해당 코디를 입은 모델의 모습을 상세하고 사실적으로 묘사하는 **영어 프롬프트**를 다음 형식으로 제공해주세요: `IMAGE_PROMPT_1: [첫 번째 코디에 대한 상세한 영어 묘사]`, `IMAGE_PROMPT_2: [두 번째 코디에 대한 상세한 영어 묘사]`
    """
    try:
        response = llm_model.generate_content(prompt)
        recommendation_text = response.text
        image_prompts = re.findall(r"IMAGE_PROMPT_\d:\s*(.*)", recommendation_text)
        display_text = re.sub(r"IMAGE_PROMPT_\d:\s*.*", "", recommendation_text).strip()
        return display_text, image_prompts
    except Exception as e:
        st.error(f"코디 추천 중 오류 발생: {e}");
        return None, None


def analyze_personal_color(face_image):
    img = Image.open(face_image)
    prompt = """
    당신은 전문 퍼스널 컬러 컨설턴트입니다. 이 인물의 얼굴 사진을 보고, 피부의 언더톤, 머리카락과 눈동자 색의 대비 등을 종합적으로 분석하여 가장 가능성이 높은 퍼스널 컬러를 진단해주세요.
    답변은 아래 형식과 같이 **진단 결과**와 **진단 근거**를 명확히 구분하여 작성해주세요. 진단 근거는 2~3가지 핵심적인 이유를 간결한 불릿 포인트로 설명해야 합니다.
    **진단 결과**: [봄 웜톤, 여름 쿨톤, 가을 웜톤, 겨울 쿨톤 중 하나]
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


# --- 사이드바 내비게이션 UI ---
st.sidebar.title("나만의 맞춤 패션 추천")
if st.sidebar.button("🏠 메인 페이지", use_container_width=True):
    st.session_state.page = "main"
if st.sidebar.button("🎨 퍼스널 컬러 분석", use_container_width=True):
    st.session_state.page = "personal_color"
if st.sidebar.button("🔎 옷 입혀보기 AI", use_container_width=True):
    st.session_state.page = "vton"

# --- 페이지 상태 초기화 ---
if "page" not in st.session_state:
    st.session_state.page = "main"
if "face_photo_object" not in st.session_state:
    st.session_state.face_photo_object = None

personal_color_options = ["봄 웜톤", "여름 쿨톤", "가을 웜톤", "겨울 쿨톤"]

# 1. 메인 페이지
if st.session_state.page == "main":
    st.title("👕 나의 맞춤 패션 추천")
    # --- ✨ (수정) 탭 추가 ---
    tab1, tab2, tab3, tab_effects = st.tabs(["⚙️ 나의 맞춤 정보", "🧠 옷 분석하기", "✨ 코디 추천받기", "📊 기대 효과"])

    with tab1:
        st.subheader("체형")
        키 = st.number_input("키", min_value=100, max_value=250, value=st.session_state.get("키", 170), step=1)
        몸무게 = st.number_input("몸무게", min_value=30, max_value=200, value=st.session_state.get("몸무게", 62), step=1)
        st.subheader("피부")
        default_index = personal_color_options.index(st.session_state.get("피부_톤", personal_color_options[0]))
        피부_톤 = st.radio("피부 톤", personal_color_options, index=default_index, key="skin_tone_radio")
        st.subheader("패션")
        fashion_styles_options = ["캐주얼", "시크", "시티보이", "클래식", "에스닉", "걸리시", "고프코어", "미니멀", "프레피", "리조트", "레트로", "로맨틱",
                                  "스포티", "스트리트", "워크웨어"]
        선호_스타일 = st.multiselect("선호하는 패션 스타일을 모두 선택해주세요.", fashion_styles_options,
                                default=st.session_state.get("선호_스타일", []))
        if st.button("내 정보 저장하기"):
            st.session_state.user_info = {"키": 키, "몸무게": 몸무게, "피부_톤": 피부_톤, "선호_스타일": 선호_스타일}
            st.session_state.키, st.session_state.몸무게, st.session_state.피부_톤, st.session_state.선호_스타일 = 키, 몸무게, 피부_톤, 선호_스타일
            st.success("정보가 저장되었습니다!")

    with tab2:
        st.subheader("👚 분석할 옷 사진 업로드")
        uploaded_file = st.file_uploader("코디를 추천받고 싶은 옷 사진을 올려주세요.", type=["jpg", "jpeg", "png"], key="cloth_uploader")
        if uploaded_file:
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, caption="업로드된 옷 이미지", use_column_width=True)
            with col2:
                if st.button("AI로 옷 분석 시작하기", use_container_width=True):
                    if 'user_info' not in st.session_state:
                        st.error("먼저 '나의 맞춤 정보' 탭에서 정보를 저장해주세요!")
                    else:
                        with st.spinner("AI가 이미지를 분석하고 있습니다... 🧠"):
                            analysis_result = analyze_clothing_image(uploaded_file)
                            if analysis_result:
                                st.success("분석 완료!");
                                st.subheader("✅ AI 분석 결과")
                                st.write(f"**의류 종류**: {analysis_result.get('item_type', 'N/A')}")
                                st.write(f"**카테고리**: {analysis_result.get('category', 'N/A')}")
                                st.write(f"**색상**: {analysis_result.get('color', 'N/A')}")
                                st.write(f"**패턴**: {analysis_result.get('pattern', 'N/A')}")
                                tags = analysis_result.get('style_tags', [])
                                st.write(f"**스타일 태그**: {', '.join(tags) if tags else 'N/A'}")
                                st.session_state.analysis_result = analysis_result
                                st.info("'코디 추천받기' 탭으로 이동하여 추천을 받아보세요!")

    with tab3:
        st.subheader("✨ AI 코디 추천 결과")
        if 'analysis_result' in st.session_state and 'user_info' in st.session_state:
            st.write("어떤 상황이나 장소에서 입을 코디를 추천받고 싶으신가요?")
            situation_input = st.text_input("예시: 주말 오후 카페에서, 도서관에서 공부할 때, 친구들과 저녁 약속",
                                            placeholder="원하는 상황을 자유롭게 입력해주세요...")
            if st.button("AI 코디 추천 및 이미지 생성", use_container_width=True):
                situation = situation_input if situation_input else "일상적인 상황"
                with st.spinner("AI 스타일리스트가 당신만을 위한 코디와 이미지를 만들고 있습니다... ✨"):
                    recommendation_text, image_prompts = get_cody_recommendation_with_image(
                        st.session_state.user_info, st.session_state.analysis_result, situation
                    )
                if recommendation_text:
                    st.subheader("AI 스타일리스트의 추천")
                    st.markdown(recommendation_text, unsafe_allow_html=True)
                    if image_prompts:
                        st.subheader("추천 코디 시각화")
                        cols = st.columns(len(image_prompts) or 1)
                        for i, prompt in enumerate(image_prompts):
                            with cols[i % len(cols)]:
                                with st.spinner(f"코디 이미지 {i + 1} 생성 중..."):
                                    image_url = generate_image_with_dalle(prompt)
                                    if image_url:
                                        st.image(image_url, caption=f"추천 코디 {i + 1}", use_column_width=True)
                                    else:
                                        st.warning(f"추천 코디 {i + 1} 이미지 생성 실패.")
                else:
                    st.error("코디 추천에 실패했습니다.")
        else:
            st.warning("먼저 '옷 분석하기'를 완료해주세요.")

    # --- ✨ (신규) 기대 효과 탭 ---
    with tab_effects:
        st.subheader("📊 프로젝트 기대 효과 시각화")
        st.markdown("본 AI 추천 시스템 도입 시 예상되는 **시간 및 자원 절약 효과**를 시각화한 예시입니다.")
        st.write("---")

        col1, col2 = st.columns(2)
        with col1:
            # --- 1. 사용자 시간 단축 효과 (Bar Chart) ---
            st.markdown("#### ⏰ 코디 결정 시간 단축")
            time_data = pd.DataFrame({
                '구분': ['앱 사용 전', '앱 사용 후'],
                '평균 소요 시간 (분)': [20, 2]
            })
            st.bar_chart(time_data.set_index('구분'))
            st.caption("AI 추천을 통해 사용자의 일일 코디 결정 시간을 평균 20분에서 2분으로 단축하는 효과를 기대할 수 있습니다.")

        with col2:
            # --- 2. 산업적 반품률 감소 효과 (Bar Chart) ---
            st.markdown("#### 📉 쇼핑몰 반품률 감소")
            return_data = pd.DataFrame({
                '구분': ['일반 쇼핑몰', 'AI 추천 도입 후'],
                '평균 반품률 (%)': [25, 10]
            })
            st.bar_chart(return_data.set_index('구분'), color="#ff8c00")
            st.caption("개인화된 추천으로 구매 만족도를 높여, 패션 쇼핑몰의 평균 반품률을 25%에서 10% 수준으로 크게 낮출 수 있습니다.")

        st.write("---")

        # --- 3. 의류 자원 절약 효과 (Pie Chart) ---
        st.markdown("#### 💰 '옷장 활용률' 증가로 자원 절약")
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**앱 사용 전**")
            pie_data_before = pd.DataFrame({
                '항목': ['자주 입는 옷', '거의 입지 않는 옷'],
                '비율': [60, 40]
            })
            fig_before = px.pie(pie_data_before, values='비율', names='항목', title='기존 옷장 활용률',
                                color_discrete_sequence=['#87CEEB', '#D3D3D3'])
            st.plotly_chart(fig_before, use_container_width=True)

        with col4:
            st.markdown("**앱 사용 후**")
            pie_data_after = pd.DataFrame({
                '항목': ['자주 입는 옷', '거의 입지 않는 옷'],
                '비율': [85, 15]
            })
            fig_after = px.pie(pie_data_after, values='비율', names='항목', title='개선된 옷장 활용률',
                               color_discrete_sequence=['#87CEEB', '#D3D3D3'])
            st.plotly_chart(fig_after, use_container_width=True)

        st.caption("AI 코디 추천을 통해 기존에 활용하지 못했던 옷들의 조합을 찾아내어, 실질적인 옷장 활용률을 60%에서 85%까지 끌어올려 의류 낭비를 줄이는 효과를 기대합니다.")


# 2. 퍼스널 컬러 분석 페이지
elif st.session_state.page == "personal_color":
    # ... (퍼스널 컬러 분석 페이지 코드는 변경 없음) ...
    st.title("🎨 AI 퍼스널 컬러 진단")
    st.write("메이크업을 하지 않은 정면 얼굴 사진을 직접 촬영하거나 업로드해주세요.")
    if "face_photo_object" not in st.session_state: st.session_state.face_photo_object = None
    if "show_camera" not in st.session_state: st.session_state.show_camera = False
    if st.button("사진 촬영 📸", use_container_width=True): st.session_state.show_camera = not st.session_state.show_camera
    uploaded_photo = st.file_uploader("또는 얼굴 사진 업로드 📂", type=["jpg", "jpeg", "png"])
    if st.session_state.show_camera:
        camera_photo = st.camera_input("카메라")
        if camera_photo:
            st.session_state.face_photo_object = camera_photo
            st.session_state.show_camera = False
            st.rerun()
    if uploaded_photo: st.session_state.face_photo_object = uploaded_photo
    if st.session_state.face_photo_object:
        st.image(st.session_state.face_photo_object, caption="분석할 얼굴 이미지", width=300)
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("AI로 분석하기", use_container_width=True):
                face_photo_to_analyze = st.session_state.face_photo_object
                save_dir = "captured_images"
                saved_filepath = save_image(save_dir, face_photo_to_analyze)
                st.success(f"사진 저장 완료: {saved_filepath}")
                with st.spinner("AI가 퍼스널 컬러를 분석 중입니다..."):
                    analysis_text = analyze_personal_color(face_photo_to_analyze)
                    if analysis_text:
                        st.markdown(analysis_text)
                        match = re.search(r"진단 결과\s*:\s*(.+)", analysis_text)
                        if match and match.group(1).strip() in personal_color_options:
                            st.session_state.analyzed_color = match.group(1).strip()
                        else:
                            st.warning("분석 결과에서 명확한 타입을 찾지 못했습니다.")
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

# 3. 옷 입혀보기 AI 페이지
elif st.session_state.page == "vton":
    # ... (옷 입혀보기 AI 페이지 코드는 변경 없음) ...
    st.title("🔎 옷 입혀보기 AI (Virtual Try-On)")
    site_url = "https://huggingface.co/spaces/levihsu/OOTDiffusion"
    st.write(f"아래 링크를 통해 외부 가상 피팅 사이트를 이용할 수 있습니다.")
    st.markdown(f"**[OOTDiffusion 가상 피팅 사이트로 이동하기]({site_url})**", unsafe_allow_html=True)
    st.info("⚠️ 외부 사이트는 안정적이지 않을 수 있으며, 로딩에 시간이 걸릴 수 있습니다.")