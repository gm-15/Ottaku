import streamlit as st
from PIL import Image
import google.generativeai as genai
import json
from openai import OpenAI
import re  # JSON 추출을 위해 re 라이브러리 추가

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="맞춤 패션 추천 시스템",
    page_icon="👕",
    layout="wide"
)

# --- Google Gemini API 설정 ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("🚨 Google API 키를 설정해주세요! .streamlit/secrets.toml 파일을 확인하세요.")
    st.stop()

# --- OpenAI API 설정 ---
try:
    openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("🚨 OpenAI API 키를 설정해주세요! .streamlit/secrets.toml 파일을 확인하세요.")
    st.stop()

# --- Gemini 모델 초기화 ---
llm_model = genai.GenerativeModel('gemini-1.5-flash')


# --- DALL-E 이미지 생성 함수 ---
def generate_image_with_dalle(prompt):
    """
    DALL-E 모델을 사용하여 이미지를 생성하는 함수
    """
    try:
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        return image_url
    except Exception as e:
        st.error(f"DALL-E 이미지 생성 중 오류가 발생했습니다: {e}")
        return None


# --- (수정된) Gemini 호출 함수 (3번 과정) ---
def analyze_clothing_image(uploaded_image):
    """
    Gemini AI에게 옷 이미지를 보내 분석을 요청하고, 결과를 JSON으로 받는 함수
    """
    img = Image.open(uploaded_image)
    prompt = """
    당신은 패션 스타일리스트이자 의류 분석 전문가입니다.
    이 이미지에 있는 옷을 분석해서 아래 JSON 형식에 맞춰 답변해주세요.
    각 항목에 대해 가장 적절한 단 하나의 값만 선택해주세요.
    만약 옷이 아닌 다른 이미지가 들어오면, 모든 값에 "N/A"를 채워주세요.
    **중요: 답변에는 JSON 코드 외에 어떤 설명이나 인사도 포함하지 말고, 오직 JSON 객체만 응답해야 합니다.**

    {
      "item_type": "상의, 하의, 아우터, 신발, 액세서리 중 하나",
      "category": "티셔츠, 셔츠, 청바지, 스커트, 재킷, 스니커즈 등 구체적인 카테고리",
      "color": "옷의 가장 주된 색상 (예: 네이비 블루)",
      "pattern": "솔리드(단색), 스트라이프, 체크, 도트, 플로럴, 프린팅 등",
      "style_tags": ["캐주얼", "미니멀", "스트리트", "포멀", "스포티"]
    }
    """
    try:
        response = llm_model.generate_content([prompt, img])

        # --- ✨ JSON 추출 로직 강화 ---
        # AI 응답에서 JSON 블록만 찾아내기
        match = re.search(r"\{.*\}", response.text, re.DOTALL)
        if match:
            json_text = match.group(0)
            return json.loads(json_text)
        else:
            st.error("AI 응답에서 JSON을 찾을 수 없습니다.")
            st.code(response.text, language="text")  # AI의 실제 응답을 보여줘서 디버깅에 도움
            return None
        # -----------------------------

    except json.JSONDecodeError:
        st.error("AI가 유효하지 않은 JSON 형식으로 응답했습니다. 다시 시도해주세요.")
        st.code(response.text, language="text")  # AI의 실제 응답을 보여줘서 디버깅에 도움
        return None
    except Exception as e:
        st.error(f"이미지 분석 중 오류가 발생했습니다: {e}")
        return None


# --- Gemini 호출 함수 (4번 과정) + 이미지 생성 ---
def get_cody_recommendation_with_image(user_info, clothing_info):
    """
    사용자 정보와 옷 분석 결과를 바탕으로 Gemini에게 코디 추천을 요청하고,
    DALL-E를 이용하여 추천된 코디 이미지를 생성하는 함수
    """
    # 이미지 생성을 위한 프롬프트를 별도로 생성하도록 요청 강화
    prompt = f"""
    당신은 개인의 특성과 취향을 고려하여 패션을 추천해주는 전문 패션 스타일리스트입니다.

    ## 고객 정보:
    - **키**: {user_info['키']}cm, **몸무게**: {user_info['몸무게']}kg
    - **피부 톤**: {user_info['피부_톤']}
    - **선호 스타일**: {', '.join(user_info['선호_스타일'])}

    ## 분석된 의류 아이템:
    - **종류**: {clothing_info['item_type']}, **카테고리**: {clothing_info['category']}, **색상**: {clothing_info['color']}, **패턴**: {clothing_info['pattern']}

    ## 요청 사항:
    1.  위 정보를 바탕으로, 총 **두 가지 스타일의 완성된 코디**를 추천하고, 각 코디를 추천한 이유를 친절하게 설명해주세요. (상의, 하의, 신발, 액세서리 포함)
    2.  각 코디 추천 설명이 끝난 후, DALL-E가 이미지를 생성할 수 있도록, 해당 코디를 입은 모델의 모습을 상세하고 사실적으로 묘사하는 **영어 프롬프트**를 다음 형식으로 제공해주세요:
        `IMAGE_PROMPT_1: [첫 번째 코디에 대한 상세한 영어 묘사]`
        `IMAGE_PROMPT_2: [두 번째 코디에 대한 상세한 영어 묘사]`

    답변은 마크다운 형식을 사용하여 가독성 좋게 작성해주세요.
    """
    try:
        response = llm_model.generate_content(prompt)
        recommendation_text = response.text

        # 정규표현식을 사용해 이미지 프롬프트를 더 안정적으로 추출
        image_prompts = re.findall(r"IMAGE_PROMPT_\d:\s*(.*)", recommendation_text)

        image_urls = []
        if image_prompts:
            for img_prompt in image_prompts:
                with st.spinner(f"코디 이미지 생성 중... (프롬프트: {img_prompt[:30]}...)"):
                    image_url = generate_image_with_dalle(img_prompt.strip())
                    image_urls.append(image_url)

        # 텍스트 결과에서 이미지 프롬프트 부분은 제외하고 보여주기
        display_text = re.sub(r"IMAGE_PROMPT_\d:\s*.*", "", recommendation_text).strip()

        return display_text, image_urls
    except Exception as e:
        st.error(f"코디 추천 및 이미지 생성 중 오류가 발생했습니다: {e}")
        return None, None


# --- Streamlit 앱 UI (이하 동일) ---
st.title("👕 나의 맞춤 패션 추천")

# 탭 구성
tab1, tab2, tab3, tab4= st.tabs(["⚙️ 나의 맞춤 정보", "🧠 옷 분석하기", "✨ 코디 추천받기", "🤖 옷 입혀보기 AI"])

with tab1:
    st.subheader("체형")
    st.write("나의 비슷한 체형의 후기를 모아볼 수 있어요.")
    키 = st.number_input("키", min_value=100, max_value=250, value=170, step=1)
    몸무게 = st.number_input("몸무게", min_value=30, max_value=200, value=62, step=1)

    st.subheader("피부")
    st.write("나와 비슷한 피부 타입 후기를 모아볼 수 있어요.")
    피부_톤 = st.radio("피부 톤", ["봄 웜톤", "여름 쿨톤", "가을 웜톤", "겨울 쿨톤"])

    st.subheader("패션")
    st.write("선호하는 패션 스타일을 다양하게 제안해 드릴게요.")
    fashion_styles = [
        "캐주얼", "시크", "시티보이", "클래식", "에스닉",
        "걸리시", "고프코어", "미니멀", "프레피", "리조트",
        "레트로", "로맨틱", "스포티", "스트리트", "워크웨어"
    ]
    선호_스타일 = st.multiselect("선호하는 패션 스타일을 모두 선택해주세요.", fashion_styles)

    if st.button("내 정보 저장하기"):
        st.session_state.user_info = {
            "키": 키, "몸무게": 몸무게, "피부_톤": 피부_톤, "선호_스타일": 선호_스타일
        }
        st.success("정보가 저장되었습니다! '옷 분석하기' 탭으로 이동해주세요.")

with tab2:
    st.subheader("👚 분석할 옷 사진 업로드")
    uploaded_file = st.file_uploader("코디를 추천받고 싶은 옷 사진을 올려주세요.", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="업로드된 옷 이미지", use_column_width=True)

        with col2:
            if st.button("AI로 옷 분석 시작하기", use_container_width=True):
                if 'user_info' not in st.session_state:
                    st.error("먼저 '나의 맞춤 정보' 탭에서 정보를 저장해주세요!")
                else:
                    with st.spinner("AI가 이미지를 분석하고 있습니다... 🧠"):
                        analysis_result = analyze_clothing_image(uploaded_file)
                        if analysis_result:
                            st.success("분석이 완료되었습니다!")
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
        st.write("아래 버튼을 눌러 AI 코디 추천과 함께 이미지를 받아보세요.")

        if st.button("AI 코디 추천 및 이미지 생성", use_container_width=True):
            with st.spinner("AI 스타일리스트가 당신만을 위한 코디를 만들고 있습니다... ✨"):
                recommendation_text, image_urls = get_cody_recommendation_with_image(
                    st.session_state.user_info,
                    st.session_state.analysis_result
                )
            if recommendation_text:
                st.subheader("AI 스타일리스트의 추천")
                st.markdown(recommendation_text)

                if image_urls:
                    st.subheader("추천 코디 시각화")
                    # 한 줄에 두 개의 이미지를 표시하도록 설정
                    if len(image_urls) > 0:
                        cols = st.columns(2)
                        for i, url in enumerate(image_urls):
                            if url:
                                with cols[i % 2]:
                                    st.image(url, caption=f"추천 코디 {i + 1}", use_column_width=True)
                            else:
                                with cols[i % 2]:
                                    st.warning(f"추천 코디 {i + 1} 이미지 생성 실패.")
            else:
                st.error("코디 추천에 실패했습니다.")
    else:
        st.warning("먼저 '나의 맞춤 정보'를 저장하고 '옷 분석하기'를 완료해주세요.")

with tab4:
    st.subheader("🤖 옷 입혀보기 AI ")
    site_url = "https://huggingface.co/spaces/Kwai-Kolors/Kolors-Virtual-Try-On"

    st.subheader(f"삽입된 사이트: {site_url}")