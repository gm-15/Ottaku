import streamlit as st
from PIL import Image
import google.generativeai as genai
import json

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

# --- Gemini 모델 초기화 ---
# 여러 함수에서 사용하므로 전역적으로 한 번만 설정
llm_model = genai.GenerativeModel('gemini-1.5-flash')


# --- Gemini 호출 함수 (3번 과정) ---
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
        json_text = response.text.strip().lstrip("```json").rstrip("```")
        return json.loads(json_text)
    except json.JSONDecodeError:
        st.error("AI가 JSON 형식으로 응답하지 않았습니다. 다시 시도해주세요.")
        return None
    except Exception as e:
        st.error(f"이미지 분석 중 오류가 발생했습니다: {e}")
        return None


# --- (신규) Gemini 호출 함수 (4번 과정) ---
def get_cody_recommendation(user_info, clothing_info):
    """
    사용자 정보와 옷 분석 결과를 바탕으로 Gemini에게 코디 추천을 요청하는 함수
    """
    prompt = f"""
    당신은 개인의 특성과 취향을 고려하여 패션을 추천해주는 전문 패션 스타일리스트입니다.

    ## 고객 정보:
    - **키**: {user_info['키']}cm
    - **몸무게**: {user_info['몸무게']}kg
    - **피부 톤**: {user_info['피부_톤']}
    - **선호 스타일**: {', '.join(user_info['선호_스타일'])}

    ## 분석된 의류 아이템:
    - **종류**: {clothing_info['item_type']}
    - **카테고리**: {clothing_info['category']}
    - **색상**: {clothing_info['color']}
    - **패턴**: {clothing_info['pattern']}

    ## 요청 사항:
    위 고객 정보와 의류 아이템 정보를 바탕으로, 총 **두 가지 스타일의 완성된 코디**를 추천해주세요.
    각 코디는 다음 항목을 반드시 포함해야 합니다:
    - 상의
    - 하의
    - 신발
    - 액세서리 (1~2개)

    각 코디에 대해 왜 그렇게 추천했는지, 고객의 특성과 어떻게 잘 어울리는지 이유를 친절하게 설명해주세요.
    답변은 마크다운 형식을 사용하여 가독성 좋게 작성해주세요.
    """
    try:
        response = llm_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"코디 추천 중 오류가 발생했습니다: {e}")
        return None


# --- Streamlit 앱 UI ---
st.title("👕 나의 맞춤 패션 추천")

# 탭 구성
tab1, tab2, tab3 = st.tabs(["⚙️ 나의 맞춤 정보", "🧠 옷 분석하기", "✨ 코디 추천받기"])

# --- 1번 과정: 사용자 정보 입력 ---
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
        # 정보를 세션 상태에 저장
        st.session_state.user_info = {
            "키": 키, "몸무게": 몸무게, "피부_톤": 피부_톤, "선호_스타일": 선호_스타일
        }
        st.success("정보가 저장되었습니다! '옷 분석하기' 탭으로 이동해주세요.")

# --- 2, 3번 과정: 옷 사진 업로드 및 분석 ---
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

# --- 4번 과정: 코디 추천 ---
with tab3:
    st.subheader("✨ AI 코디 추천 결과")

    # 이전 단계의 정보가 st.session_state에 저장되어 있는지 확인
    if 'analysis_result' in st.session_state and 'user_info' in st.session_state:
        st.write("아래 버튼을 눌러 AI 코디 추천을 받아보세요.")

        if st.button("AI 코디 추천받기", use_container_width=True):
            with st.spinner("AI 스타일리스트가 당신만을 위한 코디를 만들고 있습니다... ✨"):
                recommendation = get_cody_recommendation(
                    st.session_state.user_info,
                    st.session_state.analysis_result
                )
                if recommendation:
                    st.markdown(recommendation)  # 마크다운 형식으로 결과 출력
    else:
        st.warning("먼저 '나의 맞춤 정보'를 저장하고 '옷 분석하기'를 완료해주세요.")