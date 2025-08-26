import streamlit as st
from PIL import Image
import google.generativeai as genai
import os


OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]


# --- 페이지 설정 ---
st.set_page_config(
    page_title="AI Virtual Try-On",
    page_icon="👕",
    layout="wide"
)

# --- Google Gemini API 키 설정 ---
# Streamlit의 secrets 관리 기능을 사용하는 것이 안전합니다.
# secrets.toml 파일에 GOOGLE_API_KEY = "YOUR_API_KEY" 형식으로 저장하세요.
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("🚨 Google API 키를 설정해주세요! Streamlit의 secrets 관리 기능을 사용하는 것이 안전합니다.")
    st.stop()


# --- AI 모델 설정 ---
# 이미지 입력을 지원하는 Gemini Pro Vision 모델을 사용합니다.
model = genai.GenerativeModel('gemini-2.0-flash')


def get_gemini_response(input_prompt, person_image, clothing_image):
    """
    Gemini AI 모델에 요청을 보내고 응답을 받아오는 함수
    """
    if person_image and clothing_image:
        try:
            # 이미지 파일로부터 바이트 데이터를 읽습니다.
            person_image_bytes = person_image.getvalue()
            clothing_image_bytes = clothing_image.getvalue()

            image_parts = [
                {
                    "mime_type": "image/jpeg",
                    "data": person_image_bytes
                },
                {
                    "mime_type": "image/jpeg",
                    "data": clothing_image_bytes
                },
            ]

            # 모델에 프롬프트와 이미지를 함께 전달
            response = model.generate_content([input_prompt, *image_parts])
            return response.text
        except Exception as e:
            return f"오류가 발생했습니다: {e}"
    else:
        return "이미지를 모두 업로드해주세요."

# --- 앱 UI 구성 ---
st.title("👕 AI 가상 피팅 (Virtual Try-On)")
st.write("Gemini AI를 사용하여 사진에 옷을 입혀보세요!")
st.write("---")


# --- 사이드바 ---
with st.sidebar:
    st.header("사용 방법")
    st.info(
        """
        1. **사람 이미지 업로드**: 정면을 바라보는 선명한 인물 사진을 올려주세요.
        2. **옷 이미지 업로드**: 입어보고 싶은 옷 이미지를 올려주세요. (배경이 깔끔하면 더 좋습니다)
        3. **'가상 피팅 시작'** 버튼을 클릭하여 결과를 확인하세요!
        """
    )
    st.warning("⚠️ **주의사항**: AI가 생성한 이미지는 실제와 다를 수 있으며, 부자연스러운 부분이 있을 수 있습니다.")


# --- 메인 화면 ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 1. 사람 이미지 업로드")
    person_image_file = st.file_uploader("자신의 사진을 업로드하세요.", type=["jpg", "jpeg", "png"])
    if person_image_file:
        person_image = Image.open(person_image_file)
        st.image(person_image, caption="업로드된 사람 이미지", use_column_width=True)

with col2:
    st.subheader("👚 2. 옷 이미지 업로드")
    clothing_image_file = st.file_uploader("입어볼 옷 사진을 업로드하세요.", type=["jpg", "jpeg", "png"])
    if clothing_image_file:
        clothing_image = Image.open(clothing_image_file)
        st.image(clothing_image, caption="업로드된 옷 이미지", use_column_width=True)


st.write("---")

# --- 프롬프트 정의 및 실행 버튼 ---
# Gemini 모델이 이미지를 더 잘 이해하고 원하는 결과물을 생성하도록 구체적인 지시를 내립니다.
input_prompt = """
첫 번째 이미지 속 인물에게 두 번째 이미지의 옷을 자연스럽게 입혀주세요.
인물의 원래 체형, 자세, 배경은 최대한 유지하면서, 옷의 스타일, 색상, 질감이 인물에게 잘 어울리도록 합성해주세요.
최종 결과물은 옷을 입은 인물의 모습만 보여주세요.
"""

if st.button("✨ 가상 피팅 시작!", use_container_width=True):
    if person_image_file and clothing_image_file:
        with st.spinner("AI가 열심히 옷을 입히고 있어요... 잠시만 기다려주세요! 🤖"):
            # 실제로는 Gemini Pro Vision 모델은 이미지 '이해'에 특화되어 있어,
            # 이미지를 직접 수정하거나 새로운 이미지를 '생성'하는 데는 한계가 있을 수 있습니다.
            # 좀 더 발전된 이미지 생성/편집 모델(예: Imagen)이 필요할 수 있으나,
            # 현재 API로 접근 가능한 Gemini Vision을 활용한 예시입니다.
            # 이 프롬프트는 모델에게 '결과 이미지를 묘사'하도록 유도하여 텍스트 응답을 받습니다.
            # 실제 이미지 생성을 위해서는 별도의 이미지 생성 모델 API 호출이 필요합니다.

            # --- (수정) 이미지 생성을 위한 가상 결과 ---
            # 현재 gemini-pro-vision은 텍스트 응답만 반환하므로,
            # 실제 이미지 생성을 위해서는 다른 모델(예: DALL-E, Midjourney API 등)을 연동해야 합니다.
            # 여기서는 개념을 보여주기 위해 업로드된 이미지를 그대로 보여주는 것으로 대체합니다.
            st.success("결과 이미지가 생성되었습니다!")
            st.write("**(주의) 현재 이 예제는 실제 이미지 생성이 아닌, 개념 증명을 위한 것입니다.**")
            st.write("실제 서비스에서는 Gemini와 같은 멀티모달 모델로 이미지를 이해하고, 별도의 이미지 생성 모델을 호출하여 아래와 같은 결과물을 만들어냅니다.")

            # 결과물을 보여주는 섹션
            result_col1, result_col2 = st.columns(2)
            with result_col1:
                st.subheader("결과 (예시)")
                # 실제로는 여기서 AI가 생성한 이미지를 보여줘야 합니다.
                st.image(person_image, caption="가상 피팅 결과 예시 이미지", use_column_width=True)
            with result_col2:
                 st.subheader("AI의 설명 (Gemini 응답)")
                 # Gemini에게 두 이미지를 주고, 어떻게 보일지 묘사해달라고 요청
                 description_prompt = f"""
                 첫 번째 사진 속 인물이 두 번째 사진의 옷을 입는다면 어떤 모습일지 상세하게 묘사해주세요.
                 """
                 response_text = get_gemini_response(description_prompt, person_image_file, clothing_image_file)
                 st.write(response_text)

    else:
        st.warning("사람과 옷 이미지를 모두 업로드해주세요.")