import streamlit as st

st.set_page_config(layout="wide")

st.title("🌐 외부 웹사이트 삽입 예제")
st.write("Streamlit 앱 안에 다른 웹사이트를 표시할 수 있습니다.")
st.write("---")


# 보여주고 싶은 웹사이트 주소
site_url = "https://huggingface.co/spaces/Kwai-Kolors/Kolors-Virtual-Try-On"

st.subheader(f"삽입된 사이트: {site_url}")

# st.components.v1.iframe을 사용하여 웹사이트를 화면에 표시
# height 값을 조절하여 표시되는 창의 높이를 변경할 수 있습니다.
st.components.v1.iframe(site_url, height=600, scrolling=True)


st.write("---")
st.info("위 화면은 Streamlit 공식 블로그를 그대로 가져와 보여주는 것입니다.")