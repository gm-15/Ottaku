import streamlit as st

st.title("나의 맞춤 정보")

# 탭 구성
tab1, tab2 = st.tabs(["체형/피부/취향정보", "사이즈 추천"])

with tab1:
    st.subheader("체형")
    st.write("나의 비슷한 체형의 후기를 모아볼 수 있어요.")

    # 키, 몸무게 입력
    키 = st.number_input("키", min_value=100, max_value=250, value=170, step=1)
    몸무게 = st.number_input("몸무게", min_value=30, max_value=200, value=62, step=1)

    st.subheader("피부")
    st.write("나와 비슷한 피부 타입 후기를 모아볼 수 있어요.")

    # 피부 타입 선택
    피부_타입 = st.radio("피부 타입", ["지성", "건성", "복합성", "민감성"])

    # 피부 톤 선택
    피부_톤 = st.radio("피부 톤", ["봄 웜톤", "여름 쿨톤", "가을 웜톤", "겨울 쿨톤"])

    # 피부 고민 선택 (최대 3개)
    피부_고민 = st.multiselect(
        "피부 고민(최대 3개)",
        ["진정보습", "모공", "여드름", "유수분조절", "미백", "탄력", "영양공급"]
    )

    st.subheader("패션")
    st.write("선호하는 패션 스타일을 다양하게 제안해 드릴게요.")

    # 저장 버튼
    if st.button("저장하기"):
        st.success("저장되었습니다!")

with tab2:
    st.write("사이즈 추천 탭")
