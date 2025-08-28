# 👕 옷타쿠 (Ottaku) - AI Personal Stylist


**"옷을 사랑하는 당신을 위한 AI 기반 퍼스널 스타일리스트"**

'옷타쿠'는 '옷'과 '오타쿠'의 합성어로, 옷을 진심으로 사랑하지만 매일 아침 "뭐 입지?"를 고민하는 사람들을 위해 탄생했습니다. Gemini와 OpenAI의 강력한 AI 모델을 활용하여, 사용자의 개인화된 정보와 옷장 속 아이템을 기반으로 최적의 코디를 추천하고 시각화해주는 올인원 패션 솔루션입니다.

---

## 📌 프로젝트 개요

**Ottaku**는 옷을 좋아하지만 선택 과정에서 어려움을 겪는 소비자들을 위해 개발된 **AI 기반 맞춤형 코디 추천 서비스**입니다.

현대 소비자들은 다음과 같은 고민을 자주 합니다:
-   나에게 어울리는 옷은 무엇일까?
-   이 옷이 최신 트렌드에 맞을까?
-   기존 옷들과 조합해 입을 수 있을까?

이 과정에서 많은 시간과 에너지를 쓰거나, 충동구매로 인해 **옷을 제대로 활용하지 못하는 문제**가 발생합니다. **Ottaku**는 이러한 문제를 해결하여 단순한 쇼핑을 넘어 **스마트한 패션 경험**을 제공하는 것을 목표로 합니다.

---

## 👥 팀 구성 및 개발 정보

| 이름 | 역할 |
| :--- | :--- |
| GM-15 | 기획, 개발 |
| KHYOJAE | 기획, 개발 |

-   **개발 기간**: 2025.08.26 ~ 2025.08.29
-   **협업 도구**: GitHub, Notion
-   **개발 도구**: IntelliJ, PyCharm
-   **언어**: Python

---

## 🏗️ 프로젝트 아키텍처

![프로젝트 아키텍처](https://github.com/user-attachments/assets/d0db5fe1-7500-48e3-bbbb-30ac089f6e07)

---

## ✨ 핵심 기능 소개

'옷타쿠'는 사용자의 패션 라이프사이클 전체를 돕는 다양한 AI 기능을 제공합니다.

### 🏠 메인 페이지
> 사용자의 모든 패션 활동이 이루어지는 핵심 공간입니다.

-   **⚙️ 나의 맞춤 정보**: 성별, 체형, 퍼스널 컬러, 선호 스타일을 입력하여 AI 추천의 기반을 마련합니다.
-   **📏 AI 사이즈 추천**: 입력된 체형 정보를 바탕으로 일반적인 의류 사이즈를 즉시 추천합니다.
-   **🧠 옷 분석하기**: 옷 사진을 올리면 AI가 종류, 색상, 스타일 태그 등을 자동으로 분석합니다.
-   **✨ AI 코디 추천**: 분석된 옷과 사용자 정보를 종합하여, 특정 상황에 맞는 2가지 코디를 텍스트와 DALL-E 생성 이미지로 시각화하여 제안합니다. (음성 안내 포함)

### 👚 나의 옷장
> 사용자의 패션 자산을 관리하는 디지털 옷장입니다.

-   **내가 분석한 옷**: 분석하고 "옷장에 추가"한 옷들을 이미지와 함께 모아보고 관리할 수 있습니다.
-   **저장된 추천 코디**: AI가 생성해준 코디 중 마음에 드는 이미지를 저장하고 모아볼 수 있습니다.

### 🎨 퍼스널 컬러 분석
> 과학적인 자기 분석을 돕는 AI 도구입니다.

-   **AI 퍼스널 컬러 진단**: 얼굴 사진만으로 AI가 퍼스널 컬러와 그 이유를 상세히 설명해주고, 결과를 프로필에 바로 적용할 수 있습니다.

### 📊 패션 데이터 분석
> 데이터 기반의 객관적인 패션 인사이트를 제공합니다.

-   **나의 패션 프로필**: 사용자가 분석한 옷들의 누적 데이터를 기반으로, 선호 스타일과 색상 분포를 실시간으로 시각화합니다.
-   **최신 패션 트렌드**: Google Trends API와 연동하여, 주요 패션 키워드의 최신 관심도 변화를 실시간 그래프로 제공합니다.

### 🌤️ 오늘의 날씨 및 옷차림 추천 (사이드바)
> 매일의 실용적인 정보를 제공하는 위젯입니다.

-   **실시간 날씨 정보**: 기상청 API를 통해 오늘, 내일, 모레의 날씨 예보를 조회합니다.
-   **AI 옷차림 추천**: 조회된 기온에 맞춰 AI가 직관적인 옷차림 가이드를 제공합니다.

---

## 📸 기능별 화면

| 나의 맞춤 정보 | 사이즈 추천 |
| :---: | :---: |
| ![맞춤 정보 입력](https://github.com/user-attachments/assets/b1d6faa5-a99a-4577-a26c-504e8178b65b) | ![사이즈 추천](https://github.com/user-attachments/assets/1eb7f502-2008-4f09-ad1b-85a25500b225) |
| **옷 분석하기** | **코디 추천받기** |
| ![옷 분석](https://github.com/user-attachments/assets/5d454b0d-4415-46c3-9d0b-e4940272e6f3) | ![코디 추천](https://github.com/user-attachments/assets/2d9a236b-a3eb-49bc-b82d-aec6396bf076) |
| **나의 옷장** | **퍼스널 컬러 분석** |
| ![나의 옷장](https://github.com/user-attachments/assets/51682286-310e-4301-959e-abf68e3bdb53) | ![퍼스널 컬러 분석](https://github.com/user-attachments/assets/05a3e9fe-ce64-41ad-9fe0-a63f3369e4bd) |
| **패션 데이터 분석** | **오늘의 날씨** |
| ![패션 데이터 분석](https://github.com/user-attachments/assets/7a28d272-01d8-474b-ad02-020abdaf6471) | *(사이드바 기능)* |


---

## 🎬 시연 영상 및 링크

**Live Demo GIF**
![Ottaku](https://github.com/user-attachments/assets/decb936e-f2b8-4de1-b60d-ec745c1d5242)

-   **웹사이트 바로가기**: <https://ottaku-sporf6mdmg8tszefzzdfgn.streamlit.app/>
-   **유튜브 시연 영상**: <https://youtu.be/jqHrp3Tf1PM>

---

## 🛠️ 기술 스택 (Tech Stack)

-   **Frontend**: `Streamlit`
-   **Core AI Engine**: `Google Gemini 1.5 Flash`, `OpenAI DALL-E 3`, `OpenAI TTS`
-   **Data API**: `Pytrends` (Google Trends API), `기상청 단기예보 API`
-   **Data Visualization**: `Plotly`, `Pandas`
-   **Libraries**: `requests`, `pytz`

---

## 🚀 실행 방법 (How to Run)

1.  **GitHub 저장소 복제:**
    ```bash
    git clone [저장소 URL]
    cd [프로젝트 폴더]
    ```
2.  **필요한 라이브러리 설치:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **API 키 설정:**
    -   프로젝트 루트에 `.streamlit` 폴더를 생성합니다.
    -   해당 폴더 안에 `secrets.toml` 파일을 만들고 아래 내용을 채워넣습니다.
        ```toml
        KMA_API_KEY = "YOUR_KOREA_METEOROLOGICAL_ADMINISTRATION_API_KEY"
        GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
        OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
        ```
4.  **Streamlit 앱 실행:**
    ```bash
    streamlit run app.py
    ```

---

## 📈 기대 효과

| 구분 | 기대 효과 | 세부 내용 |
| :--- | :--- | :--- |
| **⏰ 시간 단축** | 코디 고민 및 쇼핑 시간 **90% 이상 절감** | AI가 개인화된 코디를 즉시 제안하여 의사결정 피로감 해소 |
| **💰 자원 절약** | 충동구매 감소 및 **옷장 활용률 극대화** | AI 분석을 통한 쇼핑 실패율 감소 및 기존 의류 활용법 제안 |
| **📉 산업 기여** | 온라인 쇼핑몰 **반품률 감소** | 개인화 추천으로 구매 만족도를 높여 물류 및 운영 비용 절감 |
