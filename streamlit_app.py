import streamlit as st

st.set_page_config(page_title="CAPYBARA STOCK", layout="wide")

BROWN = "#8B4513"


def render_title(text: str) -> None:
    st.markdown(
        f"<h1 style='color:{BROWN}; margin-bottom: 0.25rem;'>{text}</h1>",
        unsafe_allow_html=True,
    )


def render_home() -> None:
    render_title("CAPYBARA STOCK")
    st.write(
        "Let's start building! For help and inspiration, head over to "
        "[docs.streamlit.io](https://docs.streamlit.io/)."
    )


def render_market() -> None:
    render_title("시장")
    st.info("시장 화면은 준비 중입니다.")


def render_backtest() -> None:
    render_title("백테스트")
    st.info("백테스트 화면은 준비 중입니다.")


with st.sidebar:
    st.subheader("메인 메뉴")
    page = st.radio(
        "이동",
        ["홈", "시장", "백테스트"],
        index=0,
        label_visibility="collapsed",
    )

if page == "홈":
    render_home()
elif page == "시장":
    render_market()
else:
    render_backtest()
