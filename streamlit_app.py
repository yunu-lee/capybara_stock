import streamlit as st
st.set_page_config(page_title="CAPYBARA STOCK", layout="wide")

st.title("CAPYBARA STOCK")

with st.sidebar:
    st.header("메인 메뉴")
    selected_menu = st.radio(
        "이동",
        options=["시장", "백테스트"],
        index=0,
    )

if selected_menu == "시장":
    st.subheader("시장")
    st.info("준비 중입니다.")
elif selected_menu == "백테스트":
    st.subheader("백테스트")
    st.info("준비 중입니다.")
