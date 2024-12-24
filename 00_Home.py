import streamlit as st
import utils as u
import pandas as pd

# App configuration
st.set_page_config(
    page_title="SDG Research Project",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.sidebar.header("SDG Data Management")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your SDG Data CSV file (if you want to explore your data instead)", type=["csv"])
if uploaded_file:
    st.session_state["data"] = u.load_file(uploaded_file)
    st.sidebar.success("File uploaded successfully! You can now navigate to other pages.")
else:
    # Load a preloaded file if none is uploaded
    if "data" not in st.session_state:
        st.session_state["data"] = pd.read_csv("sdg_all.csv", encoding="latin1", delimiter=",", skip_blank_lines=True, on_bad_lines="skip")


# Home Page Content
st.title("Sustainable Development Goals Simulator 🌍")
st.write("""
Welcome to the SDG Research Project. This tool demonstrates how individual and organizational decisions impact global sustainability goals.

Navigate through the sections using the sidebar to explore data trends, run simulations, and learn more about the UN SDGs. Collaborators and investors are welcome to join this impactful journey!
""")

st.image("istock.jpg", use_container_width=True)  # Replace with your image path

st.write("### What You Can Do:")
st.markdown("""
- 📊 **Explore Data**: View trends in SDG indicators across regions and time periods.
- 🛠️ **Simulation**: Adjust SDG indicators to observe their potential ripple effects.
- 📚 **Learn More**: Understand the importance and intricacies of SDG indicators.
- 📬 **Contact Us**: Get in touch to collaborate or invest in this project.
""")
