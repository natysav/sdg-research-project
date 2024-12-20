import streamlit as st
import utils as u
import pandas as pd

st.title("About Sustainable Development Goals (SDGs)")

st.sidebar.header("SDG Data Management")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your SDG Data CSV file", type=["csv"])
if uploaded_file:
    st.session_state["data"] = u.load_file(uploaded_file)
    st.sidebar.success("File uploaded successfully! You can now navigate to other pages.")
else:
    # Load a preloaded file if none is uploaded
    if "data" not in st.session_state:
        st.session_state["data"] = pd.read_csv("sdg_all.csv", encoding="latin1", delimiter=",", skip_blank_lines=True, on_bad_lines="skip")

st.markdown("""
### What are the SDGs?
The Sustainable Development Goals (SDGs), adopted by the United Nations in 2015, are a collection of 17 interlinked global goals 
designed to achieve a better and more sustainable future for all. Each goal is associated with specific indicators and targets.

### Useful Resources:
- [United Nations SDGs Overview](https://sdgs.un.org/goals)
- [Australia's SDG Progress](https://www.sdgdata.gov.au)
- [SDG Indicators](https://unstats.un.org/sdgs/indicators/database/)
""")
