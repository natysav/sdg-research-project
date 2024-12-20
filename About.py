import streamlit as st
import utils as u

st.title("About Sustainable Development Goals (SDGs)")

st.sidebar.header("SDG Data Upload")
# Check if 'uploaded_file' exists in session_state
if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your SDG Data CSV file", type=["csv"])

if uploaded_file is not None:
    st.session_state['uploaded_file'] = uploaded_file  # Save uploaded file to session_state
    st.sidebar.success("File uploaded successfully! You can now navigate to other pages.")

st.markdown("""
### What are the SDGs?
The Sustainable Development Goals (SDGs), adopted by the United Nations in 2015, are a collection of 17 interlinked global goals 
designed to achieve a better and more sustainable future for all. Each goal is associated with specific indicators and targets.

### Useful Resources:
- [United Nations SDGs Overview](https://sdgs.un.org/goals)
- [Australia's SDG Progress](https://www.sdgdata.gov.au)
- [SDG Indicators](https://unstats.un.org/sdgs/indicators/database/)
""")
