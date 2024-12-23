import streamlit as st
import utils as u
import pandas as pd

st.title("About Sustainable Development Goals (SDGs)")

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

st.markdown("""
### What are the SDGs?
The Sustainable Development Goals (SDGs), adopted by the United Nations in 2015, are a collection of 17 interlinked global goals 
designed to achieve a better and more sustainable future for all. Each goal is associated with specific indicators and targets.

### Useful Resources:
- [United Nations SDGs Overview](https://sdgs.un.org/goals)
- [Australia's SDG Progress](https://www.sdgdata.gov.au)
- [SDG Indicators](https://unstats.un.org/sdgs/indicators/database/)

### This App
- The first purpose of this App is to Explore the data and 
find trends between Sustainable Development Indicators. 
For example, is there any relationships between "Unemployment Rate" and "Literacy Rate"?
Select these indicators on Exploration page of the App and find out.
- The second purpose of the App is to Simulate 
how changes in one indicator or a group of indicators would influence other indicators.
For example, if you change literacy rate and gini coefficient what the change is in homocides if any.
Go to Simulation page and see.
This App is developed based on real world data. The modeling is still under development and any help is highly appreciated. 
""")

# Display Indicator Information
sdg_descriptions = u.load_sdg_mapping("sdg_index_description.csv")[2]
if sdg_descriptions is not None:
    st.subheader("SDG Indicator Information")
    st.markdown("""
    The table below has the indicators determined by UN:""")
    st.dataframe(sdg_descriptions[["IndCode", "Indicator", "Optimum (= 100)", "Green threshold","Red threshold", "Lower Bound (=0)"]])

