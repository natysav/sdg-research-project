import streamlit as st
import utils as u
import pandas as pd

st.title("Learn More 📚")
st.write("""
The Sustainable Development Goals (SDGs) are a universal call to action to end poverty, protect the planet, and ensure peace and prosperity for all. 
Explore [UN SDG Resources](https://sdgs.un.org/goals) for more information.

This tool provides insights into how individual decisions can influence these goals. 
Check the "Explore Data" section to view trends and "Simulation" to experiment with potential impacts.
""")

# st.markdown("""
# ### What are the SDGs?
# The Sustainable Development Goals (SDGs), adopted by the United Nations in 2015, are a collection of 17 interlinked global goals
# designed to achieve a better and more sustainable future for all. Each goal is associated with specific indicators and targets.
#
# ### Useful Resources:
# - [United Nations SDGs Overview](https://sdgs.un.org/goals)
# - [Australia's SDG Progress](https://www.sdgdata.gov.au)
# - [SDG Indicators](https://unstats.un.org/sdgs/indicators/database/)
#
# ### This App
# - The first purpose of this App is to Explore the data and
# find trends between Sustainable Development Indicators.
# For example, is there any relationships between "Unemployment Rate" and "Literacy Rate"?
# Select these indicators on Exploration page of the App and find out.
# - The second purpose of the App is to Simulate
# how changes in one indicator or a group of indicators would influence other indicators.
# For example, if you change literacy rate and gini coefficient what the change is in homocides if any.
# Go to Simulation page and see.
# This App is developed based on real world data. The modeling is still under development and any help is highly appreciated.
# """)

# Display Indicator Information
sdg_descriptions = u.load_sdg_mapping("sdg_index_description.csv")[2]
if sdg_descriptions is not None:
    st.subheader("SDG Indicator Information")
    st.markdown("""
    The table below has the indicators determined by UN:""")
    st.dataframe(sdg_descriptions[["IndCode", "Indicator", "Optimum (= 100)", "Green threshold","Red threshold", "Lower Bound (=0)"]])

#
#
# st.subheader("What is next?")
#
# st.markdown("""
# ### Potential applications of the project
# Help to assess personal and business decisions in terms of sustainability
# and long term value for society, environment and global economy.
# It can be an extension to an investment platform.
# """)