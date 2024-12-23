import streamlit as st
import pandas as pd
import utils as u
from datetime import datetime

st.title("SDG Indicators - Data Exploration")
# Load SDG descriptions
sdg_descriptions = u.load_sdg_mapping("sdg_index_description.csv")[2]
# Load the mapping between SDG column names and indicator names
sdg_column_to_name = u.load_sdg_mapping("sdg_index_description.csv")[0]
sdg_name_to_column = u.load_sdg_mapping("sdg_index_description.csv")[1]


if "data" in st.session_state:
    df = st.session_state["data"]

    sdg_columns = [col for col in df.columns if col.startswith("sdg")]
    sdg_names = [sdg_column_to_name[col] for col in sdg_columns if col in sdg_column_to_name]

    # Sidebar Filters
    st.header("Filters")
    col01, col02 = st.columns(2)
    with col01:
    # Country Filter
        if "Country" in df.columns:
            countries = ["All"] + sorted(df["Country"].dropna().unique())
            selected_country = st.selectbox("Select Country", options=countries, index=0)
            if selected_country != "All":
                df = df[df["Country"] == selected_country]
    with col02:
    # IndexReg Filter
        if "indexreg" in df.columns:
            indexreg_options = ["All"] + sorted(df["indexreg"].dropna().unique())
            selected_indexreg = st.selectbox("Select Region", options=indexreg_options, index=0)
            if selected_indexreg != "All":
                df = df[df["indexreg"] == selected_indexreg]

    # SDG Column Selector
    selected_sdg = st.multiselect("Select SDG Columns", sdg_names, default=sdg_names[:5])
    selected_sdg_columns = [sdg_name_to_column[name] for name in selected_sdg if name in sdg_name_to_column]

    if selected_sdg:

        st.subheader("Correlation Heatmap")
        correlation_matrix = u.calculate_correlation_matrix(df, selected_sdg_columns)
        sdg_name_mapping = {col: sdg_column_to_name[col] for col in selected_sdg_columns if col in sdg_column_to_name}
        u.display_heatmap(correlation_matrix)

        st.subheader("SDG Indicators Over Time")
        if "year" in df.columns:
            u.display_line_graph(df, selected_sdg_columns)
        else:
            st.warning("Your dataset must include a 'year' column for line graphs.")

        st.subheader("Relationship Between Two SDG Indicators")
        col1, col2 = st.columns(2)
        with col1:
            x_name = st.selectbox("Select X-axis SDG Indicator", sdg_names)
            #selected_sdg = st.multiselect("Select SDG Columns", sdg_names, default=sdg_names[:5])
            x_column = sdg_name_to_column[x_name]

        with col2:
            y_name = st.selectbox("Select Y-axis SDG Indicator", [col for col in sdg_names if col != x_name])
            y_column = sdg_name_to_column[y_name]


        trendline_type = st.selectbox("Select Trendline Type", ["linear", "polynomial", "logarithmic", "exponential"])
        if x_column and y_column:
            u.display_relationship_chart(df, x_column, y_column, trendline_type)

            # Assuming you have variables sdg_indicator_1 and sdg_indicator_2 defined
            trend_options = ['linear', 'polynomial', 'logarithmic', 'exponential', 'none']
            selected_trend = st.selectbox(' Select the most suitable trend:', trend_options)
            # Save the record on button click
            if st.button('Save Trend Selection'):
                u.save_relationship_record(x_column, y_column, selected_trend, file_path='relationships.csv')
                st.success('Trend selection saved successfully.')

    else:
        st.warning("Please select at least one SDG column to explore.")
else:
    st.info("Please read About the Project and let's get started.")
