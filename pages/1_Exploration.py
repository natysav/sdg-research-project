import streamlit as st
import pandas as pd
import utils as u

st.title("SDG Indicators - Data Exploration")
# Load SDG descriptions
sdg_descriptions = pd.read_csv("sdg_index_description.csv", encoding="latin1", delimiter=",", skip_blank_lines=True, on_bad_lines="skip")

if "data" in st.session_state:
    df = st.session_state["data"]
    sdg_columns = [col for col in df.columns if col.startswith("sdg")]

    # Sidebar Filters
    st.sidebar.header("Filters")

    # Country Filter
    if "Country" in df.columns:
        countries = ["All"] + sorted(df["Country"].dropna().unique())
        selected_country = st.sidebar.selectbox("Select Country", options=countries, index=0)
        if selected_country != "All":
            df = df[df["Country"] == selected_country]

    # IndexReg Filter
    if "indexreg" in df.columns:
        indexreg_options = ["All"] + sorted(df["indexreg"].dropna().unique())
        selected_indexreg = st.sidebar.selectbox("Select IndexReg", options=indexreg_options, index=0)
        if selected_indexreg != "All":
            df = df[df["indexreg"] == selected_indexreg]

    # SDG Column Selector
    selected_sdg = st.sidebar.multiselect("Select SDG Columns", sdg_columns, default=sdg_columns[:5])

    if selected_sdg:
        # Display Indicator Information
        if sdg_descriptions is not None:
            st.subheader("SDG Indicator Information")
            selected_info = sdg_descriptions[sdg_descriptions['IndCode'].isin(selected_sdg)]
            st.table(selected_info[["IndCode", "Indicator", "Optimum (= 100)", "Green threshold","Red threshold", "Lower Bound (=0)"]])


        st.subheader("Correlation Heatmap")
        correlation_matrix = u.calculate_correlation_matrix(df, selected_sdg)
        u.display_heatmap(correlation_matrix)

        st.subheader("SDG Indicators Over Time")
        if "year" in df.columns:
            u.display_line_graph(df, selected_sdg)
        else:
            st.warning("Your dataset must include a 'year' column for line graphs.")

        st.subheader("Relationship Between Two SDG Indicators")
        col1, col2 = st.columns(2)
        with col1:
            x_column = st.selectbox("Select X-axis SDG Indicator", sdg_columns)
        with col2:
            y_column = st.selectbox("Select Y-axis SDG Indicator", [col for col in sdg_columns if col != x_column])

        trendline_type = st.selectbox("Select Trendline Type", ["linear", "polynomial", "logarithmic", "exponential"])
        if x_column and y_column:
            u.display_relationship_chart(df, x_column, y_column, trendline_type)
    else:
        st.warning("Please select at least one SDG column to explore.")
else:
    st.info("Please upload a file from the sidebar to get started.")
