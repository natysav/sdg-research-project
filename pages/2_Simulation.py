import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import utils as u

import streamlit as st

st.title("Run Simulation 🛠️")
st.write("""
This tool uses modeling and trend analysis to provide insights.
""")

# Load the mapping between SDG column names and indicator names
sdg_column_to_name = u.load_sdg_mapping("sdg_index_description.csv")[0]
sdg_name_to_column = u.load_sdg_mapping("sdg_index_description.csv")[1]


if "data" in st.session_state:
    df = st.session_state["data"]

    # Identify SDG columns and map them to indicator names
    sdg_columns = [col for col in df.columns if col.startswith("sdg")]
    sdg_names = [sdg_column_to_name[col] for col in sdg_columns if col in sdg_column_to_name]

    if not sdg_columns:
        st.error("No SDG columns found in the dataset.")
        st.stop()

    # Variable and Dependent Columns
    st.subheader("Select Variables and Dependents")

    variable_names = st.multiselect("Select Variable Indicators (Influencers)", sdg_names)

    dependent_names = st.multiselect("Select Dependent Indicators (Effects)", sdg_names)

    if not variable_names or not dependent_names:
        st.warning("Please select at least one variable indicator and one dependent indicator.")
        st.stop()

    # Map back to column names
    variable_columns = [sdg_name_to_column[name] for name in variable_names if name in sdg_name_to_column]
    dependent_columns = [sdg_name_to_column[name] for name in dependent_names if name in sdg_name_to_column]

    # Filter data for the selected columns
    selected_columns = variable_columns + dependent_columns
    simulation_data = df[selected_columns].dropna()

    # Train a regression model for each dependent variable
    models = {}
    for dependent in dependent_columns:
        X = simulation_data[variable_columns]
        y = simulation_data[dependent]
        model = LinearRegression()
        model.fit(X, y)
        models[dependent] = model

    # Sliders for Variable Columns
    st.subheader("Set Values for Variable Indicators")
    st.write("""
    Adjust SDG indicators below to observe their potential impact on other metrics.
    """)
    slider_values = {}
    for col, name in zip(variable_columns, variable_names):
        min_val = float(simulation_data[col].min())
        max_val = float(simulation_data[col].max())
        default_val = float(simulation_data[col].mean())
        slider_values[col] = st.slider(f"Set value for {name}", min_val, max_val, default_val)

    # Simulate Dependent Values
    st.subheader("Simulation Results for Dependent Indicators")
    simulated_values = {}
    for dependent, model in models.items():
        input_data = np.array([slider_values[col] for col in variable_columns]).reshape(1, -1)
        simulated_value = model.predict(input_data)[0]
        simulated_values[dependent] = simulated_value

    # Display Simulated Values
    simulated_df = pd.DataFrame.from_dict(
        {sdg_column_to_name[dependent]: value for dependent, value in simulated_values.items()},
        orient="index",
        columns=["Simulated Value"]
    )
    st.table(simulated_df)

else:
    st.info("Please read About the project and let's get started.")
