import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import utils as u

st.title("SDG Indicators - Simulation")

if "data" in st.session_state:
    df = st.session_state["data"]
    sdg_columns = [col for col in df.columns if col.startswith("sdg")]

    st.subheader("Select Variables and Dependents")
    variable_columns = st.multiselect("Select Variable Columns (Independent Variables)", sdg_columns)
    dependent_columns = st.multiselect("Select Dependent Columns", sdg_columns)

    if variable_columns and dependent_columns:
        selected_columns = variable_columns + dependent_columns
        simulation_data = df[selected_columns].dropna()

        models = {}
        for dependent in dependent_columns:
            X = simulation_data[variable_columns]
            y = simulation_data[dependent]
            model = LinearRegression()
            model.fit(X, y)
            models[dependent] = model

        st.subheader("Set Values for Variable Columns")
        slider_values = {}
        for col in variable_columns:
            min_val = float(simulation_data[col].min())
            max_val = float(simulation_data[col].max())
            default_val = float(simulation_data[col].mean())
            slider_values[col] = st.slider(f"Set value for {col}", min_val, max_val, default_val)

        st.subheader("Simulated Values for Dependent Columns")
        simulated_values = {}
        for dependent, model in models.items():
            input_data = np.array([slider_values[col] for col in variable_columns]).reshape(1, -1)
            simulated_value = model.predict(input_data)[0]
            simulated_values[dependent] = simulated_value

        simulated_df = pd.DataFrame.from_dict(simulated_values, orient="index", columns=["Simulated Value"])
        st.table(simulated_df)
else:
    st.info("Please upload a file from the sidebar to get started.")
