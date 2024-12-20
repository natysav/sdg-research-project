import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import numpy as np
from scipy.optimize import curve_fit
import statsmodels.api as sm

def load_sdg_mapping(file_path):
    """
    Load SDG mapping from column names to indicator names and vice versa.
    """
    sdg_description = pd.read_csv("sdg_index_description.csv", encoding="latin1", delimiter=",", skip_blank_lines=True, on_bad_lines="skip")
    column_to_name = dict(zip(sdg_description['IndCode'], sdg_description['Indicator']))
    name_to_column = dict(zip(sdg_description['Indicator'], sdg_description['IndCode']))
    return column_to_name, name_to_column, sdg_description

@st.cache_data
def load_file(uploaded_file):
    """Load and cache the uploaded file."""
    if uploaded_file is not None:
        try:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(uploaded_file, encoding="latin1", delimiter=",", skip_blank_lines=True, on_bad_lines="skip")
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        return None

# Function to calculate correlation matrix
def calculate_correlation_matrix(df, sdg_columns):
    numeric_df = df[sdg_columns].apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    return numeric_df.corr()

# Function to display heatmap
def display_heatmap(correlation_matrix, tooltips=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
        xticklabels = [tooltips.get(col, col) for col in
                   correlation_matrix.columns] if tooltips else correlation_matrix.columns,
        yticklabels = [tooltips.get(row, row) for row in
                   correlation_matrix.index] if tooltips else correlation_matrix.index,
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    st.pyplot(fig)

# Function to display line graph for SDG indicators over time
def display_line_graph(df, selected_sdg_columns):
    """
    Displays a line graph showing the average value of each SDG indicator by year.

    Args:
        df (pd.DataFrame): DataFrame containing SDG data.
        selected_sdg_columns (list): List of SDG columns to display on the line graph.
    """
    if 'year' not in df.columns:
        st.error("The dataset does not contain a 'year' column.")
        return

    # Filter the relevant columns
    columns_to_plot = ['year'] + selected_sdg_columns
    filtered_df = df[columns_to_plot]

    # Group by year and calculate the mean
    grouped_df = filtered_df.groupby('year', as_index=False).mean()

    # Melt the DataFrame for easier plotting with Plotly Express
    melted_df = grouped_df.melt(id_vars='year', value_vars=selected_sdg_columns, var_name='SDG Indicator',
                                value_name='Average Value')

    # Plot the line graph
    fig = px.line(
        melted_df,
        x='year',
        y='Average Value',
        color='SDG Indicator',
        title="Average SDG Indicators Over Time",
        labels={'Average Value': 'Average Value', 'year': 'Year'}
    )
    st.plotly_chart(fig)

# Function to display relationship chart
def nonlinear_fit(x, y, model_type):
    """
    Fit a nonlinear or linear model to the data.

    Args:
        x (array-like): The x data.
        y (array-like): The y data.
        model_type (str): The type of model ('linear', 'polynomial', 'logarithmic', 'exponential').

    Returns:
        fitted_x (array-like): The sorted x values for plotting.
        fitted_y (array-like): The y values predicted by the fitted model.
    """
    x = np.array(x)
    y = np.array(y)

    # Remove NaN and infinite values
    mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isinf(x) & ~np.isinf(y)
    x = x[mask]
    y = y[mask]

    # Sort the data by x for a continuous line
    sorted_indices = np.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]

    if model_type == "linear":
        # Linear regression (1st-degree polynomial)
        coeffs = np.polyfit(x, y, 1)
        linear_model = np.poly1d(coeffs)
        return x, linear_model(x)

    elif model_type == "polynomial":
        # Polynomial regression (2nd-degree polynomial)
        coeffs = np.polyfit(x, y, 2)
        poly_model = np.poly1d(coeffs)
        return x, poly_model(x)

    elif model_type == "logarithmic":
        # Logarithmic regression
        def log_model(x, a, b):
            return a * np.log(x) + b

        # Filter out x <= 0 (logarithm undefined for non-positive values)
        log_mask = x > 0
        x_log = x[log_mask]
        y_log = y[log_mask]

        if len(x_log) < 2:  # Require at least two points for curve fitting
            return x, np.full_like(x, np.nan)

        try:
            popt, _ = curve_fit(log_model, x_log, y_log)
            return x, log_model(x, *popt)
        except RuntimeError:
            return x, np.full_like(x, np.nan)  # Return NaNs if fitting fails

    elif model_type == "exponential":
        # Exponential regression
        def exp_model(x, a, b):
            return a * np.exp(b * x)

        if len(x) < 2:  # Require at least two points for curve fitting
            return x, np.full_like(x, np.nan)

        try:
            popt, _ = curve_fit(exp_model, x, y, maxfev=10000)
            return x, exp_model(x, *popt)
        except RuntimeError:
            return x, np.full_like(x, np.nan)  # Return NaNs if fitting fails

    else:
        return x, y  # No fitting applied


# Add this function to calculate the sum of squared residuals
def calculate_least_squares(x, y, trendline_type):
    x = np.array(x)
    y = np.array(y)

    if trendline_type == "linear":
        model = np.polyfit(x, y, 1)
        predicted_y = np.polyval(model, x)
    elif trendline_type == "polynomial":
        model = np.polyfit(x, y, 2)
        predicted_y = np.polyval(model, x)
    elif trendline_type == "logarithmic":
        log_x = np.log(x + 1e-10)  # Avoid log(0)
        model = np.polyfit(log_x, y, 1)
        predicted_y = np.polyval(model, log_x)
    elif trendline_type == "exponential":
        log_y = np.log(y + 1e-10)  # Avoid log(0)
        model = np.polyfit(x, log_y, 1)
        predicted_y = np.exp(np.polyval(model, x))
    else:
        return None

    residuals = y - predicted_y
    least_squares = np.sum(residuals**2)
    return least_squares

# Modified display_relationship_chart function
def display_relationship_chart(df, x_column, y_column, trendline_type):
    # Drop rows with NaN or infinite values in the selected columns
    df = df[[x_column, y_column]].dropna()
    df = df[np.isfinite(df[x_column]) & np.isfinite(df[y_column])]

    # Create scatter plot with trendline
    scatter_fig = px.scatter(
        df,
        x=x_column,
        y=y_column,
        trendline="ols" if trendline_type == "linear" else None,
        labels={x_column: x_column, y_column: y_column},
        title=f"Relationship Between {x_column} and {y_column}",
    )

    # Add the calculated trendline manually for other types
    if trendline_type != "linear":
        x = df[x_column].values
        y = df[y_column].values

        if trendline_type == "polynomial":
            z = np.polyfit(x, y, 2)
            p = np.poly1d(z)
            scatter_fig.add_scatter(x=np.sort(x), y=p(np.sort(x)), mode="lines", name="Polynomial Fit")
        elif trendline_type == "logarithmic":
            z = np.polyfit(np.log(x + 1e-10), y, 1)
            p = np.poly1d(z)
            scatter_fig.add_scatter(x=np.sort(x), y=p(np.log(np.sort(x) + 1e-10)), mode="lines", name="Logarithmic Fit")
        elif trendline_type == "exponential":
            z = np.polyfit(x, np.log(y + 1e-10), 1)
            p = lambda x: np.exp(z[0] * x + z[1])
            scatter_fig.add_scatter(x=np.sort(x), y=p(np.sort(x)), mode="lines", name="Exponential Fit")

    # Calculate least squares for all trendline types
    least_squares_values = {
        "Linear": calculate_least_squares(df[x_column], df[y_column], "linear"),
        "Polynomial": calculate_least_squares(df[x_column], df[y_column], "polynomial"),
        "Logarithmic": calculate_least_squares(df[x_column], df[y_column], "logarithmic"),
        "Exponential": calculate_least_squares(df[x_column], df[y_column], "exponential"),
    }

    # Display the scatter plot
    st.plotly_chart(scatter_fig, use_container_width=True)

    # Display the least squares values
    st.write("### Least Squares Values for Trendlines")
    least_squares_df = pd.DataFrame(
        {
            "Trendline Type": list(least_squares_values.keys()),
            "Least Squares Value": list(least_squares_values.values()),
        }
    )
    # Remove the index and adjust the height
    st.dataframe(least_squares_df.reset_index(drop=True), height=200)  # Adjust height as needed