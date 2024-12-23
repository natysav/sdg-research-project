import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

import statsmodels.api as sm
from datetime import datetime

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
    st.write("## Least Squares Values for Trendlines")
    least_squares_df = pd.DataFrame(
        {
            "Trendline Type": list(least_squares_values.keys()),
            "Least Squares Value": list(least_squares_values.values()),
        }
    )
    # Remove the index and adjust the height
    st.dataframe(least_squares_df.reset_index(drop=True), height=200)  # Adjust height as needed
def save_relationship_record(sdg_indicator_1, sdg_indicator_2, trend, file_path='relationships.csv'):
    """
    Saves a new record of relationships between two variables with the selected trend to a CSV file.

    Parameters:
    - sdg_indicator_1 (str): The first SDG indicator.
    - sdg_indicator_2 (str): The second SDG indicator.
    - trend (str): The trend selected ('linear', 'polynomial', 'logarithmic', 'exponential', or 'none').
    - file_path (str): Path to the CSV file. Default is 'relationships.csv'.

    Returns:
    None
    """
    # Create a DataFrame for the new record
    new_record = pd.DataFrame({
        'sdg_indicator_1': [sdg_indicator_1],
        'sdg_indicator_2': [sdg_indicator_2],
        'trend': [trend],
        'created_timestamp': [datetime.now()]
    })

    try:
        # Try to load the existing CSV file
        relationships_df = pd.read_csv(file_path)
        # Append the new record
        relationships_df = pd.concat([relationships_df, new_record], ignore_index=True)
    except FileNotFoundError:
        # If the file does not exist, create a new DataFrame
        relationships_df = new_record

    # Save the updated DataFrame to the file
    relationships_df.to_csv(file_path, index=False)


def show_relationships_table():
    # Load SDG indicators
    sdg_index_df = pd.read_csv('sdg_index_description.csv', encoding="latin1", delimiter=",", skip_blank_lines=True, on_bad_lines="skip")
    indicators = sdg_index_df['IndCode'].tolist()

    # Create all ordered pairs of SDG indicators (A-B is different from B-A)
    pairs = []
    for i, ind1 in enumerate(indicators):
        for ind2 in indicators:
            if ind1 != ind2:  # Avoid self-pairs (A-A)
                pairs.append((ind1, ind2))

    # Load relationships.csv
    try:
        relationships_df = pd.read_csv('relationships.csv')
        relationships_df['created_timestamp'] = pd.to_datetime(relationships_df['created_timestamp'])
    except FileNotFoundError:
        relationships_df = pd.DataFrame(columns=['sdg_indicator_1', 'sdg_indicator_2', 'trend', 'created_timestamp'])

    # Prepare the table
    relationships_table = []
    for ind1, ind2 in pairs:
        # Filter the records for the pair
        pair_records = relationships_df[
            ((relationships_df['sdg_indicator_1'] == ind1) & (relationships_df['sdg_indicator_2'] == ind2)) |
            ((relationships_df['sdg_indicator_1'] == ind2) & (relationships_df['sdg_indicator_2'] == ind1))
            ]

        # Use the latest trend if records exist, otherwise default to 'linear'
        if not pair_records.empty:
            latest_record = pair_records.sort_values(by='created_timestamp', ascending=False).iloc[0]
            trend = latest_record['trend']
        else:
            trend = 'linear'

        # Append to the table
        relationships_table.append({'sdg_indicator_1': ind1, 'sdg_indicator_2': ind2, 'trend': trend})

    # Convert to DataFrame for display
    relationships_table_df = pd.DataFrame(relationships_table)
    return relationships_table_df


def train_model(independent, dependent, trend, data):
    """
    Train a model based on the specified trend for the given pair of variables.

    Parameters:
    - independent (str): The independent variable column name.
    - dependent (str): The dependent variable column name.
    - trend (str): The trend type ('linear', 'polynomial', 'logarithmic', 'exponential', 'none').
    - data (pd.DataFrame): Dataset with two columns (independent and dependent).

    Returns:
    - model: Trained model or None if trend is 'none'.
    - predictions: Predicted values (if applicable).
    - mse: Mean Squared Error or None if trend is 'none'.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_squared_error
    import numpy as np

    # Extract independent (X) and dependent (y) data
    X = data[[independent]].values
    y = data[dependent].values

    if trend == 'none':
        return None, None, None

    if trend == 'linear':
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)
    elif trend == 'polynomial':
        degree = 2  # Adjust as needed
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        predictions = model.predict(X_poly)
    elif trend == 'logarithmic':
        X_log = np.log(X)
        model = LinearRegression()
        model.fit(X_log, y)
        predictions = model.predict(X_log)
    elif trend == 'exponential':
        y_log = np.log(y)
        model = LinearRegression()
        model.fit(X, y_log)
        predictions = np.exp(model.predict(X))  # Reverse the log transformation
    else:
        return None, None, None

    mse = mean_squared_error(y, predictions)
    return model, predictions, mse

def simulate_sdg_values(simulation_data, relationships, variable_columns, dependent_columns, slider_values):
    """
    Simulates dependent values based on relationships and user-set slider values for independent variables.

    Parameters:
    - simulation_data (pd.DataFrame): Filtered dataset with selected independent and dependent variables.
    - relationships (pd.DataFrame): Relationships table specifying trends between variables.
    - variable_columns (list): Selected independent variable columns.
    - dependent_columns (list): Selected dependent variable columns.
    - slider_values (dict): User-selected values for independent variables.

    Returns:
    - simulated_values (dict): Dictionary of dependent variables and their simulated values.
    """
    simulated_values = {}

    for dependent in dependent_columns:
        # Filter relationships for the current dependent variable
        dependent_relationships = relationships[
            (relationships['sdg_indicator_2'] == dependent) &
            (relationships['sdg_indicator_1'].isin(variable_columns))
        ]

        # Check if there are any relationships
        if dependent_relationships.empty:
            # No trends defined; keep the dependent variable unchanged
            st.warning(f"No valid relationships for {dependent}. It will remain unchanged.")
            simulated_values[dependent] = simulation_data[dependent].mean()
            continue

        # Initialize simulated value
        simulated_value = 0
        valid_relationship = False

        # Use relationships to predict dependent variable
        for _, relationship in dependent_relationships.iterrows():
            independent = relationship['sdg_indicator_1']
            trend = relationship['trend']

            # Train a model for the specific trend
            # Pass only the independent and dependent columns required for the specific pair
            model, _, _ = train_model(
                independent=independent,
                dependent=dependent,
                trend=trend,
                data=simulation_data[[independent, dependent]]
            )

            if model and trend != 'none':
                # Prepare the input for prediction using only the specific independent variable
                input_value = np.array([[slider_values[independent]]])
                simulated_value += model.predict(input_value)[0]
                valid_relationship = True

        if valid_relationship:
            simulated_values[dependent] = simulated_value
        else:
            # No valid relationships, keep the dependent unchanged
            simulated_values[dependent] = simulation_data[dependent].mean()

    return simulated_values

