import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

# Generate 60 monthly dates starting from January 2018
dates = pd.date_range(start='2018-01-01', periods=84, freq='MS')  # 'MS' stands for Month Start

# Convert the dates to Matplotlib's internal numeric format
dates_numeric = mdates.date2num(dates)

# Generate sample data for stock prices and sustainability scores (60 points each)
# You can replace these with your actual data
np.random.seed(0)  # for reproducibility
stock_prices = np.linspace(100, 150, 84) + np.random.normal(0, 3, 84)  # a linear trend with some noise
sustainability_scores = np.linspace(70, 90, 84) + np.random.normal(0, 2, 84)  # a linear trend with noise

# Create the 3D figure and axis
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D line graph
ax.plot(dates_numeric, stock_prices, sustainability_scores,
        color='green', linewidth=2, marker='o', markersize=3, label='Trend Line')

# Set axis labels
ax.set_xlabel('Date')
ax.set_ylabel('Stock Price')
ax.set_zlabel('Sustainability Score')
ax.set_title('3D Line Graph: Monthly Stock Price vs. Sustainability Score (2018-2025)')

# Format the x-axis to show date labels
# Get current x-ticks and set their labels to formatted dates
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.autofmt_xdate()  # rotate date labels for better readability
ax.set_box_aspect((4, 1, 1))  # (x, y, z) aspect ratio

# Optionally, add a legend
ax.legend()

plt.show()