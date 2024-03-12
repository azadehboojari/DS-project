import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
csv_file_path = './pvwatts_irradiance_tucson.csv'

df = pd.read_csv(csv_file_path)
df.head()


# month specification
months=[0,1,2,3,4,5,6,7,8,9,10,11]
specific_xticks = ['Jan', 'Feb', 'Mar', 'Apr','May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# -------------------------------------------------------------------------------------------
# Daily average Beam Irradaiance
daily_avg= df.groupby(['Month', 'Day']).agg({'Beam Irradiance (W/m^2)': 'mean',
                                             'DC Array Output (W)': 'mean', 
                                             'Plane of Array Irradiance (W/m^2)': 'mean',
                                             'Diffuse Irradiance (W/m^2)': 'mean'
                                             })
daily_avg["Month"]=[index[0] for index in daily_avg.index]
print(daily_avg)
# -------------------------------------------------------------------------------------------

# Monthly average of Beam Irradiance Bar Chart
monthly_avg_beam= df.groupby(['Month'])['Beam Irradiance (W/m^2)'].mean()
print("beam",monthly_avg_beam)
# # Create a bar chart
# monthly_avg_beam.plot(kind='bar', color='skyblue')

# # Set labels and title
# plt.ylabel('Average Beam Irradiance (W/m^2)')
# plt.title('Average Beam Irradiance per Month')

# # Rotate x-axis labels for better readability
# plt.xticks(months,specific_xticks,rotation=45)

# # Show the plot
# plt.tight_layout()
# plt.figure(figsize=(10, 6))
# plt.show()

# -------------------------------------------------------------------------------------------

# Monthly average of POA Irradiance Bar Chart
monthly_avg_POA= df.groupby(['Month'])['Plane of Array Irradiance (W/m^2)'].mean()
print("POA",monthly_avg_POA)
# Create a bar chart
monthly_avg_POA.plot(kind='bar', color='skyblue')

# Set labels and title
plt.ylabel('Average Plane of Array Irradiance (W/m^2)')
plt.title('Average Plane of Array Irradiance per Month')

# Rotate x-axis labels for better readability
plt.xticks(months,specific_xticks,rotation=45)

# Show the plot
plt.tight_layout()
plt.figure(figsize=(10, 6))
plt.show()
# -------------------------------------------------------------------------------------------
# Monthly average of diffuse Irradiance Bar Chart
monthly_avg_diffuse= df.groupby(['Month'])['Diffuse Irradiance (W/m^2)'].mean()
print(monthly_avg_diffuse)
# monthly_avg_diffuse.plot(kind='bar', color='skyblue')

# # Set labels and title
# plt.ylabel('Average Diffuse Irradiance (W/m^2)')
# plt.title('Average Diffuse Irradiance per Month')

# # Rotate x-axis labels for better readability
# plt.xticks(months,specific_xticks,rotation=45)

# # Show the plot
# plt.tight_layout()
# plt.figure(figsize=(10, 6))
# plt.show()

# -------------------------------------------------------------------------------------------

# Monthly average of DC output Bar Chart
monthly_avg_DC_output= df.groupby(['Month'])['DC Array Output (W)'].mean()
print("DC",monthly_avg_DC_output)

# # Create a bar chart
# monthly_avg_DC_output.plot(kind='bar', color='skyblue')

# # Set labels and title
# plt.ylabel('Average DC Array Output (W)')
# plt.title('Average DC Array Output per Month')

# # Rotate x-axis labels for better readability
# plt.xticks(months,specific_xticks,rotation=45)

# # Show the plot
# plt.tight_layout()
# plt.figure(figsize=(10, 6))
# plt.show()

# -------------------------------------------------------------------------------------------
# # correlation
# monthly_correlations_beam = daily_avg.groupby(daily_avg['Month']).corr().unstack()['DC Array Output (W)']['Beam Irradiance (W/m^2)']
# monthly_correlations_plane = daily_avg.groupby(daily_avg['Month']).corr().unstack()['DC Array Output (W)']['Plane of Array Irradiance (W/m^2)']
# monthly_correlations_diffuse = daily_avg.groupby(daily_avg['Month']).corr().unstack()['DC Array Output (W)']['Diffuse Irradiance (W/m^2)']
# print(monthly_correlations_beam)
# print(monthly_correlations_plane)
# print(monthly_correlations_diffuse)
# plt.scatter(x=specific_xticks, y= monthly_correlations_beam )
# plt.scatter(x=specific_xticks, y= monthly_correlations_plane )
# plt.scatter(x=specific_xticks, y= monthly_correlations_diffuse )
# plt.plot(specific_xticks, monthly_correlations_beam,color='cornflowerblue', marker='o', linestyle='-', linewidth=1, label='Beam Irradiance')
# plt.plot(specific_xticks, monthly_correlations_plane,color='skyblue', marker='o', linestyle='-', linewidth=1, label='POA Irradiance')
# plt.plot(specific_xticks, monthly_correlations_diffuse,color='pink', marker='o', linestyle='-', linewidth=1, label='Diffuse Irradiance (W/m^2)')
# plt.title('Correlation Comparison')
# plt.ylabel('Correlations')
# plt.legend(loc='center right', fontsize='small')
# plt.show()


# -------------------------------------------------------------------------------------------

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

X = np.array([(monthly_avg_diffuse).values, (monthly_avg_POA.values), (monthly_avg_beam.values)]).transpose(1,0)
y = (monthly_avg_DC_output).values
print(y.shape, X.shape)

# Perform multiple linear regression
linear_regression_model = LinearRegression()
linear_regression_model.fit(X, y)

# Assess coefficients
coefficients = linear_regression_model.coef_
print("Coefficients:")
print(coefficients)

# Calculate feature importances using Random Forests
random_forest_model = RandomForestRegressor()
random_forest_model.fit(X, y)
feature_importances = random_forest_model.feature_importances_
print("Feature Importances:")
print(feature_importances)
# Obtain predictions from the linear regression model
y_pred_linear = linear_regression_model.predict(X)

# Obtain predictions from the random forest model
y_pred_rf = random_forest_model.predict(X)

# Calculate mean squared error for the linear regression model
mse_linear = mean_squared_error(y, y_pred_linear)
print("Mean Squared Error (Linear Regression):", mse_linear)

# Calculate mean squared error for the random forest model
mse_rf = mean_squared_error(y, y_pred_rf)
print("Mean Squared Error (Random Forest):", mse_rf)

# -------------------------------------------------------------------------------------------
# forcasting
# from statsmodels.tsa.arima.model import ARIMA

# Sample data (replace this with your array of 12 values)
data = np.array(monthly_avg_DC_output)
# Fit ARIMA model
# model = ARIMA(data, order=(1, 1, 1))  # Example order, you may need to adjust this
# model_fit = model.fit()

# Forecast next 6 months
forecast = [671.14167646, 623.91608039, 592.08133444, 570.62155133, 556.15552443, 546.40398439]

print("Forecasted values for the next 6 months:")
print(forecast)

# Plot previous data and forecasted values
plt.plot(np.arange(1, len(data) + 1), data, label='Current Data', color='skyblue')

# Plot forecast
plt.plot(np.arange(len(data) + 1, len(data) + 7), forecast, color='red', linestyle='--', label='Forecast')
months_added= ['Jan', 'Feb', 'Mar', 'Apr','May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
# Set xtick labels
plt.xticks(np.arange(1, len(data) + 7), months_added, rotation=90)

plt.xlabel('Month')
plt.ylabel('DC Output (W)')
plt.title('Monthly Average DC Output with Forecast')
plt.legend(loc='center right', fontsize='small')
plt.grid(True)
plt.show()

