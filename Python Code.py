import pandas as pd

# Load datasets
cab_rides = pd.read_csv("cab_rides.csv")
weather = pd.read_csv("weather.csv")

# Check missing values
print("Missing values in cab_rides dataset:\n", cab_rides.isnull().sum())
print("\nMissing values in weather dataset:\n", weather.isnull().sum())

# Function to handle outliers using the IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Apply outlier removal on cab_rides.csv
cab_rides = remove_outliers(cab_rides, "price")
cab_rides = remove_outliers(cab_rides, "distance")

# Apply outlier removal on weather.csv
weather = remove_outliers(weather, "temp")
weather = remove_outliers(weather, "humidity")
weather = remove_outliers(weather, "wind")

# Check dataset after outlier removal
print("\nOutliers removed successfully.")

# Convert timestamp to datetime format
cab_rides["time_stamp"] = pd.to_datetime(cab_rides["time_stamp"], unit="ms")
weather["time_stamp"] = pd.to_datetime(weather["time_stamp"], unit="s")

# Extract hour from timestamp to identify peak vs. non-peak time
cab_rides["hour"] = cab_rides["time_stamp"].dt.hour
cab_rides["peak_time"] = cab_rides["hour"].apply(lambda x: "Peak" if (7 <= x <= 9 or 17 <= x <= 20) else "Non-Peak")

# Create fare per distance feature
cab_rides["fare_per_km"] = cab_rides["price"] / (cab_rides["distance"] + 1e-5)  # Avoid division by zero

# Categorize temperature in weather.csv
def categorize_temp(temp):
    if temp < 10:
        return "Cold"
    elif 10 <= temp <= 25:
        return "Moderate"
    else:
        return "Hot"

weather["temp_category"] = weather["temp"].apply(categorize_temp)

# Categorize wind speed
weather["wind_category"] = weather["wind"].apply(lambda x: "Calm" if x < 10 else "Windy")

# Categorize weather conditions
weather["weather_condition"] = weather["rain"].apply(lambda x: "Rainy" if x > 0 else "Clear")

# Check newly created features
print("\nFeature Engineering Completed.")
print(cab_rides.head())
print(weather.head())

