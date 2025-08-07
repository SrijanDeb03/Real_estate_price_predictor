import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_excel("Updated_Real_Estate_Data.xlsx", sheet_name="Sheet1")

# Function to clean price column
def clean_price(price):
    return float(price.replace('₹', '').replace(',', '').replace(' Cr', 'e7').replace(' L', 'e5'))

# Clean & preprocess price column
df = df[df['Price'].str.match(r'₹[\d.,]+ (Cr|L)$', na=False)]
df['Price'] = df['Price'].apply(clean_price)

df['Balcony'] = df['Balcony'].map({'Yes': 1, 'No': 0})

# Manual Target Encoding for Location
location_price_map = df.groupby('Location')['Price'].mean().to_dict()
df['Location_Encoded'] = df['Location'].map(location_price_map)

# Select Features
features = ['Total_Area', 'Price_per_SQFT', 'Baths', 'Balcony', 'Location_Encoded', 'Election_Cycle', 'Inflation', 'Sudden_Calamity', 'Government_Policy_Impact']
X = df[features]
y = df['Price']

# Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X.drop(columns=['Location_Encoded']))

# Combine with Encoded Location
X_final = np.hstack((X_poly, X[['Location_Encoded']].values))

# Normalize Price (Log Transform)
y = np.log1p(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Scale features
scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.transform(X_test)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# Reshape for CNN-LSTM
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# CNN-LSTM Model
model = Sequential([
    Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(X_train_lstm.shape[1], 1)),
    BatchNormalization(),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

# Compile Model
optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Train Model
history = model.fit(X_train_lstm, y_train_scaled, epochs=60, batch_size=32, validation_data=(X_test_lstm, y_test_scaled), verbose=1)

# Predict
y_pred_lstm_scaled = model.predict(X_test_lstm).flatten()
y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm_scaled.reshape(-1, 1)).flatten()

# Performance Metrics
mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
rmse_lstm = np.sqrt(mean_squared_error(y_test, y_pred_lstm))
r2_lstm = r2_score(y_test, y_pred_lstm)

# SVM Model
svm_model = SVC(kernel='rbf', C=100, gamma=0.1)
svm_model.fit(X_train_scaled, (y_train > np.median(y_train)).astype(int))


y_pred_svm = svm_model.predict(X_test_scaled)
median_price = np.median(y_test)
y_test_class = (y_test > median_price).astype(int)
conf_matrix = confusion_matrix(y_test_class, y_pred_svm)
accuracy = accuracy_score(y_test_class, y_pred_svm) * 100

# Plot Training Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()

# Plot Actual vs Predicted Prices
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_lstm, alpha=0.7, color='green')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

# Plot Residuals
residuals = y_test - y_pred_lstm
plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=30, kde=True, color='purple')
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals")
plt.show()

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Below Median', 'Above Median'], yticklabels=['Below Median', 'Above Median'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM Confusion Matrix")
plt.show()

# Function to format price
def format_price(price):
    if price >= 1e7:
        return f"₹{price / 1e7:.2f} Cr"
    elif price >= 1e5:
        return f"₹{price / 1e5:.2f} L"
    else:
        return f"₹{price:.2f}"

# Function to predict price from user input
def predict_price():
    print("\n🔹 Enter Property Details 🔹")
    total_area = float(input("Enter Total Area (sqft): "))
    price_per_sqft = float(input("Enter Price per Sqft: "))
    baths = int(input("Enter Number of Baths: "))
    balcony = int(input("Has Balcony? (1 = Yes, 0 = No): "))
    election_cycle = int(input("Election Cycle Impact (0 or 1): "))
    inflation = float(input("Inflation Rate: "))
    sudden_calamity = int(input("Sudden Calamity Impact (0 or 1): "))
    government_policy = int(input("Government Policy Impact (0 or 1): "))

    location = input("Enter Property Location: ")
    location_encoded = location_price_map.get(location, np.median(df['Location_Encoded']))

    user_data = np.array([[total_area, price_per_sqft, baths, balcony, election_cycle, inflation, sudden_calamity, government_policy]])
    user_data_poly = poly.transform(user_data)
    user_data_final = np.hstack((user_data_poly, [[location_encoded]]))
    user_data_scaled = scaler_x.transform(user_data_final)
    user_data_lstm = user_data_scaled.reshape((user_data_scaled.shape[0], user_data_scaled.shape[1], 1))

    predicted_price_scaled = model.predict(user_data_lstm).flatten()
    predicted_price = np.expm1(scaler_y.inverse_transform(predicted_price_scaled.reshape(-1, 1)).flatten()[0])


    price_class = svm_model.predict(user_data_scaled)[0]
    price_category = "Above Median Price" if price_class == 1 else "Below Median Price"
    adjustment_factor = 1.1 if price_class == 1 else 0.9
    adjusted_price = predicted_price * adjustment_factor
    formatted_adjusted_price = format_price(adjusted_price)

    print("\n🔹 Prediction Results 🔹")
    print(f"Estimated Property Price: {formatted_adjusted_price}")
    print(f"Price Category (SVM Prediction): {price_category}")

print(f"CNN-LSTM Model -> MAE: {mae_lstm:.2f}, RMSE: {rmse_lstm:.2f}, R²: {r2_lstm:.2f}")
print(f"SVM Model Accuracy: {accuracy:.2f}%")

predict_price()
