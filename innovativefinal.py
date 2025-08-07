import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Layer
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K

# Load Dataset
df = pd.read_excel("Updated_Real_Estate_Data.xlsx", sheet_name='Sheet1')

# Preprocess Data
def clean_price(price):
    return float(price.replace('₹', '').replace(',', '').replace(' Cr', 'e7').replace(' L', 'e5'))

df = df[df['Price'].str.match(r'₹[\d.,]+ (Cr|L)$', na=False)]
df['Price'] = df['Price'].apply(clean_price)
df['Balcony'] = df['Balcony'].map({'Yes': 1, 'No': 0})

# Encode Location
location_price_map = df.groupby('Location')['Price'].mean().to_dict()
df['Location_Encoded'] = df['Location'].map(location_price_map)

# Feature Selection
features = ['Total_Area', 'Price_per_SQFT', 'Baths', 'Balcony', 'Location_Encoded',
            'Election_Cycle', 'Inflation', 'Sudden_Calamity', 'Government_Policy_Impact']
X = df[features]
y = df['Price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize Features
scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.transform(X_test)

# Log Transform Target Variable
y_train_log = np.log(y_train)
y_test_log = np.log(y_test)

# Train SVM Model (Binary Classification)
median_price = np.median(y_train)
y_train_class = (y_train > median_price).astype(int)
y_test_class = (y_test > median_price).astype(int)

svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train_scaled, y_train_class)

# Extract SVM Hyperplane Parameters
w = svm_model.coef_[0]  # Weights
b = svm_model.intercept_[0]  # Bias
hyperplane_eq = f"{' + '.join([f'{w[i]:.4f}*x{i+1}' for i in range(len(w))])} + {b:.4f} = 0"
print("\nSVM Hyperplane Equation:")
print(hyperplane_eq)

class SVMHyperplaneLayer(Layer):
    def __init__(self, weights, bias, **kwargs):
        super(SVMHyperplaneLayer, self).__init__(**kwargs)
        self.weights_tensor = tf.constant(weights, dtype=tf.float32)
        self.bias_tensor = tf.constant(bias, dtype=tf.float32)
        self.projection_layer = Dense(len(weights), activation=None)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer='zeros',
            trainable=True,
            name='kernel'
        )
        super().build(input_shape)

    def call(self, inputs):
        expanded_weights = tf.tile(tf.expand_dims(self.weights_tensor, 0), [tf.shape(inputs)[0], 1])
        projected_inputs = self.projection_layer(inputs)
        projected_inputs = tf.ensure_shape(projected_inputs, (None, len(self.weights_tensor)))
        svm_output = tf.reduce_sum(projected_inputs * expanded_weights, axis=1, keepdims=True) + self.bias_tensor
        linear_output = tf.matmul(inputs, self.kernel)
        return svm_output + linear_output

# Build Neural Network with SVM Hyperplane Layer
model = Sequential([
    Dense(32, activation='sigmoid', input_shape=(X_train_scaled.shape[1],)),
    Dense(16, activation='sigmoid'),
    SVMHyperplaneLayer(weights=w, bias=b),
    Dense(1)
])

# Compile Model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train Neural Network
history = model.fit(X_train_scaled, y_train_log, epochs=150, batch_size=16, verbose=1, validation_data=(X_test_scaled, y_test_log))

# Predictions
y_pred_log = model.predict(X_test_scaled).flatten()
y_pred = np.exp(y_pred_log)

# Compute Performance Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R² Score: {r2:.2f}")

# Function to Predict Prices
def predict_price():
    print("\nEnter Property Details")
    total_area = float(input("Enter Total Area (sqft): "))
    price_per_sqft = float(input("Enter Price per Sqft: "))
    baths = int(input("Enter Number of Baths: "))
    balcony = int(input("Has Balcony? (1 = Yes, 0 = No): "))
    election_cycle = float(input("Enter Election Cycle Impact (0-1): "))
    inflation = float(input("Enter Inflation Rate (%): "))
    sudden_calamity = float(input("Enter Sudden Calamity Impact (0-1): "))
    government_policy = float(input("Enter Government Policy Impact (0-1): "))

    print("Available Locations:", df['Location'].unique())
    location = input("Enter Property Location: ")
    location_encoded = location_price_map.get(location, np.median(df['Location_Encoded']))

    user_data = np.array([[total_area, price_per_sqft, baths, balcony, location_encoded, election_cycle, inflation, sudden_calamity, government_policy]])
    user_data_scaled = scaler_x.transform(user_data)
    predicted_price_log = model.predict(user_data_scaled).flatten()
    predicted_price = np.exp(predicted_price_log[0])
    print(f"\nEstimated Property Price: ₹{predicted_price:,.2f}")

# Run Prediction
predict_price()