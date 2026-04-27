import pandas as pd
import numpy as np
import joblib
import requests
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# ------------------ WEATHER MODULE (unchanged) ------------------ #
API_KEY = "fbe50b1cc15d344538591172e2fd6f2e"
DELHI_LAT = 28.6139
DELHI_LON = 77.2090

def get_weather_data():
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={DELHI_LAT}&lon={DELHI_LON}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    return {
        "temperature": data['main']['temp'],
        "humidity": data['main']['humidity'],
        "rainfall": data.get('rain', {}).get('1h', 0),
        "wind_speed": data['wind']['speed'],
        "cloud_cover": data['clouds']['all']
    }

# ------------------ CROP RECOMMENDATION MODEL (unchanged) ------------------ #
class CropModel:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.model = None
        self.id_to_label = None
        self.feature_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    
    def train(self):
        data = pd.read_csv(self.dataset_path)
        unique_crops = sorted(data['label'].unique())
        label_to_id = {crop: idx for idx, crop in enumerate(unique_crops)}
        self.id_to_label = {idx: crop for crop, idx in label_to_id.items()}
        data['label_id'] = data['label'].map(label_to_id)
        X = data[self.feature_columns]
        y = data['label_id']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
        self.model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=42
        )
        self.model.fit(X_train, y_train)
        accuracy = self.model.score(X_test, y_test)
        print(f"✅ Crop model trained! Accuracy: {accuracy:.4f}")
    
    def predict(self, input_data):
        try:
            features = [
                float(input_data['N']),
                float(input_data['P']),
                float(input_data['K']),
                float(input_data['temperature']),
                float(input_data['humidity']),
                float(input_data['ph']),
                float(input_data['rainfall'])
            ]
        except KeyError as e:
            raise ValueError(f"Missing field: {e}")
        features_array = np.array(features).reshape(1, -1)
        predicted_id = self.model.predict(features_array)[0]
        predicted_crop = self.id_to_label[predicted_id]
        probabilities = self.model.predict_proba(features_array)[0]
        confidence = float(max(probabilities))
        return {'crop': predicted_crop, 'confidence': round(confidence, 3)}
    
    def save(self, path="crop_model.pkl"):
        joblib.dump((self.model, self.id_to_label), path)
    
    def load(self, path="crop_model.pkl"):
        self.model, self.id_to_label = joblib.load(path)

# ------------------ YIELD PREDICTION MODEL (NEW) ------------------ #
class YieldModel:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.model = None
        self.state_dict = None
        self.crop_dict = None
        self.feature_columns = [
            'State Code', 'Crop', 'Area_ha', 'Soil_Test_N_ppm', 'Soil_Test_P_ppm',
            'Soil_Test_K_ppm', 'Temperature_C', 'Humidity_%', 'pH', 'Rainfall_mm'
        ]
    
    def train(self):
        df = pd.read_csv(self.dataset_path)
        
        # Build state mapping (State Name → State Code)
        state_map = df[['State Name', 'State Code']].drop_duplicates()
        self.state_dict = dict(zip(state_map['State Name'], state_map['State Code']))
        
        # Build crop mapping (unique crop names → integers)
        unique_crops = df['Crop'].unique()
        self.crop_dict = {crop: idx for idx, crop in enumerate(sorted(unique_crops))}
        
        # Encode crop
        df['Crop'] = df['Crop'].map(self.crop_dict)
        
        # Select only required features + target
        X = df[self.feature_columns]
        y = df['Yield_kg_per_ha']
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
        
        # Train Random Forest
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        r2 = self.model.score(X_test, y_test)
        print(f"✅ Yield model trained! Features: {self.feature_columns}")
        print(f"   R² score: {r2:.4f}")
    
    def predict(self, input_data):
        """
        input_data from frontend:
            state_name, crop, area_ha,
            N_req (→ Soil_Test_N_ppm),
            P_req (→ Soil_Test_P_ppm),
            K_req (→ Soil_Test_K_ppm),
            Temperature, Humidity, pH, Rainfall
        """
        # Map state name to state code
        state_code = self.state_dict.get(input_data['state_name'])
        if state_code is None:
            raise ValueError(f"Unknown state: {input_data['state_name']}")
        
        # Map crop name to encoded integer (case-insensitive)
        crop_name = input_data['crop'].strip().lower()
        crop_code = None
        for key, val in self.crop_dict.items():
            if key.lower() == crop_name:
                crop_code = val
                break
        if crop_code is None:
            # Fallback: use first crop code (or raise error)
            crop_code = list(self.crop_dict.values())[0]
            print(f"⚠️ Crop '{crop_name}' not in training set, using default (index {crop_code})")
        
        # Build feature vector in exact order
        features = [
            float(state_code),
            float(crop_code),
            float(input_data['area_ha']),
            float(input_data.get('N_req', 0)),
            float(input_data.get('P_req', 0)),
            float(input_data.get('K_req', 0)),
            float(input_data['Temperature']),
            float(input_data['Humidity']),
            float(input_data['pH']),
            float(input_data['Rainfall'])
        ]
        
        # Predict
        prediction = self.model.predict([features])[0]
        return max(0, prediction)   # yield cannot be negative
    
# ------------------ FLASK APP ------------------ #
app = Flask(__name__)
CORS(app)

# ----- Load / Train Crop Recommendation Model -----
CROP_MODEL_PATH = "crop_model.pkl"
crop_model = CropModel("Crop_recommendation.csv")

if os.path.exists(CROP_MODEL_PATH):
    try:
        crop_model.load(CROP_MODEL_PATH)
        print("✅ Crop model loaded from file")
    except:
        print("⚠️ Error loading crop model. Retraining...")
        crop_model.train()
        crop_model.save(CROP_MODEL_PATH)
else:
    print("🚀 Training crop model for first time...")
    crop_model.train()
    crop_model.save(CROP_MODEL_PATH)

# ----- Load / Train Yield Prediction Model (NEW) -----
YIELD_MODEL_PATH = "yield_model.pkl"
YIELD_DATASET_PATH = "Crops_Yield_Dataset.csv"   # <-- update path if needed

yield_model = YieldModel(YIELD_DATASET_PATH)

if os.path.exists(YIELD_MODEL_PATH):
    try:
        # For simplicity, retrain every time (or implement load)
        # You can implement load/save similar to crop_model
        yield_model.train()
        print("✅ Yield model retrained (you can implement persistence)")
    except:
        yield_model.train()
else:
    print("🚀 Training yield model...")
    yield_model.train()
    # Optionally save:
    # joblib.dump((yield_model.model, yield_model.state_dict, yield_model.crop_dict, yield_model.feature_columns), YIELD_MODEL_PATH)

# ----- Existing Routes -----
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No input data provided'}), 400
        result = crop_model.predict(data)
        return jsonify({'success': True, 'crop': result['crop'], 'confidence': result['confidence']})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/weather', methods=['GET'])
def weather():
    try:
        data = get_weather_data()
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# ----- NEW: Yield Prediction Endpoint -----
@app.route('/predict_yield', methods=['POST'])
def predict_yield():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No input data provided'}), 400
        
        # Validate required fields
        required = ['state_name', 'crop', 'area_ha', 'Temperature', 'Humidity', 'pH', 'Rainfall']
        for field in required:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing field: {field}'}), 400
        
        # Optional fields: N_req, P_req, K_req, Wind_Speed, Solar_Radiation
        # Provide defaults if missing
        data.setdefault('N_req', 0)
        data.setdefault('P_req', 0)
        data.setdefault('K_req', 0)
        
        prediction_kg_ha = yield_model.predict(data)
        
        return jsonify({
            'success': True,
            'yield_kg_per_ha': round(prediction_kg_ha, 2)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': crop_model.model is not None})

@app.route('/')
def home():
    return jsonify({
        "message": "🌱 Crop Recommendation + 🌾 Yield Prediction + ☁️ Weather API"
    })

if __name__ == '__main__':
    print("🌍 Server running on http://localhost:5000")
    print("   POST /predict         -> crop recommendation")
    print("   POST /predict_yield   -> yield prediction (kg/ha)")
    print("   GET  /weather         -> live weather data")
    app.run(debug=True, host='0.0.0.0', port=5000)