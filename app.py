from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model and label encoder
with open('random_forest_fish_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Detect AJAX requests or requests accepting JSON
    is_ajax = (
        request.headers.get('X-Requested-With') == 'XMLHttpRequest' or
        'application/json' in request.headers.get('Accept', '') or
        request.is_json
    )
    
    try:
        if request.is_json:
            data = request.get_json()
            ph = float(data.get('ph'))
            temperature = float(data.get('temperature'))
            turbidity = float(data.get('turbidity'))
        else:
            ph = float(request.form['ph'])
            temperature = float(request.form['temperature'])
            turbidity = float(request.form['turbidity'])

        features = np.array([[ph, temperature, turbidity]])
        pred_encoded = model.predict(features)[0]
        predicted_fish = label_encoder.inverse_transform([pred_encoded])[0]

        if is_ajax:
            return jsonify({
                "success": True,
                "prediction": predicted_fish
            })
        return render_template('result.html', prediction=predicted_fish)
    except Exception as e:
        if is_ajax:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 400
        return render_template('result.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

