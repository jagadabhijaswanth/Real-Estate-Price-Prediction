from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model safely
with open('linear_regression_model.pkl', 'rb') as f:
    linearmodel = pickle.load(f)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        data = request.get_json()
        # Convert inputs and handle errors
        house_age = float(request.form["House age"])
        distance = float(request.form["Distance to the nearest MRT station"])
        stores = int(request.form["Number of convenience stores"])  # Should be int
        latitude = float(request.form["Latitude"])
        longitude = float(request.form["Longitude"])
        
        # Prepare input features
        input_features = np.array([[house_age, distance, stores, latitude, longitude]])

        # Make prediction
        prediction = linearmodel.predict(input_features)[0]  # Extract first element

        # Return formatted prediction
        return render_template("home.html", prediction_text=f"Predicted House Price: ${prediction:,.2f}")

    except ValueError:
        return render_template("home.html", prediction_text="Invalid input. Please enter numeric values.")
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
