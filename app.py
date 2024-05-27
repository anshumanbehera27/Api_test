from flask import Flask, request, jsonify
import numpy as np
import pickle

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Initialize the Flask application
app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the form data
        cgpa = request.form.get('cgpa')
        iq = request.form.get('iq')
        profile_score = request.form.get('profile_score')

        # Debugging output: print received values
        print(f"Received values: cgpa={cgpa}, iq={iq}, profile_score={profile_score}")

        # Ensure all inputs are present
        if cgpa is None or iq is None or profile_score is None:
            return jsonify({'error': 'Missing input data'}), 400

        # Convert inputs to appropriate numeric types
        cgpa = float(cgpa)
        iq = int(iq)
        profile_score = float(profile_score)

        # Debugging output: print converted values
        print(f"Converted values: cgpa={cgpa}, iq={iq}, profile_score={profile_score}")

        # Create an input array for the model
        input_query = np.array([[cgpa, iq, profile_score]])

        # Debugging output: print input query
        print(f"Input query: {input_query}")

        # Make prediction
        result = model.predict(input_query)[0]

        # Return the result as JSON
        return jsonify({'placement': str(result)})
    
    except ValueError as ve:
        # Specific handling for ValueError to provide more detailed feedback
        print(f"ValueError: {ve}")
        return jsonify({'error': f'ValueError: {str(ve)}'}), 400
    except Exception as e:
        # General exception handling
        print(f"Exception: {e}")
        return jsonify({'error': f'Exception: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
