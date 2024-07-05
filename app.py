from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the model from the .pkl file
loaded_model = pickle.load(open("diapredict.pkl", "rb"))

@app.route('/', methods=['GET'])
def index():
    return render_template('diahome.html')

@app.route('/submit-form', methods=['POST'])
def submit_form():
    # Get the input values from the form
    pregnancies = int(request.form['pregnancies'])
    glucose = int(request.form['glucose'])
    blood_pressure = int(request.form['bloodPressur'])
    skin_thickness = int(request.form['skinThickness'])
    insulin = int(request.form['insulin'])
    bmi = float(request.form['bmi'])
    DiabetesPedigreeFunction=int(request.form['DiabetesPedigreeFunction'])
    age = int(request.form['age'])
   
    # Create a list of input values
    input_values = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi,DiabetesPedigreeFunction, age]

    # Make a prediction with the loaded model
    prediction = loaded_model.predict([input_values])[0]

    # Render the result.html template with the predicted output
    return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)