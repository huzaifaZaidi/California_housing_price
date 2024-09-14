# import pickle
# from flask import Flask,request,jsonify,session,render_template

# import numpy as np
# import pandas as pd

# app=Flask(__name__)
# regmodel=pickle.load(open('housing_reg.pkl','rb'))
# scalar=pickle.load(open('scaling.pkl','rb'))
# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     data=request.json['data']
#     print(data)
#     data_values = np.array(list(data.values())).reshape(1, -1)
#     print(data_values)
#     new_data=scalar.transform(data_values)
#     output=regmodel.predict(new_data)
#     print(output[0])
#     return jsonify(output[0])

# if __name__=="__main__":
#     app.run(debug=True)
import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
regmodel = pickle.load(open('housing_reg.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')  # Ensure 'home.html' exists in a 'templates' folder

@app.route('/predict_api', methods=['POST'])
def predict_api():
    # Get the data from the JSON request
    data = request.json['data']
    print(data)

    # Convert the input into a numpy array and reshape it to fit the model
    data_values = np.array(list(data.values())).reshape(1, -1)
    print(data_values)

    # Transform the input using the loaded scalar
    new_data = scalar.transform(data_values)
    
    # Make a prediction using the loaded regression model
    output = regmodel.predict(new_data)
    print(output[0])

    # Return the prediction as a JSON response
    return jsonify(output[0])
    # return f"The estimated price of the house is {round(output[0]*100000,3)} $" 

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in  request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]*100000
    return render_template("home.html",prediction_text="The Predicted House Price is {:.3f} $".format(output))

if __name__ == "__main__":
    app.run(debug=True)
