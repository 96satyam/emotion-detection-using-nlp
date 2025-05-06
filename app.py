from flask import Flask, render_template, request
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Home page route
@app.route('/')
def index():
    return render_template("home.html")

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        input_text = request.form.get('text')
        if not input_text or input_text.strip() == "":
            return render_template('home.html', results="Please enter valid text.", input_text="")

        try:
            data = CustomData(text=input_text)
            pred_df = data.get_data_as_data_frame()

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            return render_template('home.html', results=results[0], input_text=input_text)

        except Exception as e:
            return render_template('home.html', results=f"Error: {str(e)}", input_text=input_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
