from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

filename = "breast-cancer-prediction.pkl"
model = pickle.load(open(filename,'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():

    temp = list()

    if request.method == "POST":
        radius = request.form["radius"]
        temp.append(radius)
        texture = request.form["texture"]
        temp.append(texture)
        perimeter = request.form["perimeter"]
        temp.append(perimeter)
        area = request.form["area"]
        temp.append(area)
        smoothness = request.form["smoothness"]
        temp.append(smoothness)


    result = int(model.predict([temp]))

    if result == 1:
        value = "You are suffering from Breast cancer disease"
    else:
        value = "You are Absolutly disease free"

    return render_template('result.html', value=value)





if __name__ == '__main__':
    app.run(debug=True)

