from flask import Flask,render_template,request, url_for,send_from_directory, jsonify
from flask_cors import CORS, cross_origin
import predictor
import pandas as pd 

app = Flask(__name__)
cors = CORS(app)

active_drivers = [
                  ['Daniel Ricciardo','McLaren'], 
                  ['Mick Schumacher','Haas F1 Team'], 
                  ['Carlos Sainz','Ferrari'],
                  ['Valtteri Bottas','Mercedes'], 
                  ['Lance Stroll','Aston Martin'], 
                  ['George Russell','Williams'],
                  ['Lando Norris','McLaren'], 
                  ['Charles Leclerc','Ferrari'], 
                  ['Lewis Hamilton','Mercedes'], 
                  ['Yuki Tsunoda','AlphaTauri'],
                  ['Max Verstappen','Red Bull'], 
                  ['Pierre Gasly','AlphaTauri'], 
                  ['Fernando Alonso','Alpine F1'],
                  ['Sergio Pérez','Red Bull'], 
                  ['Esteban Ocon','Alpine F1'], 
                  ['Alexander Albon','Williams'],
                #   ['Guanyu Zhou','Alfa Romeo'],
                  ['Kevin Magnussen','Haas F1 Team'],
                  ['Nico Hülkenberg','Aston Martin'],
                #   ['Logan Sargeant',''],
                #   ['Oscar Piastri',''],
                  ]


@app.route("/images/<path:path>")
def static_dir(path):
    return send_from_directory("images", path)

@app.route('/')
def get_input():
    return render_template('index.html')

@app.route('/predict1', methods=['POST'])
def predict_pos():
    content = request.json
    circuit = content["circuit"]

    res = []
    for row in active_drivers:
        #for elem in row:
        driver = row[0]
        constructor = row[1]
        quali = predictor.getQualifData(circuit, driver)
        my_rangeprediction, driver_confidence, constructor_reliability = predictor.pred(driver,constructor,quali,circuit)
        #print ("%s: %s : %s : %s : %s " % (driver, constructor, my_rangeprediction, driver_confidence, constructor_reliability))
        driverproba, constructorproba = predictor.getproba(driver,constructor)
        # predpercentage = "{:.2%}".format(driverproba)
        predpercentage = driverproba
        elem = [driver, constructor, my_rangeprediction, driver_confidence, constructor_reliability, predpercentage]
        res.append(elem)
    
    # print (res)
    df1 = pd.DataFrame(res, columns = ['driver','constructor','podium', 'driver_confidence', 'constructor_reliability', 'prediction'] )
    df1 = df1.sort_values(['podium','prediction'], ascending=[True, False]).head(5)
    df1 = df1.drop(['prediction'],axis=1)

    print(df1)
    return df1.to_json(orient="records")

@app.route('/predict',methods=['POST'])
def predict_position():
    circuit = request.form['circuit']
    # weather = request.form['weather']

    #res = predictor.pred(circuit, weather)
    res = []
    for row in active_drivers:
        #for elem in row:
        driver = row[0]
        constructor = row[1]
        quali = predictor.getQualifData(circuit, driver)
        my_rangeprediction, driver_confidence, constructor_reliability = predictor.pred(driver,constructor,quali,circuit)
        #print ("%s: %s : %s : %s : %s " % (driver, constructor, my_rangeprediction, driver_confidence, constructor_reliability))
        driverproba, constructorproba = predictor.getproba(driver,constructor)
        # predpercentage = "{:.2%}".format(driverproba)
        predpercentage = driverproba
        elem = [driver, constructor, my_rangeprediction, driver_confidence, constructor_reliability, predpercentage]
        res.append(elem)
    
    # print (res)
    df1 = pd.DataFrame(res, columns = ['Driver','Constructor','podium', 'driver_confidence', 'constructor_reliability', 'Prediction'] )
    df1 = df1.sort_values(['podium','Prediction'], ascending=[True, False]).head(5)
    df1 = df1.drop(['Prediction'],axis=1)
    
    return render_template('index.html',tables=[df1.to_html(classes='driver')])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8443, debug=True)


