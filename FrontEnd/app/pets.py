from flask import Flask
app = Flask(__name__)

from flask import render_template
import pymysql as mdb
from flask import jsonify
from flask import request


@app.route('/')
@app.route("/home")
def index_jquery():
    
    return render_template('home.html')

@app.route("/presentation")
def show_presentation():
    return render_template('presentation.html')

@app.route("/seeall")
def my_jquery():
    db = mdb.connect('localhost', 'root','shreddie131','HomewardBound',charset='utf8');
    with db:
        cur = db.cursor()

        cur.execute("SELECT AnimalID, AdoptionPercentage FROM Demo_Critter_Predictions ORDER BY AdoptionPercentage ASC")
        query_results_predictions = cur.fetchall()

        information = []
        for i in query_results_predictions:
            information.append([str(int(i[0])), int(round(i[1]))])

        
    return render_template('seeall.html', information = information)

@app.route("/result")
def resultpage():

    db = mdb.connect('localhost', 'root','shreddie131','HomewardBound',charset='utf8');
    
    try:
        IDnumber = int(request.args.get('ID'))

        with db:
            cur = db.cursor()
            cur.execute("SELECT * FROM Demo_Critter_Bio WHERE AnimalID = '" + str(IDnumber) + "' LIMIT 1")
            query_results = cur.fetchall()

            cur.execute("SELECT AdoptionPercentage,  ImprovementFix,ImprovementName FROM Demo_Critter_Predictions WHERE AnimalID = '" + str(IDnumber) + "' LIMIT 1")
            query_results_predictions = cur.fetchall()

            cur.execute("SELECT Days FROM Demo_Critter_Days WHERE AnimalID = '" + str(IDnumber) + "' LIMIT 1")
            length_of_stay = int(round(cur.fetchall()[0][0]))


    
        theID = int(query_results[0][1])

        if query_results[0][2] == '':
            name = "I do not yet have a name."
        else:
            name = "Hi! My name is " + query_results[0][2]

        breed = query_results[0][3]
        age = query_results[0][4]
        size = query_results[0][5]
        sex = query_results[0][6]
        fixed = 'Yes' if query_results[0][7] == '1' else 'No'

        theList = [name, breed, age, size, sex, fixed]


        theDog = "static/PetPics/" + str(int(query_results[0][1]))

         ## prediction related information
        thePrediction = [int(round(query_results_predictions[0][0])), query_results_predictions[0][1], query_results_predictions[0][2]]

        thePercentage = str(thePrediction[0]) + '%'

        

        return render_template("result2.html", length_of_stay = length_of_stay, theID = theID, theDog = theDog, theList= theList, thePrediction = thePrediction, thePercentage = thePercentage)
    
    except

        return render_template("error.html")




@app.route("/recommendation")
def newpage():
    db = mdb.connect('localhost', 'root','shreddie131','HomewardBound',charset='utf8');
    IDnumber =  request.args.get('alignment')
    
    with db:
        cur = db.cursor()
        cur.execute("SELECT * FROM Demo_Critter_Bio WHERE AnimalID = '" + str(IDnumber) + "' LIMIT 1")
        query_results = cur.fetchall()
        cur.execute("SELECT AdoptionPercentage,  ImprovementFix, ImprovementName, WithFix, WithName, WithBoth FROM Demo_Critter_Predictions WHERE AnimalID = '" + str(IDnumber) + "' LIMIT 1")
        query_results_predictions = cur.fetchall()



    if query_results[0][2] == '':
        name = "I do not yet have a name."
    else:
        name = "Hi! My name is " + query_results[0][2]

    breed = query_results[0][3]
    age = query_results[0][4]
    size = query_results[0][5]
    sex = query_results[0][6]
    fixed = 'Yes' if query_results[0][7] == '1' else 'No'

    theList = [name, breed, age, size, sex, fixed]
    theDog = "static/PetPics/" + str(int(query_results[0][1]))

    
    thePrediction = [int(round(query_results_predictions[0][0])), 
    query_results_predictions[0][1], 
    query_results_predictions[0][2],
    int(round(query_results_predictions[0][3])),
    int(round(query_results_predictions[0][4])),
    int(round(query_results_predictions[0][5]))]

    allPercentages = [str(thePrediction[0]) + '%', 
                    str(thePrediction[3] - thePrediction[0]) + '%',
                    str(thePrediction[4] - thePrediction[0]) + '%',
                    str(thePrediction[5] - thePrediction[0]) + '%']
    

    print thePrediction[4] - thePrediction[0]
    color = ["green","red"]

    return render_template("result3.html", color= color, thePrediction = thePrediction, theDog = theDog, theList = theList, allPercentages = allPercentages)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port = 5000)


