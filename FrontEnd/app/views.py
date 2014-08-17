from flask import render_template
from app import app
import pymysql as mdb
from flask import jsonify
from flask import request

db = mdb.connect('localhost', 'root','shreddie131','HomewardBound',charset='utf8');

@app.route('/')
@app.route('/index')
def index():
	return render_template("index.html",
        title = 'Home', user = { 'nickname': 'Miguel' },
        )

@app.route('/db')
def cities_page():
	with db: 
		cur = db.cursor()
		cur.execute("SELECT Name FROM city LIMIT 15;")
		query_results = cur.fetchall()
	cities = ""
	for result in query_results:
		cities += result[0]
		cities += "<br>"
	return cities


#
#@app.route("/jquery")
#def index_jquery():
#    
#    return render_template('index_js.html')

# @app.route("/circle")
# def testing():
#     return render_template('circle.html')

@app.route("/home")
def index_jquery():
    
    # return render_template('index_js_pets.html')
    return render_template('home.html')


@app.route("/seeall")
def my_jquery():

    with db:
        cur = db.cursor()

        cur.execute("SELECT AnimalID, AdoptionPercentage FROM Demo_Critter_Predictions ORDER BY AdoptionPercentage ASC")
        query_results_predictions = cur.fetchall()

        information = []
        for i in query_results_predictions:
            information.append([str(int(i[0])), int(round(i[1]))])

        
    return render_template('seeall.html', information = information)
#
#@app.route("/db_json")
#def cities_json():
#    with db:
#        cur = db.cursor()
#        cur.execute("SELECT Name, CountryCode, Population FROM city ORDER BY Population DESC;")
#        query_results = cur.fetchall()
#    cities = []
#    for result in query_results:
#        cities.append(dict(name=result[0], country=result[1], population=result[2]))
#    return jsonify(dict(cities=cities))


# @app.route("/db_json")
# def cities_json():
#     with db:
#         cur = db.cursor()
#         cur.execute("SELECT AnimalID, AdoptionProbability FROM Demo_Data WHERE AnimalID = '46904' LIMIT 1")
#         query_results = cur.fetchall()
#     cities = []
#     for result in query_results:
#         cities.append(dict(AnimalID=result[0], AdoptionProbability=result[1]))

#     return jsonify(dict(cities=cities))


@app.route("/result")
def resultpage():


    
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





@app.route("/recommendation")
def newpage():
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




