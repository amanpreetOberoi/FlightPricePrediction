#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, render_template
#from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("FlightPricePrediction.pkl", "rb"))



@app.route("/")
#@cross_origin()
def home():
    return render_template("home.html")




@app.route("/predict", methods = ["GET", "POST"])
#@cross_origin()
def predict():
    if request.method == "POST":

        # Date_of_Journey
        date_dep = request.form["Dep_Time"]
        Journey_day = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").day)
        Journey_month = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").month)
        # print("Journey Date : ",Journey_day, Journey_month)

        # Departure
        Dep_hour = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").hour)
        Dep_min = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").minute)
        # print("Departure : ",Dep_hour, Dep_min)

        # Arrival
        date_arr = request.form["Arrival_Time"]
        Arrival_hour = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").hour)
        Arrival_min = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").minute)
        # print("Arrival : ", Arrival_hour, Arrival_min)

        # Duration
        dur_hour = abs(Arrival_hour - Dep_hour)
        dur_min = abs(Arrival_min - Dep_min)
        # print("Duration : ", dur_hour, dur_min)

        # Total Stops
        Total_stops = int(request.form["stops"])
        # print(Total_stops)

        # Airline
        airline=request.form['airline']
        al=0
        if(airline=='Jet Airways'):
            al=4 
        elif (airline=='IndiGo'):
            al=3
        elif (airline=='Air India'):
            al = 1
        elif (airline=='Multiple carriers'):
            al=6
        elif (airline=='SpiceJet'):
            al=8
        elif (airline=='Vistara'):
            al=10
        elif (airline=='Air Asia'):
            al=0
        elif (airline=='GoAir'):
            al=2
        elif (airline=='Multiple carriers Premium economy'):
            al=7
        elif (airline=='Jet Airways Business'):
            al=5
        elif (airline=='Vistara Premium economy'):
            al=11

        else:
            al=9
        # print(Jet_Airways,
        #     IndiGo,
        #     Air_India,
        #     Multiple_carriers,
        #     SpiceJet,
        #     Vistara,
        #     GoAir,
        #     Multiple_carriers_Premium_economy,
        #     Jet_Airways_Business,
        #     Vistara_Premium_economy,
        #     Trujet)

        # Source
        Source = request.form["Source"]
        sour=0
        if (Source == 'Delhi'):
            sour=2
        elif (Source == 'Kolkata'):
            sour=3
        elif (Source == 'Mumbai'):
            sour=4
        elif (Source == 'Chennai'):
            sour=1
        else:
            sour=0
        # print(s_Delhi,
        #     s_Kolkata,
        #     s_Mumbai,
        #     s_Chennai)

        # Destination
        # Banglore = 0 (not in column)
        dst=0
        Destination = request.form["Destination"]
        if (Destination == 'Cochin'):
            dst=1
        
        elif (Destination == 'Delhi'):
            dst=2

        elif (Destination == 'New_Delhi'):
            dst=5

        elif (Destination == 'Hyderabad'):
            dst=3

        elif (Destination == 'Kolkata'):
            dst=4

        else:
            dst=0
           

        # print(
        #     d_Cochin,
        #     d_Delhi,
        #     d_New_Delhi,
        #     d_Hyderabad,
        #     d_Kolkata
        # )
        

    #     ['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour',
    #    'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hours',
    #    'Duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
    #    'Airline_Jet Airways', 'Airline_Jet Airways Business',
    #    'Airline_Multiple carriers',
    #    'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
    #    'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
    #    'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
    #    'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
    #    'Destination_Kolkata', 'Destination_New Delhi']
        
        prediction=model.predict([[
            al,
            sour,
            dst,
            Total_stops,
            0,
            Journey_day,
            Journey_month,
            Arrival_hour,
            Arrival_min,
            Dep_hour,
            Dep_min,
            dur_hour,
            dur_min
        ]])

        output=round(prediction[0],2)

        return render_template('home.html',prediction_text="Your Flight price is Rs. {}".format(output))


    return render_template("home.html")




if __name__ == "__main__":
    app.run(debug=True)

