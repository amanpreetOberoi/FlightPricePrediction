## Flight-Price-Prediction ##
    * Estimates Flight Prices to help users look for best prices when booking flight tickets.
    * Perform EDA on  features like Departure Time, Date of Journey, and many more to quantify the data and make it more understandable.
    * Optimized  Regression model using RandomsearchCV to reach the best model.
### Dataset ###
    Size of training set: 10683 records

    * Size of test set: 2671 records
    * FEATURES: Airline: The name of the airline.
    * Date_of_Journey: The date of the journey
    * Source: The source from which the service begins.
    * Destination: The destination where the service ends.
    * Route: The route taken by the flight to reach the destination.
    * Dep_Time: The time when the journey starts from the source.
    * Arrival_Time: Time of arrival at the destination.
    * Duration: Total duration of the flight.
    * Total_Stops: Total stops between the source and destination.
    * Additional_Info: Additional information about the flight
    * Price: The price of the ticket
### Packages ###
     * Numpy
     * Pandas
     * Matplotlib
     * Seaborn
     * flask
     * pickle
     
### Model & Score ###
    * Random Forest Regressor
            MAE: 788.806785873344
            MSE: 2572575.8212732906
            RMSE: 1603.9251295722286
