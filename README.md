# TSF_INTERNSHIP
*Dependency-of-Score-on-Study-Hours
A project to elaborate the deployment of Machine Learning models using Flask API. This project is a part of my internship program at The Sparks Foundation which predicts
marks based on number of study hours.

*Requirements
Scikit-learn
Pandas (Library for Machine Learning model)
Flask (API)


*Project Components
This project has four major components :

->model.pkl - This contains Machine Learning model to predict score based on training data in 'student_scores.csv' file.

->app.py - This contains Flask APIs that receives details through GUI or API calls, computes the predicted value based on our model and returns it.

->static - A web application often requires a static file such as a javascript file or a CSS file supporting the display of a web page.

->template - This folder contains the HTML template to allow user to enter the number of hours of study and displays the percentage calculated.


*Running the project

Ensure that you are in the project home directory. Create the machine learning model by running below command - python model.py. This would create a serialized version of our model into a file model.pkl

Run app.py using below command to start Flask API.

python app.py

By default, flask will run on port 5000.

Navigate to URL http://localhost:5000 You should be able to view the homepage as below :

Enter a valid numerical value in input box and click Predict.

If your input is valid, you should be able to see the expected percentage on the HTML page!
