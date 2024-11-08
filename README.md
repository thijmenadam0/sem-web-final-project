# Trajectory Love in Fanfiction
This GitHub contains the code for the final project for the course Semantic Web Technology at the RUG. It also contains visualization notebooks for SHAP and data visualization, these are only important for our own experiments and not for running the rest of the code.

To run the code you first need to install the requirements
```
pip install -r requirements.txt
```

## Data Preparation
The data used for this project can be found in the trajectory_love.csv file. It is data obtained from the GOLEM dataset (https://golemlab.eu/), which is a dataset based on fanfiction. This research uses data where fanfiction stories are tagged with trajectory love keywords. These are keywords like 'Enemies to Lovers', 'Friends to Lovers' and 'Eventual Romance'.

After installing the requirements you have to run the prepare_data.py file at least once to obtain a train, dev and test file. The command below shows the help command, to see all possible uses of this file. The prepare data file can prepare a lot of data, but all of this is included on the GitHub.
```
python3 prepare_data.py -h

usage: prepare_data.py [-h] [-L] [-v {tfidf,w2v,mix}]

options:
  -h, --help            show this help message and exit
  -L, --log_transform   Choose whether to log transform and drop outliers from data.
  -v {tfidf,w2v,mix}, --vectorizer {tfidf,w2v,mix}
                        Choose the vectorizer that is used on the textual features in the data.
```



You now have succesfully created the train, dev and test files in a 70, 20, 10 split respectively.

## Running the models
This code contains options to run 5 regression models: a Decision Trees mode, a Random Forest model, a Logistic Regression model, a Linear Support Vector Regression model and a Support Vector Regression model.

There are a lot of different hyperparameters and other options you can experiment with for every model. Below you can see a few examples of command line arguments with a quick explanation on what the specific command line argument does.

### Help Command
```
python3 fanfiction_model.py -h
```
This is the help command and shows all possible input parameters. It also makes clear that you need to input a model for the code to run. The model choices are stated above and their abbreviations for this program are dt, rf, lr, svrl and svr respectively. 

```
usage: fanfiction_model.py [-h] [-p] [-e] [-t] [-v {tfidf,wTv,mix}] [-L] {svr,svrl,dt,rf,lr} ...

positional arguments:
  {svr,svrl,dt,rf,lr}   Choose the classifying algorithm to use
    svr                 Use Support Vector Regression as Regression model
    svrl                Use Linear Support Vector Regression as Regression model
    dt                  Use Decision Tree Regression as Regression model
    rf                  Use Random Forest Regression as Regression model
    lr                  Use Linear Regression as Regression model

options:
  -h, --help            show this help message and exit
  -p, --print_pred      Also prints the predicted and actual values.
  -e, --exporter        Also exports the predictions to a .csv file
  -t, --test            Makes it so the model also predicts on the test files.
  -v {tfidf,wTv,mix}, --vectorizer {tfidf,wTv,mix}
                        Choose the vectorized data you want to use, either Word2Vec or Tfidf or a mix of both The data
                        should be prepared with either prepare_data files.
  -L, --log_transformed
                        Choose whether the data used in log transformed or not.
```


### Help Command per model
```
python3 fanfiction_model.py svr -h
```

For this demonstration we will be looking at the svr model. This can be called with the command above, the command above will show the help page for the svr, which includes all hyperparameter changes that can be made and how it should be used like so: 
```
usage: fanfiction_model.py svr [-h] [-k {linear,poly,rbf,sigmoid}] [-d DEGREE] [-g {scale,auto}] [-C C]

options:
  -h, --help            show this help message and exit
  -k {linear,poly,rbf,sigmoid}, --kernel {linear,poly,rbf,sigmoid}
                        Choose the kernel for the SVR
  -d DEGREE, --degree DEGREE
                        ONLY FOR POLY KERNEL, changes the degree of the polynomial kernel function
  -g {scale,auto}, --gamma {scale,auto}
                        Choose the gamma (kernel coefficient for rbf, poly and sigmoid) for the SFR
  -C C, --C C           Set the regularization parameter
```

### Example
We can run the model with any combination of commands or just without any hyperparameters like so (the results are also added underneath):
```
python3 fanfiction_model.py svr

Regression Results for Support Vector Regression on the Development set:

Mean Squared Error: 511726.489
Root Mean Squared Error: 715.351
```

With (as an example) the best hyperparameters we found for the svr model that uses Word2Vec (the results are also added underneath):
```
python3 fanfiction_model.py -v wTv svr -k linear -C 200

Regression Results for Support Vector Regression on the Development set:

Mean Squared Error: 442667.232
Root Mean Squared Error: 665.332
```
