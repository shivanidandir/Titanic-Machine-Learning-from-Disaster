# Titanic-Machine-Learning-from-Disaster
Kaggle competition

Created a machine learning application in python to predict the survival rate of passenger in the Titanic shipwreck. Used Random Forest Classifier for prediction and done feature engineering on the dataset. Got an accuracy of 79.425% which positioned in top 17% of the competitors.


**DESCRIPTION**

**DATA WRANGLING AND PREPROCESSING**

Firstly I loaded packages and then loaded data into train and test variables. Now, to do operations I combined the data into Titanic variable (So that I don't have to do operation twice).I also created an index for each train and test so that I can separate them out later into their respective train and test. Use the .info() method to get a description of the columns in the dataframe.The code takes a value like “Braund, Mr. Owen Harris” from the Name column and extracts “Mr”: and created Data dictionary for the same and normalized it.
Filled the missing value with the median. For 'Cabin'(70%) filled 'u' for unkown values, for 'Embarked' filled with the most frequent value and for 'Fare' only one missing value was there so filled it with median.

**FEATURE ENGINEERING**

* Family size is an important parameter because larger the family less are the survival chances so calculated it with the siblings/spouses column and parent child coluumn added 1 because of the individual.
* Cabin is potentially relevant since it is possible that some cabins were closer to the life boats and thus those that were closer to them may have had a greater chance at securing a spot.
* Created dummy variable for categorical values to perform algorithmic manipulation.
* Split dataset with the index values.

**MODELING**

* Used Random Forest classifier because it gave higher accuracy 84% than logistic Regression -82%.
* Used GridSearchCV to pass in a range of parameters and have it return the best score and the associated parameters.
* Converted the result into csv file with passengerId and Survived as column names 
