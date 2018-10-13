# Disaster Response Pipeline Project
In this project, I analyzed the data provided by Figure Eight containing the messages in three different categories. Goal of the project is to build a model for an API that classifies disaster messages across 36 different categories.

### Overview of the Dataset
- Messages are either related to a distaster or not.Out of 25825 messages, 19688 of messages are related to disaster and 6137 of messages are not. 
![Related Messages](images/Related_Messages.png?raw=true)
- Each message belong to one of the three genres(Direct, Social, News).
![Messages Genres](images/Message_Genres.png?raw=true)
- There are 36 target features. By analyzing the dataset it can be seen that if a message is related to a disaster,only then the other 35 features might have a value 1, otherwise 0. Distribution of the target variables is shown below. 
![Target Features](images/Target_Features.png?raw=true?raw=true)

### Preprocessing
Preprocessing is done in ```data/process_data.py``` file containing an ETL pipeline. 
- Data is read from the csv files ```data/disaster_messages.csv``` and ```data/disaster_categories.csv```.
- Both the messages and the categories datasets are merged.
- Merged data is cleaned. 
    - Duplicated mesages are removed. 
    - Non-English messages are removed. 
- Cleaned data is stored in ```data/DisasterResponse.db```.

### Machine Learning Pipeline
ML pipeline is implemented in ```models/train_classifier.py```.
- It loads the data from ```data/DisasterResponse.db```.
- Data is split into trainging and testing sets. 
- A function ```tokenize()``` is implemented to clean the messages data and tokenize it for tf-idfcalculations. 
- Pipelines are implemented for text and machine learning processing. 
- Parameter selection is based on GridSearchCV.
- Trained classifier is stored in ```models/classifier.p```.

### Flask App
Flask app is implemented in the ```app``` folder. 
- Main page gives an overview of the dat as shown in the images above. 
- Main page allows the user to write a message and choose a genre of the message.
![Query Interface](images/Query_Interface.png?raw=true?raw=true?raw=true)
- Output for the given message is shown below. It categorizes the message into related categories. 
![Results Example](images/Results.png?raw=true)

## Instructions:
[Optional] 1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.p`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/

