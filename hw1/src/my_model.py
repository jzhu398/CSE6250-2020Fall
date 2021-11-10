import utils
import pandas as pd
from sklearn.metrics import *
#from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_svmlight_file
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import KFold, ShuffleSplit
from numpy import mean
#Note: You can reuse code that you wrote in etl.py and models.py and cross.py over here. It might help.
# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT
RANDOM_STATE = 545510477
'''
You may generate your own features over here.
Note that for the test data, all events are already filtered such that they fall in the observation window of their respective patients. Thus, if you were to generate features similar to those you constructed in code/etl.py for the test data, all you have to do is aggregate events for each patient.
IMPORTANT: Store your test data features in a file called "test_features.txt" where each line has the
patient_id followed by a space and the corresponding feature in sparse format.
Eg of a line:
60 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514
Here, 60 is the patient id and 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514 is the feature for the patient with id 60.

Save the file as "test_features.txt" and save it inside the folder deliverables

input:
output: X_train,Y_train,X_test
'''
def my_features(filtered_events,feature_map):
	#TODO: complete this

    idx_event = pd.merge(filtered_events,feature_map, on = 'event_id')
    #Drop Nan value
    idx_event.dropna(subset = ["value"], inplace=True)
    #Rename columns
    idx_event = idx_event[['patient_id','idx','value']].rename(columns = {'idx':'feature_id','value':'feature_value'})
    #Split events into two groups - one is for "sum", the other is for "count"
    diagdrug_sum =idx_event[idx_event['feature_id'] <= 2679]
    lab_count = idx_event[idx_event['feature_id'] > 2679]
    #Calculate sum
    diagdrug = diagdrug_sum.groupby(['patient_id','feature_id']).agg('sum').reset_index()
    #Calculate max
    sum_max = diagdrug.groupby(['feature_id']).agg('max').reset_index()
    #Join sum and max columns
    diagdrug_m = pd.merge(diagdrug, sum_max, on = 'feature_id')
    #Normalize the values obtained above using min-max normalization(the min value will be 0 in all scenarios)
    diagdrug_m['feature_value'] = diagdrug_m['feature_value_x']/diagdrug_m['feature_value_y']
    diagdrug_m = diagdrug_m[['patient_id_x', 'feature_id', 'feature_value']].rename(columns = {'patient_id_x': 'patient_id'})    
    #Calculate count
    lab = lab_count.groupby(['patient_id','feature_id']).agg('count').reset_index()
    #Calculate max
    count_max = lab.groupby(['feature_id']).agg('max').reset_index()
    #Join count and max columns
    lab_m = pd.merge(lab, count_max, on = 'feature_id')
    #Normalize the values obtained above using min-max normalization(the min value will be 0 in all scenarios)
    lab_m['feature_value'] = lab_m['feature_value_x']/lab_m['feature_value_y']
    lab_m = lab_m[['patient_id_x', 'feature_id', 'feature_value']].rename(columns = {'patient_id_x': 'patient_id'})    

    aggregated_events = pd.concat([lab_m,diagdrug_m]).reset_index(drop = True)

    #Patient_id and value is array of tuples(feature_id, feature_value)
    patient_features = aggregated_events.groupby('patient_id')[['feature_id','feature_value']].apply(lambda x: [tuple(x) for x in x.values]).to_dict()

    deliverable3 = open('../deliverables/test_features.txt','w')  
    deliverable4 = open('../deliverables/test_features.train','w')

    count = 0
    for patient_id in sorted(patient_features):
        p1 = "%d" %(patient_id)
        p2 = "%d" %(0)
        for feature in sorted(patient_features[patient_id]):
            string_p = "%d:%.6f" %(feature[0], feature[1])
            p1 = p1 + " " + string_p + " "
            p2 = p2 + " " + string_p + " "
        deliverable3.write(p1 + "\n")
        deliverable4.write(p2 + "\n")
        count = count + 1
        if count > 632:
            break  

'''
You can use any model you wish.

input: X_train, Y_train, X_test
output: Y_pred
'''
def my_classifier_predictions(X_train,Y_train,X_test):
	#TODO: complete this
    #RandomForest
    # model_rf = RandomForestClassifier(n_estimators=2000, max_depth=5, random_state=RANDOM_STATE).fit(X_train,Y_train)
    # Y_pred = model_rf.predict(X_test)
    # return Y_pred    
    
#    #DecisionTree with AdaBoost
#    model_dt = DecisionTreeRegressor(max_depth = 5, random_state=RANDOM_STATE)
#    model_ada = AdaBoostRegressor(model_dt, n_estimators= 2000, learning_rate=0.01, random_state=RANDOM_STATE).fit(X_train,Y_train)
#    Y_pred = model_ada.predict(X_test)
#    return Y_pred    

#    #GBoost
#    model_gb = GradientBoostingRegressor(learning_rate=0.01, max_depth = 5, n_estimators= 2000, random_state=RANDOM_STATE).fit(X_train,Y_train)
#    Y_pred = model_gb.predict(X_test)
#    return Y_pred
    
    #voting
    reg1 = GradientBoostingRegressor(learning_rate=0.01, max_depth = 5, n_estimators= 2000, random_state=RANDOM_STATE)
    reg2 = ExtraTreesRegressor(n_estimators=2000, max_depth=5, random_state=RANDOM_STATE)
    reg3 = RidgeRegressor(random_state=RANDOM_STATE)
    reg4 = MLPRegressor(random_state=RANDOM_STATE)
    model_voting = VotingRegressor(voting='soft', estimators=[('gb', reg1),('et', reg2),('rc',reg3),('mlp',reg4)], weights=[3,1,1,3]).fit(X_train,Y_train)
    Y_pred = model_voting.predict(X_test)
    return Y_pred

def main():
    test_events = pd.read_csv('../data/test/events.csv') 
    test_feature_map = pd.read_csv('../data/test/event_feature_map.csv') 
    my_features(test_events,test_feature_map)
    X_train, Y_train = utils.get_data_from_svmlight('../deliverables/features_svmlight.train') 
    X_test, Y_test = utils.get_data_from_svmlight("../deliverables/test_features.train")
    Y_pred = my_classifier_predictions(X_train,Y_train, X_test)
    utils.generate_submission("../deliverables/test_features.txt",Y_pred) 
	#The above function will generate a csv file of (patient_id,predicted label) and will be saved as "my_predictions.csv" in the deliverables folder.
if __name__ == "__main__":
    main()

	