import models_partc
from sklearn.model_selection import KFold, ShuffleSplit
from numpy import mean
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *

import utils

# USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

# USE THIS RANDOM STATE FOR ALL OF YOUR CROSS VALIDATION TESTS, OR THE TESTS WILL NEVER PASS
RANDOM_STATE = 545510477

#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_kfold(X,Y,k=5):
	#TODO:First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the folds
    kf = KFold(n_splits=k, random_state = RANDOM_STATE)
    lr_kf = LogisticRegression()
    accracy_list =[]
    auc_list =[]
    for train_index, test_index in kf.split(X):
        lr_kf_pred = lr_kf.fit(X[train_index],Y[train_index])
        acc = accuracy_score(lr_kf_pred.predict(X[test_index]),Y[test_index])
        accracy_list.append(acc)
        
        auc = roc_auc_score(lr_kf_pred.predict(X[test_index]),Y[test_index])
        auc_list.append(auc)
        
    mean_accuracy = mean(accracy_list)
    mean_auc = mean(auc_list)
    return mean_accuracy, mean_auc


#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_randomisedCV(X,Y,iterNo=5,test_percent=0.2):
	#TODO: First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the iterations
    kf = ShuffleSplit(n_splits = iterNo, random_state = RANDOM_STATE, test_size = test_percent)
    lr_kf = LogisticRegression()
    accracy_list =[]
    auc_list =[]
    for train_index, test_index in kf.split(X):           
        lr_kf_pred = lr_kf.fit(X[train_index],Y[train_index])
        acc = accuracy_score(lr_kf_pred.predict(X[test_index]),Y[test_index])
        accracy_list.append(acc) 
        
        auc = roc_auc_score(lr_kf_pred.predict(X[test_index]),Y[test_index])
        auc_list.append(auc)
    
    mean_accuracy = mean(accracy_list)
    mean_auc = mean(auc_list)
    return mean_accuracy, mean_auc


def main():
	X,Y = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	print("Classifier: Logistic Regression__________")
	acc_k,auc_k = get_acc_auc_kfold(X,Y)
	print(("Average Accuracy in KFold CV: "+str(acc_k)))
	print(("Average AUC in KFold CV: "+str(auc_k)))
	acc_r,auc_r = get_acc_auc_randomisedCV(X,Y)
	print(("Average Accuracy in Randomised CV: "+str(acc_r)))
	print(("Average AUC in Randomised CV: "+str(auc_r)))

if __name__ == "__main__":
	main()

