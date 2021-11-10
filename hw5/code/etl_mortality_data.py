import os
import numpy as np
import pickle
import pandas as pd


PATH_TRAIN = "/Users/zhujingyao/Downloads/Big_Data_for_Health/hw5/data/mortality/train/"
PATH_VALIDATION = "/Users/zhujingyao/Downloads/Big_Data_for_Health/hw5/data/mortality/validation/"
PATH_TEST = "/Users/zhujingyao/Downloads/Big_Data_for_Health/hw5/data/mortality/test/"
PATH_OUTPUT = "m/Users/zhujingyao/Downloads/Big_Data_for_Health/hw5/data/mortality/processed/"

def convert_icd9(icd9_object):
	"""
	:param icd9_object: ICD-9 code (Pandas/Numpy object).
	:return: extracted main digits of ICD-9 code
	"""
	icd9_str = str(icd9_object)
	# -: Extract the the first 3 or 4 alphanumeric digits prior to the decimal point from a given ICD-9 code.
	# -: Read the homework description carefully.
	converted = None
	if icd9_str[0] == 'E':
		converted = icd9_str[:4]
	else:
		converted = icd9_str[:3]

	#converted = icd9_str

	return converted

def build_codemap(df_icd9, transform):
    lengthofICD9Code = df_icd9['ICD9_CODE'].dropna()
    lengthofICD9Code = lengthofICD9Code.apply(transform)
    lengthofICD9Code = lengthofICD9Code.unique()
    return dict(zip(lengthofICD9Code, np.arange(len(lengthofICD9Code))))

def create_dataset(path, codemap, transform):
	"""
	:param path: path to the directory contains raw files.
	:param codemap: 3-digit ICD-9 code feature map
	:param transform: e.g. convert_icd9
	:return: List(patient IDs), List(labels), Visit sequence data as a List of List of List.
	"""
	# -: 1. Load data from the three csv files
	# -: Loading the mortality file is shown as an example below. Load two other files also.
	df_mortality = pd.read_csv(os.path.join(path, "MORTALITY.csv"))
	df_diagnoses = pd.read_csv(os.path.join(path, "DIAGNOSES_ICD.csv"))
	df_admission = pd.read_csv(os.path.join(path, "ADMISSIONS.csv"))

	# -: 2. Convert diagnosis code in to unique feature ID.
	# -: HINT - use 'transform(convert_icd9)' you implemented and 'codemap'.
	df_diagnoses["ICD9_CODE"] = df_diagnoses["ICD9_CODE"].transform(convert_icd9)
	df_diagnoses["mapped"] = df_diagnoses["ICD9_CODE"].apply(lambda x: x in codemap)
	#print(df_diagnoses.sort_values("mapped").head())
	df_diagnoses = df_diagnoses[df_diagnoses.mapped]
	df_diagnoses["ICD9_CODE"] = df_diagnoses["ICD9_CODE"].apply(lambda x: codemap[x])
	#print(df_diagnoses.sort_values("mapped").head())
	

	# -: 3. Group the diagnosis codes for the same visit.
	codes_by_visit = df_diagnoses.groupby("HADM_ID")["ICD9_CODE"].apply(list)
	codes_by_visit = codes_by_visit.to_dict()

	# -: 4. Group the visits for the same patient.
	df_admission["mapped"] = df_admission["HADM_ID"].apply(lambda x: x in codes_by_visit)
	df_admission = df_admission[df_admission.mapped]
	visits_by_patient = df_admission.sort_values("ADMITTIME").groupby("SUBJECT_ID")["HADM_ID"].apply(list)

	# -: 5. Make a visit sequence dataset as a List of patient Lists of visit Lists
	# -: Visits for each patient must be sorted in chronological order.
	
	# -: 6. Make patient-id List and label List also.
	# -: The order of patients in the three List output must be consistent.

	#print(codes_by_visit.head())
	#print(visits_by_patient.head())

	visits_by_patient = visits_by_patient.to_dict()

	patient_ids = list(visits_by_patient.keys())
	#print(patient_ids)
	#print(patient_ids[:5])

	visitSeq = [visits_by_patient[pid] for pid in patient_ids]	
	seq_data = [[codes_by_visit[visit] for visit in visits] for visits in visitSeq]
	#print(seq_data[:5])
	
	mort_dict = df_mortality.set_index("SUBJECT_ID").sort_values("SUBJECT_ID").to_dict()["MORTALITY"]
	labels = [mort_dict[int(pid)] for pid in patient_ids]
	
	#print(len(labels), len(seq_data), len(patient_ids))
	

	#patient_ids = [0, 1, 2]
	#labels = [1, 0, 1]
	#seq_data = [[[0, 1], [2]], [[1, 3, 4], [2, 5]], [[3], [5]]]
	return patient_ids, labels, seq_data

def main():
	# Build a code map from the train set
	print("Build feature id map")
	df_icd9 = pd.read_csv(os.path.join(PATH_TRAIN, "DIAGNOSES_ICD.csv"), usecols=["ICD9_CODE"])
	codemap = build_codemap(df_icd9, convert_icd9)
	os.makedirs(PATH_OUTPUT, exist_ok=True)
	pickle.dump(codemap, open(os.path.join(PATH_OUTPUT, "mortality.codemap.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Train set
	print("Construct train set")
	train_ids, train_labels, train_seqs = create_dataset(PATH_TRAIN, codemap, convert_icd9)

	pickle.dump(train_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(train_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(train_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Validation set
	print("Construct validation set")
	validation_ids, validation_labels, validation_seqs = create_dataset(PATH_VALIDATION, codemap, convert_icd9)

	pickle.dump(validation_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(validation_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(validation_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Test set
	print("Construct test set")
	test_ids, test_labels, test_seqs = create_dataset(PATH_TEST, codemap, convert_icd9)

	pickle.dump(test_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.test"), 'wb'), pickle.HIGHEST_PROTOCOL)

	print("Complete!")


if __name__ == '__main__':
	main()
