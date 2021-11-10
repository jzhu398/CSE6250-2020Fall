from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

import os
from glob import glob
from itertools import chain, groupby
import params
from sklearn.model_selection import train_test_split

def preprocess_metadata(data_folder=params.DATA_FOLDER, metadata_file=params.INDICES_FILE):

	spark = SparkSession.builder.master("local").appName("metadata preprossing") \
	                            .config("spark.sql.debug.maxToStringFields","100")   \
	                            .getOrCreate() 
	
	schema = StructType([
		StructField('image_index', StringType(), False),
		StructField('finding_labels', StringType(), False),
		StructField('follow_up_number', IntegerType(), False),
		StructField('patient_id', IntegerType(), False),
		StructField('patient_age', IntegerType(), False),
		StructField('patient_gender', StringType(), False),
		StructField('view_position', StringType(), False),
		StructField('image_width', IntegerType(), False),
		StructField('image_height', IntegerType(), False),
		StructField('spacing_x', FloatType(), False),
		StructField('spacing_y', FloatType(), False),
	])
	
	df = spark.read.csv(os.path.join(data_folder, metadata_file), schema=schema)

	image_schema = StructType([
		StructField('image_index', StringType(), False),
		StructField('path', StringType(), False),
	])

	image_files = [{"image_index" : os.path.basename(x), "path" : x} for x in glob(os.path.join(data_folder, 'images_*', 'images', '*.png'))]
	df_image = spark.createDataFrame(data=image_files, schema=image_schema)
	
	df = df.join(df_image, ['image_index'])
	df = df.withColumn('labels', F.split(df.finding_labels, '\\|'))
	
	labels_collect = [row.labels for row in df.collect()]
	labels = list(chain(*labels_collect))
	freqs = [(len(list(g)), k) for k, g in groupby(sorted(labels))]
	labels_filtered = [label for count, label in freqs if count >= params.MIN_CASES and label != 'No Finding']
	
	print("Raw metadata statistics:")
	for idx, c in enumerate((reversed(sorted(freqs)))):
		print(f'{idx+1}. {c[1]} : {c[0]}')
	print("#" * 60)

	print("These labels are selected based on minimal case threshold :")
	for idx, label in enumerate(sorted(labels_filtered)):
		print(f'{idx+1}. {label}')
	print("#" * 60)
	
	for label in sorted(labels_filtered):
		df = df.withColumn(label, F.when(F.array_contains(df.labels, label), 1).otherwise(0))
	
	metadata = df.toPandas()
	#df.write.mode('overwrite').csv('processed_data')
	return metadata, labels_filtered

def stratify_train_test_split(metadata): 
	'''
	Create a stratified train/test dataset
	'''
	metadata['sample_weight'] = metadata.apply(lambda row : sum(row[13:]), axis=1)
	metadata['sample_weight'] = metadata['sample_weight'] /  metadata['sample_weight'].sum() + 4e-2

	metadata = metadata.sample(params.SAMPLE_SIZE, weights=metadata['sample_weight'].values)
	stratify = metadata['finding_labels'].map(lambda x: x[:4])

	train_df, test = train_test_split(metadata,test_size=0.2, random_state=2020, stratify=stratify)
	stratify_update = train_df['finding_labels'].map(lambda x: x[:4])
	train, valid = train_test_split(train_df,test_size=0.25, random_state=1000, stratify=stratify_update)
	return train, valid, test

if __name__ == '__main__':
	metadata, labels = preprocess_metadata()
	train, valid, test = stratify_train_test_split(metadata)