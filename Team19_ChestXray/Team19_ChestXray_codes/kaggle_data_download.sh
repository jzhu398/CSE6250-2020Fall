export KAGGLE_USERNAME=$1                                                                                                                               
export KAGGLE_KEY=$2

kaggle datasets download -d nih-chest-xrays/data -p /data
unzip data/data.zip -d /data
