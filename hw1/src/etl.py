import utils
import pandas as pd

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    
    '''
    TODO: This function needs to be completed.
    Read the events.csv, mortality_events.csv and event_feature_map.csv files into events, mortality and feature_map.
    
    Return events, mortality and feature_map
    '''

    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    events = pd.read_csv(filepath+'events.csv')
    
    #Columns in mortality_event.csv - patient_id,timestamp,label
    mortality = pd.read_csv(filepath+'mortality_events.csv')

    #Columns in event_feature_map.csv - idx,event_id
    feature_map = pd.read_csv(filepath+'event_feature_map.csv')

    return events, mortality, feature_map


def calculate_index_date(events, mortality, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 a

    Suggested steps:
    1. Create list of patients alive ( mortality_events.csv only contains information about patients deceased)
    2. Split events into two groups based on whether the patient is alive or deceased
    3. Calculate index date for each patient
    
    IMPORTANT:
    Save indx_date to a csv file in the deliverables folder named as etl_index_dates.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, indx_date.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)

    Return indx_date
    '''
    #Create dataset for patients alive
    live = events[~events['patient_id'].isin(mortality['patient_id'])]
    #Calculate index date for each patient
    live = live.groupby(['patient_id'])['timestamp'].max().reset_index().rename(columns={'timestamp':'indx_date'})
    
    dead = mortality
    #Change data type
    dead.loc[:,'timestamp'] = pd.to_datetime(dead.loc[:,'timestamp'])
    #Index date is 30 days prior to the death date
    dead['indx_date'] = (dead['timestamp'] - pd.Timedelta(days=30)).dt.date
    dead_date = dead[['patient_id', 'indx_date']]
    #Union alive and dead index date
    indx_date = pd.concat([live, dead_date]).reset_index(drop=True)
    indx_date.to_csv(deliverables_path+'etl_index_dates.csv',columns = ['patient_id','indx_date'],index = False)
    return indx_date


def filter_events(events, indx_date, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 b

    Suggested steps:
    1. Join indx_date with events on patient_id
    2. Filter events occuring in the observation window(IndexDate-2000 to IndexDate)
    
    
    IMPORTANT:
    Save filtered_events to a csv file in the deliverables folder named as etl_filtered_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)

    Return filtered_events
    '''
    #Join index date to events dataset
    event_all = events.merge(indx_date, on='patient_id', how='left')
    #Change data type from str to datestime
    event_all.loc[:,'timestamp'] = pd.to_datetime(event_all.loc[:,'timestamp'])
    event_all.loc[:,'indx_date'] = pd.to_datetime(event_all.loc[:,'indx_date'])
    #Calculate observation window(IndexDate-2000 to IndexDate)
    event_all['diff'] = event_all['indx_date'] - event_all['timestamp']
    #Create filters
    filtered_events = event_all[(event_all['diff'] >= pd.Timedelta(0,'D')) & (event_all['diff'] <= pd.Timedelta(2000,'D'))].reset_index(drop=True)
    filtered_events = filtered_events[['patient_id','event_id','value']]
    filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv',index = False)
    return filtered_events


def aggregate_events(filtered_events_df, mortality_df,feature_map_df, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 c

    Suggested steps:
    1. Replace event_id's with index available in event_feature_map.csv
    2. Remove events with n/a values
    3. Aggregate events using sum and count to calculate feature value
    4. Normalize the values obtained above using min-max normalization(the min value will be 0 in all scenarios)
    
    
    IMPORTANT:
    Save aggregated_events to a csv file in the deliverables folder named as etl_aggregated_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header .
    For example if you are using Pandas, you could write: 
        aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    Return aggregated_events
    '''
    #Join event_feature_map.csv and Replace event_id's index
    idx_event = pd.merge(filtered_events_df,feature_map_df, on = 'event_id')
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
    aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv',index = False)
    return aggregated_events

def create_features(events, mortality, feature_map):
    
    deliverables_path = '../deliverables/'

    #Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)

    #Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  deliverables_path)
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)

    '''
    TODO: Complete the code below by creating two dictionaries - 
    1. patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
    2. mortality : Key - patient_id and value is mortality label
    '''
    #Patient_id and value is array of tuples(feature_id, feature_value)
    patient_features = aggregated_events.groupby('patient_id')[['feature_id','feature_value']].apply(lambda x: [tuple(x) for x in x.values]).to_dict()
    #patient_id and value is mortality label
    live_df = events[~events['patient_id'].isin(mortality['patient_id'])]
    live_df['is_dead'] = 0
    dead_df = events[events['patient_id'].isin(mortality['patient_id'])]
    dead_df['is_dead'] = 1
    total_df = pd.concat([live_df, dead_df]).reset_index(drop = True).drop(['event_id', 'event_description', 'timestamp', 'value'], axis=1) 
    
    mortality = pd.Series(total_df.is_dead.values, index = total_df.patient_id).to_dict()

    return patient_features, mortality

def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    
    '''
    TODO: This function needs to be completed

    Refer to instructions in Q3 d

    Create two files:
    1. op_file - which saves the features in svmlight format. (See instructions in Q3d for detailed explanation)
    2. op_deliverable - which saves the features in following format:
       patient_id1 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
       patient_id2 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...  
    
    Note: Please make sure the features are ordered in ascending order, and patients are stored in ascending order as well.     
    '''
    deliverable1 = open(op_file, 'wb')
    deliverable2 = open(op_deliverable, 'wb')
    for patient_id in sorted(patient_features):
        string_p = ""
        for feature in sorted(patient_features[patient_id]):
            string_p += " " + str(int(feature[0])) + ":" + format(feature[1], '.6f')
        svmlight = str(mortality[patient_id]) + string_p + " \n"
        deliverable1.write(bytes((svmlight),'UTF-8')); #Use 'UTF-8'
        deliverable2.write(bytes((str(int(patient_id)) + " " + svmlight),'UTF-8'));

def main():
    train_path = '../data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')

if __name__ == "__main__":
    main()