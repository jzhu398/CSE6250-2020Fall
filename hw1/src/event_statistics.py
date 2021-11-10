import time
import pandas as pd
import numpy as np

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    '''
    TODO : This function needs to be completed.
    Read the events.csv and mortality_events.csv files. 
    Variables returned from this function are passed as input to the metric functions.
    '''
    #Read two datasets using pandas
    events = pd.read_csv(filepath + 'events.csv')
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    return events, mortality

def event_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the event count metrics.
    Event count is defined as the number of events recorded for a given patient.
    '''
    #Create dataset for patients alive
    live = events[~events['patient_id'].isin(mortality['patient_id'])]
    #Count total for each individual patients alive
    live_count = live[['patient_id']].value_counts(ascending=True)
    #Create dataset for patients dead
    dead = events[events['patient_id'].isin(mortality['patient_id'])]
    #Count total for each individual patients dead
    dead_count = dead[['patient_id']].value_counts(ascending=True)
    #Event count metrics
    avg_dead_event_count = dead_count.mean()
    max_dead_event_count = dead_count.max()
    min_dead_event_count = dead_count.min()
    avg_alive_event_count = live_count.mean()
    max_alive_event_count = live_count.max()
    min_alive_event_count = live_count.min()

    return min_dead_event_count, max_dead_event_count, avg_dead_event_count, min_alive_event_count, max_alive_event_count, avg_alive_event_count

def encounter_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the encounter count metrics.
    Encounter count is defined as the count of unique dates on which a given patient visited the ICU. 
    '''
    live_unique = events[~events['patient_id'].isin(mortality['patient_id'])]
    dead_unique = events[events['patient_id'].isin(mortality['patient_id'])]
    #select unique date for each individual patients alive
    encounter_live = live_unique.groupby(['patient_id'])['timestamp'].nunique()
    #select unique date for each individual patients dead
    encounter_dead = dead_unique.groupby(['patient_id'])['timestamp'].nunique()
    #Encounter count metrics
    avg_dead_encounter_count = encounter_dead.mean()
    max_dead_encounter_count = encounter_dead.max()
    min_dead_encounter_count = encounter_dead.min()
    avg_alive_encounter_count = encounter_live.mean()
    max_alive_encounter_count = encounter_live.max()
    min_alive_encounter_count = encounter_live.min()

    return min_dead_encounter_count, max_dead_encounter_count, avg_dead_encounter_count, min_alive_encounter_count, max_alive_encounter_count, avg_alive_encounter_count

def record_length_metrics(events, mortality):
    '''
    TODO: Implement this function to return the record length metrics.
    Record length is the duration between the first event and the last event for a given patient. 
    '''
    
    live_date = events[~events['patient_id'].isin(mortality['patient_id'])]
    dead_date = events[events['patient_id'].isin(mortality['patient_id'])]
    #Change data type
    live_date.loc[:,'timestamp'] = pd.to_datetime(live_date.loc[:,'timestamp'])
    dead_date.loc[:,'timestamp'] = pd.to_datetime(dead_date.loc[:,'timestamp'])
    #Duration (in number of days) between the first event and last event for a given patient
    LOS_live = live_date.groupby(['patient_id'])['timestamp'].agg(np.ptp).dt.days
    LOS_dead = dead_date.groupby(['patient_id'])['timestamp'].agg(np.ptp).dt.days
    #Record length metrics
    avg_dead_rec_len = LOS_dead.mean()
    max_dead_rec_len = LOS_dead.max()
    min_dead_rec_len = LOS_dead.min()
    avg_alive_rec_len = LOS_live.mean()
    max_alive_rec_len = LOS_live.max()
    min_alive_rec_len = LOS_live.min()

    return min_dead_rec_len, max_dead_rec_len, avg_dead_rec_len, min_alive_rec_len, max_alive_rec_len, avg_alive_rec_len

def main():
    '''
    DO NOT MODIFY THIS FUNCTION.
    '''
    # You may change the following path variable in coding but switch it back when submission.
    train_path = '../data/train/'

    # DO NOT CHANGE ANYTHING BELOW THIS ----------------------------
    events, mortality = read_csv(train_path)

    #Compute the event count metrics
    start_time = time.time()
    event_count = event_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute event count metrics: " + str(end_time - start_time) + "s"))
    print(event_count)

    #Compute the encounter count metrics
    start_time = time.time()
    encounter_count = encounter_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute encounter count metrics: " + str(end_time - start_time) + "s"))
    print(encounter_count)

    #Compute record length metrics
    start_time = time.time()
    record_length = record_length_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute record length metrics: " + str(end_time - start_time) + "s"))
    print(record_length)
    
if __name__ == "__main__":
    main()
