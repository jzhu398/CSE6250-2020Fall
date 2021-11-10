-- ***************************************************************************
-- TASK
-- Aggregate events into features of patient and generate training, testing data for mortality prediction.
-- Steps have been provided to guide you.
-- You can include as many intermediate steps as required to complete the calculations.
-- ***************************************************************************

-- ***************************************************************************
-- TESTS
-- To test, please change the LOAD path for events and mortality to ../../test/events.csv and ../../test/mortality.csv
-- 6 tests have been provided to test all the subparts in this exercise.
-- Manually compare the output of each test against the csv's in test/expected folder.
-- ***************************************************************************

-- register a python UDF for converting data into SVMLight format
REGISTER utils.py USING jython AS utils;

-- load events file
events = LOAD '../data/events.csv' USING PigStorage(',') AS (patientid:int, eventid:chararray, eventdesc:chararray, timestamp:chararray, value:float);

-- select required columns from events
events = FOREACH events GENERATE patientid, eventid, ToDate(timestamp, 'yyyy-MM-dd') AS etimestamp, value;

-- load mortality file
mortality = LOAD '../data/mortality.csv' USING PigStorage(',') as (patientid:int, timestamp:chararray, label:int);

mortality = FOREACH mortality GENERATE patientid, ToDate(timestamp, 'yyyy-MM-dd') AS mtimestamp, label;

--To display the relation, use the dump command e.g. DUMP mortality;

-- ***************************************************************************
-- Compute the index dates for dead and alive patients
-- ***************************************************************************
eventswithmort = JOIN events BY patientid LEFT OUTER, mortality BY patientid;
-- perform join of events and mortality by patientid;
eventswithmort_right_label = FOREACH eventswithmort GENERATE events::patientid AS patientid, events::eventid AS eventid, events::value AS value, events::etimestamp as etimestamp, (mortality::label IS NULL ? 0:1) AS label;
-- change the label from (null, 1) to (0,1)

filtered_dead = FILTER eventswithmort BY (mortality::label == 1);
-- filter for dead patient
deadevents =  FOREACH filtered_dead GENERATE events::patientid AS patientid,events::eventid AS eventid,events::value AS value,mortality::label AS label, DaysBetween(SubtractDuration(mortality::mtimestamp,'P30D'), events::etimestamp) AS time_difference;
-- detect the events of dead patients and create it of the form (patientid, eventid, value, label, time_difference) where time_difference is the days between index date and each event timestamp

filter_alive = FILTER eventswithmort_right_label BY (label == 0);
-- filter for alive patient
group_alive = GROUP filter_alive BY patientid;
-- group alive patient by patientid
max_date_alive = FOREACH group_alive GENERATE group AS patientid, MAX(filter_alive.etimestamp) AS indexdate;
-- find the index date, which is max date for alive patient
whole_table = JOIN filter_alive BY patientid, max_date_alive BY patientid;
aliveevents = FOREACH whole_table GENERATE filter_alive::patientid AS patientid, filter_alive::eventid AS eventid, filter_alive::value AS value, filter_alive::label AS label, DaysBetween(max_date_alive::indexdate,filter_alive::etimestamp) AS time_difference;
-- detect the events of alive patients and create it of the form (patientid, eventid, value, label, time_difference) where time_difference is the days between index date and each event timestamp

--TEST-1
deadevents = ORDER deadevents BY patientid, eventid;
aliveevents = ORDER aliveevents BY patientid, eventid;
STORE aliveevents INTO 'aliveevents' USING PigStorage(',');
STORE deadevents INTO 'deadevents' USING PigStorage(',');

-- ***************************************************************************
-- Filter events within the observation window and remove events with missing values
-- ***************************************************************************
alive_dead_events = UNION aliveevents, deadevents;
-- join the table together to collect all info
nonnull_events = FILTER alive_dead_events BY value IS NOT NULL;
-- remove all null in dataset
filtered = FILTER nonnull_events BY (time_difference<=2000L) AND (time_difference>=0L);
-- contains only events for all patients within the observation window of 2000 days and is of the form (patientid, eventid, value, label, time_difference)

--TEST-2
filteredgrpd = GROUP filtered BY 1;
filtered = FOREACH filteredgrpd GENERATE FLATTEN(filtered);
filtered = ORDER filtered BY patientid, eventid,time_difference;
STORE filtered INTO 'filtered' USING PigStorage(',');

-- ***************************************************************************
-- Aggregate events to create features
-- ***************************************************************************
unique_id = GROUP filtered BY (patientid,eventid);
-- find the unique patientid and eventid
featureswithid = FOREACH unique_id GENERATE group.$0 AS patientid,group.$1 AS eventid,COUNT(filtered.value) AS featurevalue;
-- for group of (patientid, eventid), count the number of  events occurred for the patient and create relation of the form (patientid, eventid, featurevalue)

--TEST-3
featureswithid = ORDER featureswithid BY patientid, eventid;
STORE featureswithid INTO 'features_aggregate' USING PigStorage(',');

-- ***************************************************************************
-- Generate feature mapping
-- ***************************************************************************
data_collect = FOREACH featureswithid GENERATE eventid;
-- only include eventid in data_collect
distinct_event_id = DISTINCT data_collect;
-- find the distinct value for eventid
distinct_event_id = ORDER distinct_event_id BY eventid;
-- order by eventid by default(ASC)
rank_event_id = RANK distinct_event_id;
-- rank eventid
all_features = FOREACH rank_event_id GENERATE $0 AS idx, $1 AS eventid;
-- create (idx, eventid). Rank should start from 0.

-- store the features as an output file
STORE all_features INTO 'features' using PigStorage(' ');

join_table = JOIN featureswithid BY eventid, all_features BY eventid;
-- perform join of featureswithid and all_features by eventid and replace eventid with idx
features = FOREACH join_table GENERATE featureswithid::patientid AS patientid, all_features::idx AS idx, featureswithid::featurevalue AS featurevalue;
-- It is of the form (patientid, idx, featurevalue)

--TEST-4
features = ORDER features BY patientid, idx;
STORE features INTO 'features_map' USING PigStorage(',');

-- ***************************************************************************
-- Normalize the values using min-max normalization
-- Use DOUBLE precision
-- ***************************************************************************
group_id_max = GROUP features BY idx;
-- group events by idx
maxvalues = FOREACH group_id_max GENERATE group AS idx, MAX(features.featurevalue) AS maxvalue;
-- compute the maximum feature value in each group. It is of the form (idx, maxvalue)

normalized = JOIN features BY idx, maxvalues BY idx;
-- join features and maxvalues by idx

features = FOREACH normalized GENERATE features::patientid AS patientid, features::idx AS idx, ((double)features::featurevalue/(double)maxvalues::maxvalue) AS normalizedfeaturevalue;
-- compute the final set of normalized features of the form (patientid, idx, normalizedfeaturevalue)

--TEST-5
features = ORDER features BY patientid, idx;
STORE features INTO 'features_normalized' USING PigStorage(',');

-- ***************************************************************************
-- Generate features in svmlight format
-- features is of the form (patientid, idx, normalizedfeaturevalue) and is the output of the previous step
-- e.g.  1,1,1.0
--  	 1,3,0.8
--	     2,1,0.5
--       3,3,1.0
-- ***************************************************************************

grpd = GROUP features BY patientid;
grpd_order = ORDER grpd BY $0;
features = FOREACH grpd_order
{
    sorted = ORDER features BY idx;
    generate group as patientid, utils.bag_to_svmlight(sorted) as sparsefeature;
}

-- ***************************************************************************
-- Split into train and test set
-- labels is of the form (patientid, label) and contains all patientids followed by label of 1 for dead and 0 for alive
-- e.g. 1,1
--	2,0
--      3,1
-- ***************************************************************************

labels = FOREACH filtered GENERATE patientid, label;
-- collect patientid and label in labels dataset
labels = DISTINCT labels;
-- create it of the form (patientid, label) for dead and alive patients

--Generate sparsefeature vector relation
samples = JOIN features BY patientid, labels BY patientid;
samples = DISTINCT samples PARALLEL 1;
samples = ORDER samples BY $0;
samples = FOREACH samples GENERATE $3 AS label, $1 AS sparsefeature;

--TEST-6
STORE samples INTO 'samples' USING PigStorage(' ');

-- randomly split data for training and testing
DEFINE rand_gen RANDOM('6505');
samples = FOREACH samples GENERATE rand_gen() as assignmentkey, *;
SPLIT samples INTO testing IF assignmentkey <= 0.20, training OTHERWISE;
training = FOREACH training GENERATE $1..;
testing = FOREACH testing GENERATE $1..;

-- save training and tesing data
STORE testing INTO 'testing' USING PigStorage(' ');
STORE training INTO 'training' USING PigStorage(' ');
