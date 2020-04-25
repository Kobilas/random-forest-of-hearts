from csv import reader
from random import seed
from random import randrange
from math import sqrt, floor

# loads csv to list of lists
# prepares it using str_flt_col_to_float for all columns except class value column
# and str_int_col_to_int for class value column
def load_prep_csv(filename):
    dt = []
    with open(fname, 'r') as f:
        csv_rdr = reader(f)
        next(csv_rdr, None) # skip header
        for row in csv_rdr:
            dt.append(row)
    # for each attribute in dataset
    # len(dt[0]) - 1 is length of first list of lists, minus the class variable
    for i in range(0, len(dt[0]) - 1):
        str_flt_col_to_float(dt, i)
    str_int_col_to_int(dt, len(dt[0]) - 1) # converts class value column to enumeration of it in terms of int
    return dt

# convert string in column in dataset (list of lists) to float
# column is the idx of the column in the list of list
# may be more efficient to convert only the columns we know need to be converted to float
def str_flt_col_to_float(data, column):
    for row in data:
        row[column] = float(row[column].strip()) # strip whitespace and convert to float

# converts string values in class column to enumeration of integers
def str_int_col_to_int(data, column):
    class_column = [row[column] for row in data] # create list representing column of class values
    unique_classes = set(class_column) # get all unique values of class column
    enum_classes = {}
    for i, x in enumerate(unique_classes):
        enum_classes[x] = i
    for row in data:
        row[column] = enum_classes[row[column]]

# partitions data into num_partitions and returns it as a list of list of lists
def partition_data(data, num_partitions):
    dt_cp = data # copy of data so that records may be randomly indexed and popped
    max_len_part = floor(len(data) / num_partitions) # max length of each partition
    dt_parts = [[] for empty_part in range(num_partitions)]
    for part_i in range(num_partitions):
        while len(dt_parts[part_i]) < max_len_part:
            rand_idx = randrange(len(dt_cp)) # get random record from data
            dt_parts[part_i].append(dt_cp.pop(rand_idx)) # append to current partition we are creating
    return dt_parts

# calculates and returns accuracy as a percentage float
def get_accuracy(actual, prediction):
    num_correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return (num_correct / float(len(actual))) * 100.0

# evaluting random forest using k-fold cross-validation where k is num_partitions
# can add repetition using different seeds to allow for "larger" dataset in the future
def evaluate_en_masse(data, num_partitions):
    dt_parts = partition_data(data, num_partitions)
    mass_scores = []
    for i, part in enumerate(dt_parts):
        dt_train = dt_parts # right now dt_parts is a list (dataset) of lists (parts) of lists (rows)
        dt_train.remove(part) # remove fold that will be used as test dataset
        dt_train = sum(dt_train, []) # collapse remaining folds into a list (dataset) of lists (rows)
        dt_test = []
        for row in part:
            row_cp = row
            row_cp[-1] = None # hide result in test set from algorithm
            dt_test.append(row_cp)
        print("dt_train #" + str(i) + ": " + str(dt_train))
        print("=============================================================================================")
        print("dt test #" + str(i) + ": " + str(dt_test))
        print("\n\n")

seed(666) # set random seed, 666 for my ucid mk666
# load data to list and prep it
fname = "heart.csv"
data = load_prep_csv(fname)
evaluate_en_masse(data, 3)