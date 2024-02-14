import forest_to_csv
import csv_to_c # forest_to_binary, create_array_for_c, get_forest_structure
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sys
pd.options.mode.chained_assignment = None  # default='warn'

OUTPUT_NAME = "temp_forest"
C_DATA_OUTPUT = "forest_data"
BINARY_OUTPUT = "temp.bin"

def forest_to_c(rf: RandomForestClassifier,
                output_name=OUTPUT_NAME, c_data_output=C_DATA_OUTPUT, binary_output=BINARY_OUTPUT):
    forest_to_csv.forest_struct_to_csv(rf, output_name)

    forest_bytes = csv_to_c.forest_to_binary(''.join((output_name, '.csv')),
                                    binary_output, write_to_file=False, metadata=''.join((output_name, '.meta')))
    csv_name = ''.join((output_name, '.csv'))
    meta_name = ''.join((output_name, '.meta'))
    csv_to_c.create_array_for_c(forest_bytes, csv_to_c.get_forest_structure(csv_name), meta_name, c_data_output)
