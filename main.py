from forest_to_csv import forest_struct_to_csv
from csv_to_c import forest_to_binary, create_array_for_c, get_forest_structure
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sys
pd.options.mode.chained_assignment = None  # default='warn'

DATA_SOURCE = "observations-386953.csv"
OUTPUT_NAME = "temp_forest"
C_DATA_OUTPUT = "forest_data"
BINARY_OUTPUT = "temp.bin"

def main(rf: RandomForestClassifier):
    forest_struct_to_csv(rf, OUTPUT_NAME)

    forest_bytes = forest_to_binary(''.join((OUTPUT_NAME, '.csv')), BINARY_OUTPUT, write_to_file=False)
    csv_name = ''.join((OUTPUT_NAME, '.csv'))
    meta_name = ''.join((OUTPUT_NAME, '.meta'))
    create_array_for_c(forest_bytes, get_forest_structure(csv_name), meta_name, C_DATA_OUTPUT)
    return

if __name__ == "__main__":
    main(sys.argv[1])
