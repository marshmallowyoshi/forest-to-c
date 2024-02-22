"""
Convert a RandomForestClassifier object created using sklearn ensemble to an equivalent model implemented in C
"""
import warnings
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import forest_to_csv
import csv_to_c
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'

OUTPUT_NAME = "temp_forest"
C_DATA_OUTPUT = "forest_data"
BINARY_OUTPUT = "temp.bin"

def forest_to_c(rf: RandomForestClassifier,
                output_name=OUTPUT_NAME, c_data_output=C_DATA_OUTPUT, binary_output=BINARY_OUTPUT,
                keep_temporary_files=False):
    """
    Convert a random forest classifier to C code and save the output to a CSV file.
    C_DATA_OUTPUT should be kept as default for use with included C code.

    Args:
        rf (RandomForestClassifier): The random forest classifier to convert.
        output_name (str): The name of the output file. Defaults to OUTPUT_NAME.
        c_data_output (str): The directory where the C data output will be saved. Defaults to C_DATA_OUTPUT.
        binary_output (str): The directory where the binary output will be saved. Defaults to BINARY_OUTPUT.
        keep_temporary_files (bool): Whether to keep the temporary files. Defaults to False.

    Returns:
        None

    Creates:
        OUTPUT_NAME.csv, OUTPUT_NAME.meta
        forest_data.c, forest_data.h
    """
    forest_to_csv.forest_struct_to_csv(rf, output_name)

    forest_bytes = csv_to_c.forest_to_binary(''.join((output_name, '.csv')),
                                    binary_output, write_to_file=False, metadata=''.join((output_name, '.meta')))
    csv_name = ''.join((output_name, '.csv'))
    meta_name = ''.join((output_name, '.meta'))
    csv_to_c.create_array_for_c(forest_bytes, csv_to_c.get_forest_structure(csv_name), meta_name, c_data_output)

    if keep_temporary_files:
        return None
    else:
        os.remove(csv_name)
        os.remove(meta_name)
