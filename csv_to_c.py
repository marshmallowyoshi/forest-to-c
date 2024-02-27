"""
This script is used to convert the forest.csv file into a compact binary file for use in embedded systems.
"""

import csv
import numpy as np

def forest_to_binary(file_name:str, binary_name:str, metadata:str, write_to_file=True):
    """
    Converts a forest structure csv file into a binary file.
    """
    forest = csv.reader(open(file_name, 'r', encoding='utf-8-sig'), delimiter=',')

    if write_to_file:
        with open(binary_name, 'wb') as f:
            f.write(byte_struct(forest, metadata_to_dict(metadata)))
        return 0
    else:
        return byte_struct(forest, metadata_to_dict(metadata))
    
def convert_number_to_bytes(number, byte_type:str):
    """ Convert a number to a byte array.
    With the input byte_type being one of the following C types:
    int8_t, int16_t, int32_t, float_t
    Returns the original number if given an invalid byte_type"""
    match byte_type:
        case 'int8_t':
            return number.to_bytes(1, 'big', signed=True)
        case 'int16_t':
            return number.to_bytes(2, 'big', signed=True)
        case 'int32_t':
            return number.to_bytes(4, 'big', signed=True)
        case 'float_t':
            return np.float32(number).tobytes()
        case _:
            return number
        

def byte_struct(forest, meta:dict):
    """ Create a byte array from the forest structure"""
    empty_bytes = b''
    all_bytes = empty_bytes
    all_bytes_list = []
    branch_list = []

    for idx, row in enumerate(forest):
        if idx == 0:
            continue

        if row[1] != '':
            # threshold
            threshold = convert_number_to_bytes(row[1], meta['threshold_t'])
        else:
            threshold = empty_bytes

        if row[2] != '':
            # feature
            feature = convert_number_to_bytes(int(row[2]), meta['feature_t'])
        else:
            feature = empty_bytes

        if row[3] != '':
            # depth and value
            depth = convert_number_to_bytes(-1 * int(row[0]), meta['depth_t']) #(-1 * int(row[0])).to_bytes(1, 'big', signed=True)

            value = b''.join([convert_number_to_bytes(int(x), meta['score_t']) for x in row[3].strip('.][').replace('\n', '').split(', ')])
        else:
            depth = convert_number_to_bytes(int(row[0]), meta['depth_t'])

            value = empty_bytes

        if row[4] != '':
            branch_list += [[int(x) for x in row[4].strip('][').split(', ')]]
        else:
            branch_list += [[]]

        all_bytes_list += [[depth, threshold, feature, value]]

    # create linked branch positions
    all_bytes_list = live_traversal(all_bytes_list, branch_list, meta['next_node_t'])
    # final byte string
    all_bytes = b''.join([b''.join(x) for x in all_bytes_list])
    return all_bytes

def byte_count(
        all_bytes_list,
        start_idx,
        end_idx
        ):
    """ Get length of a range of bytes"""
    counter = 0
    for idx in range(start_idx, end_idx):
        counter += sum([len(x) for x in all_bytes_list[idx]])
    return counter

def links_to_pointers(
    branches,
    all_bytes_list,
    idx,
    next_node_t,
    ):
    if len(branches) != 2:
        return b''
    else:
        branch_idx = byte_count(all_bytes_list, idx, branches[1])
        return convert_number_to_bytes(branch_idx, next_node_t)

def live_traversal(
        all_bytes_list,
        all_branches,
        next_node_t,
        ):
    live_bytes = all_bytes_list[::-1]
    live_branches = all_branches[::-1]
    for idx, branch in enumerate(live_branches):
        live_bytes[idx] = live_bytes[idx] + [links_to_pointers(branch, live_bytes[::-1], len(live_bytes) - idx, next_node_t)]
    return live_bytes[::-1]

def bytes_to_decimal(all_bytes):
    return [b for b in all_bytes]

def bytes_to_hex(all_bytes):
    return [hex(b) for b in all_bytes]

def get_forest_structure(file_name):
    forest = csv.reader(open(file_name, 'r', encoding='utf-8-sig'), delimiter=',')
    return [True if row[4] != '' else False for row in forest][1:]

def metadata_to_dict(metadata):
    meta = {'largest_sample_size': 0,
            'feature_count': 0,
            'tree_count': 0,
            'class_count': 0,
            'classes': [],
            'max_depth': 0,
            'depth_t': '',
            'threshold_t': 'float_t',
            'feature_t': '',
            'next_node_t': '',
            'score_t': '',
            'branch_size': 0,
            'leaf_size': 0}

    line_count = 0
    with open(metadata, 'r', encoding='utf-8-sig') as f:
        # read metadata
        for line in f:
            line_count += 1
            if line.startswith('largest_sample_size'):
                meta['largest_sample_size'] = int(line.split(':')[1].strip())
            if line.startswith('feature_count'):
                meta['feature_count'] = int(line.split(':')[1].strip())
            elif line.startswith('tree_count'):
                meta['tree_count'] = int(line.split(':')[1].strip())
            elif line.startswith('class_count'):
                meta['class_count'] = int(line.split(':')[1].strip())
            elif line.startswith('classes'):
                meta['classes'] = [x for x in line.split(':')[1].strip().split(', ')]
            elif line.startswith('max_depth'):
                meta['max_depth'] = int(line.split(':')[1].strip())

    # get types
    # branch
    if meta['max_depth'] < 128:
        meta['depth_t'] = 'int8_t'
    else:
        meta['depth_t'] = 'int16_t'
    if meta['feature_count'] < 128:
        meta['feature_t'] = 'int8_t'
    else:
        meta['feature_t'] = 'int16_t'
    # leaf
    if meta['class_count'] < 128:
        meta['score_t'] = 'int8_t'
    elif meta['largest_sample_size'] < 32768:
        meta['score_t'] = 'int16_t'
    else:
        meta['score_t'] = 'int32_t'
    # next node
    meta['next_node_t'] = 'int16_t'
    

    

    # map to types to sizes
    type_sizes = {
        'int8_t': 1,
        'int16_t': 2,
        'int32_t': 4,
        'float_t': 4
    }

    # size in bytes
    meta['branch_size'] = (type_sizes[meta['depth_t']]
                           + type_sizes[meta['threshold_t']]
                           + type_sizes[meta['feature_t']]
                           + type_sizes[meta['next_node_t']])
    meta['leaf_size'] = type_sizes[meta['depth_t']] + (type_sizes[meta['score_t']]*meta['class_count'])
    max_byte_count = max(meta['leaf_size'], meta['branch_size'])*line_count
    if max_byte_count < 32768:
        meta['next_node_t'] = 'int16_t'
    else:
        meta['next_node_t'] = 'int32_t'

    return meta

def create_array_for_c(all_bytes, forest_structure, metadata, c_file_names='forest_data', byte_format='hex'):
    """
    Generate source and header files containing data and structure information.
    """

    c_file = "".join((c_file_names, '.c'))
    h_file = "".join((c_file_names, '.h'))

    if byte_format == 'hex':
        byte_list = bytes_to_hex(all_bytes)
    elif byte_format == 'decimal':
        byte_list = bytes_to_decimal(all_bytes)
    else:
        raise ValueError

    meta = metadata_to_dict(metadata)

    with open(c_file, 'w', encoding='utf-8-sig') as f:
        # includes
        f.write('#include <stdint.h>\n')

        # data
        f.write(f'char *classes[{meta["class_count"]}] = ')
        f.write('{')
        for c in meta['classes']:
            f.write(f'\"{c}\",\n')
        f.write('};\n')

        f.write('uint8_t forest_structure[] = {\n')
        ptr = 0
        for row in forest_structure:
            if row:
                vals = byte_list[ptr:ptr+meta['branch_size']]
                ptr += meta['branch_size']
            else:
                vals = byte_list[ptr:ptr+meta['leaf_size']]
                ptr += meta['leaf_size']
            f.write(f'{",".join([str(x) for x in vals])},\n')
        f.write('};\n')

    with open(h_file, 'w', encoding='utf-8-sig') as f:
        f.write('#include <stdint.h>\n')
        # Constants
        f.write(f'#define FEATURE_COUNT {meta["feature_count"]}\n')
        f.write(f'#define FOREST_SIZE {meta["tree_count"]}\n')
        f.write(f'#define NUM_CLASSES {meta["class_count"]}\n\n')

        # Types
        f.write(f'typedef {meta["depth_t"]} depth_t;\n')
        f.write(f'typedef {meta["feature_t"]} feature_t;\n')
        f.write(f'typedef {meta["next_node_t"]} next_node_t;\n')
        f.write(f'typedef {meta["score_t"]} score_t;\n')
        
        # Sample Structure
        f.write('typedef struct {\n')
        f.write('    float samples[FEATURE_COUNT];\n')
        f.write('    int status;\n')
        f.write('} samples_t;\n\n')

        # Probability Structure
        f.write('typedef struct {\n')
        f.write('    float proba[NUM_CLASSES];\n')
        f.write('    int status;\n')
        f.write('} proba_t;\n\n')

        # Branch Structure
        f.write('typedef struct {\n')
        f.write('    depth_t depth;\n')
        f.write('    float threshold;\n')
        f.write('    feature_t feature;\n')
        f.write('    next_node_t next_node;\n')
        f.write('} branch_t;\n\n')

        # Leaf Structure
        f.write('typedef struct {\n')
        f.write('    depth_t depth;\n')
        f.write('    score_t score[NUM_CLASSES];\n')
        f.write('} leaf_t;\n\n')

        # Forest Linked List
        f.write('typedef struct node {\n')
        f.write('    score_t val[NUM_CLASSES];\n')
        f.write('    struct node * next;\n')
        f.write('} node_t;\n\n')
    return 0
