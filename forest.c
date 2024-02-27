/* ***************** Header / include files ( #include ) **********************/
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <errno.h>

#include "forest_data.h"
#include "forest_data.c"

/* *************** Constant / macro definitions ( #define ) *******************/
const int16_t branch_size = sizeof(depth_t)+sizeof(float)+sizeof(feature_t)+sizeof(next_node_t);
const int16_t leaf_size = sizeof(depth_t)+(sizeof(score_t)*NUM_CLASSES);

/* *********************** Function prototypes ******************************/
samples_t read_samples(char* file_name, FILE *read_file);
proba_t predict(float_t samples[FEATURE_COUNT]);
depth_t read_depth(uint8_t* ptr);
branch_t read_branch(uint8_t* ptr);
leaf_t read_leaf(uint8_t* ptr);
uint32_t find_next_tree(uint8_t* ptr, uint32_t sz, uint32_t index);
proba_t predict_proba(node_t * head);
int predict_index(float * probs);

/* *********************** Function definitions ******************************/
int main(int argc, char *argv[])
{
    samples_t sample_struct;
    proba_t probs;
    int index;
    FILE* read_file;

    printf("Reading forest from %s\n", argv[1]);

    sample_struct = read_samples(argv[1], read_file);
    if (sample_struct.status != 0) {
        return -1;
    }

    probs = predict(sample_struct.samples);
    if (probs.status != 0) {
        return -1;
    }

    index = predict_index(probs.proba);


    printf("Predicted class: %s\n", classes[index]);
    // printf("%s\n", classes[index]);

    return 0;
}

samples_t read_samples(char* file_name, FILE *read_file)
{
    // variables
    samples_t sample_struct;
    sample_struct.status = 0;
    if(file_name == NULL)
    {
        sample_struct.status = -1;
        return sample_struct;
    }
    read_file = fopen(file_name, "r");
    if (read_file == NULL)
    {
        sample_struct.status = -1;
        return sample_struct;
    }

    // read samples
    for (int i = 0; i < FEATURE_COUNT; i++) {
        fscanf(read_file, "%f,", &sample_struct.samples[i]);
    }
    
    return sample_struct;
}

proba_t predict(float_t samples[FEATURE_COUNT])
{
    // variables
    uint8_t* fptr1 = NULL;
    uint32_t index = 0;
    branch_t new_branch;
    leaf_t new_leaf;
    depth_t depth;
    uint32_t sz;
    bool first = true;
    proba_t proba;
    proba.status = 0;
    
    // nodes in linked list represent trees in forest
    node_t * head = NULL;
    head = (node_t *) malloc(NUM_CLASSES);

    if (head == NULL) {
        proba.status = -1;
        return proba;
    }

    head->next = NULL;
    node_t * current = head;

    // read forest from file
    fptr1 = forest_structure;
    sz = sizeof(forest_structure);

    while (index < sz) {
        depth = read_depth(&fptr1[index]);
        if (depth < 0) {
            if (first == true) {
                first = false;
            }
            else {
                current->next = (node_t *) calloc(1, sizeof(node_t));
                current = current->next;
            }
            new_leaf = read_leaf(&fptr1[index]);
            index += leaf_size;
            memcpy(current->val, new_leaf.score, sizeof(new_leaf.score));
            current->next = NULL;
            index = find_next_tree(fptr1, sz, index);
        }
        else {
            new_branch = read_branch(&fptr1[index]);
            index += branch_size;
            if (samples[new_branch.feature] > new_branch.threshold) {
                index += new_branch.next_node;
            }
            else {
                continue;
            }
        }
    }
    proba = predict_proba(head);

    free(head);
    return proba;
}

depth_t read_depth(uint8_t* ptr)
{
    if (ptr == NULL) {
        return -1;
    }
    depth_t d;
    if (sizeof(depth_t) == 1) {
        d = *ptr++;
    }
    else if (sizeof(depth_t) == 2) {
        d = (int16_t)(*ptr++) << 8;
        d |= (int16_t)(*ptr++);
    }
    return d;
}

branch_t read_branch(uint8_t* ptr)
{
    union {
        float_t f;
        uint8_t t[4];
    } u;
    branch_t b;
    if (sizeof(depth_t) == 1) {
        b.depth = *ptr++;
    }
    else if (sizeof(depth_t) == 2) {
        b.depth = (int16_t)(*ptr++) << 8;
        b.depth |= (int16_t)(*ptr++);
    }

    u.t[0] = (*ptr++);
    u.t[1] = (*ptr++);
    u.t[2] = (*ptr++);
    u.t[3] = (*ptr++);
    b.threshold = u.f;
    b.feature = *ptr++;
    if (sizeof(next_node_t) == 2) {
        b.next_node = (int16_t)((*ptr++)) << 8;
        b.next_node |= (int16_t)(*ptr++);
    }
    else if (sizeof(next_node_t) == 4) {
        b.next_node = (int32_t)((*ptr++)) << 24;                                                      // TODO test this case
        b.next_node |= (int32_t)(*ptr++) << 16;
        b.next_node |= (int32_t)(*ptr++) << 8;
        b.next_node |= (int32_t)(*ptr++);
    }

    return b;
}

leaf_t read_leaf(uint8_t* ptr)
{
    leaf_t l;
    if (sizeof(depth_t) == 1) {
        l.depth = *ptr++;
    }
    else if (sizeof(depth_t) == 2) {
        l.depth = (int16_t)(*ptr++) << 8;
        l.depth |= (int16_t)(*ptr++);
    }
    if (sizeof(score_t) == 1) {
        for (int i = 0; i < NUM_CLASSES; i++) {
            l.score[i] = *ptr++;
        }
    }
    else if (sizeof(score_t) == 2) {
        for (int i = 0; i < NUM_CLASSES; i++) {
            l.score[i] = (score_t)((*ptr++)) << 8;                                                      // TODO test this case
            l.score[i] |= (score_t)(*ptr++);
        }   
    }
    else if (sizeof(score_t) == 4) {
        for (int i = 0; i < NUM_CLASSES; i++) {
            l.score[i] = (score_t)((*ptr++)) << 24;                                                      // TODO test this case
            l.score[i] |= (score_t)(*ptr++) << 16;
            l.score[i] |= (score_t)(*ptr++) << 8;
            l.score[i] |= (score_t)(*ptr++);
        }
    }

    return l;
}

uint32_t find_next_tree(uint8_t* ptr, uint32_t sz, uint32_t index)
{
    depth_t d;
    d = read_depth(&ptr[index]);
    while (d != 1) {
        if (index >= sz) {
            break;
        }
        if (d < 0) {
            index += leaf_size;
        }
        else {
            index += branch_size;
        }
        d = read_depth(&ptr[index]);
    }
    return index;
}

proba_t predict_proba(node_t * head) 
{
    node_t * current = head;
    int16_t sample_count[NUM_CLASSES];
    int32_t total = 0;
    int16_t tree_index = 0;
    float_t total_proba[FOREST_SIZE][NUM_CLASSES];
    proba_t probs;
    probs.status = 0;

    while (current != NULL) {
        for (int i = 0; i < NUM_CLASSES; i++) {
            sample_count[i] = (current->val[i]);
            total += current->val[i];
        }
        current = current->next;
        for (int i = 0; i < NUM_CLASSES; i++) {
            total_proba[tree_index][i] = (float_t)sample_count[i] / total;
        }
        tree_index++;
        total = 0;
    }

    for (int i = 0; i < NUM_CLASSES; i++) {
        probs.proba[i] = 0;
        for (int j = 0; j < FOREST_SIZE; j++) {
            probs.proba[i] += (float_t)total_proba[j][i] / FOREST_SIZE;
        }
    }

    return probs;
}

int predict_index(float *probs)
{
    int index = 0;
    for (int i = 0; i < NUM_CLASSES; i++) {
        if (probs[i] > probs[index]) {
            index = i;
        }
    }
    return index;
}