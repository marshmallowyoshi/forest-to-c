#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "forest_data.h"
#include "forest_data.c"

int16_t branch_size = sizeof(depth_t)+sizeof(float)+sizeof(feature_t)+sizeof(next_node_t);
int16_t leaf_size = sizeof(depth_t)+(sizeof(score_t)*NUM_CLASSES);

float * predict(float_t samples[FEATURE_COUNT]);
depth_t read_depth(uint8_t* ptr);
branch_t read_branch(uint8_t* ptr);
leaf_t read_leaf(uint8_t* ptr);
uint32_t find_next_tree(uint8_t* ptr, uint32_t sz, uint32_t index);
float * predict_proba(node_t * head);
int predict_index(float * probs);

int main(void)
{
    float_t samples[FEATURE_COUNT] = {0.22044,  0.440961,         -0.96901,         0.247022};
    float * ret;
    int index;
    ret = predict(samples);
    index = predict_index(ret);
    printf("%s\n", classes[index]);    

    return 0;
}

float * predict(float_t samples[FEATURE_COUNT])
{
    // variables
    uint8_t* fptr1;
    uint32_t index = 0;
    branch_t new_branch;
    leaf_t new_leaf;
    depth_t depth;
    uint32_t sz;
    bool first = true;
    float *proba;
    
    // nodes in linked list represent trees in forest
    node_t * head = NULL;
    head = (node_t *) malloc(NUM_CLASSES);

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
    return proba;
}

depth_t read_depth(uint8_t* ptr)
{
    depth_t d;
    d = (depth_t)*ptr;
    return d;
}

branch_t read_branch(uint8_t* ptr)
{
    union {
        float_t f;
        uint8_t t[4];
    } u;
    branch_t b;
    b.depth = *ptr++;

    u.t[0] = (*ptr++);
    u.t[1] = (*ptr++);
    u.t[2] = (*ptr++);
    u.t[3] = (*ptr++);
    b.threshold = u.f;
    b.feature = *ptr++;
    if (sizeof(next_node_t) == 2) {
        b.next_node = (int16_t)((*ptr++));
        b.next_node |= (int16_t)(*ptr++) << 8;
    }
    else if (sizeof(next_node_t) == 4) {
        b.next_node = (int32_t)((*ptr++));                                                      // TODO test this case
        b.next_node |= (int32_t)(*ptr++) << 8;
        b.next_node |= (int32_t)(*ptr++) << 16;
        b.next_node |= (int32_t)(*ptr++) << 24;
    }

    return b;
}

leaf_t read_leaf(uint8_t* ptr)
{
    leaf_t l;
    l.depth = *ptr++;
    if (sizeof(score_t) == 2) {
        for (int i = 0; i < NUM_CLASSES; i++) {
            l.score[i] = (score_t)((*ptr++));                                                      // TODO test this case
            l.score[i] |= (score_t)(*ptr++) << 8;
        }   
    }
    else if (sizeof(score_t) == 4) {
        for (int i = 0; i < NUM_CLASSES; i++) {
            l.score[i] = (score_t)((*ptr++));                                                      // TODO test this case
            l.score[i] |= (score_t)(*ptr++) << 8;
            l.score[i] |= (score_t)(*ptr++) << 16;
            l.score[i] |= (score_t)(*ptr++) << 24;
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

float * predict_proba(node_t * head) 
{
    node_t * current = head;
    int16_t sample_count[NUM_CLASSES];
    int32_t total = 0;
    int16_t tree_index = 0;
    float_t total_proba[FOREST_SIZE][NUM_CLASSES];
    float_t *probs = malloc(NUM_CLASSES * sizeof(float_t));

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
        probs[i] = 0;
        for (int j = 0; j < FOREST_SIZE; j++) {
            probs[i] += (float_t)total_proba[j][i] / FOREST_SIZE;
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