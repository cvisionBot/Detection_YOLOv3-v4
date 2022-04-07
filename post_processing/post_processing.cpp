#include <stdio.h>
#include <math.h>
#include "yolo_anchors.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <io.h>
#include <stdlib.h>
#include <cassert>

using namespace std;

#define OUTPUT_MAX_OBJ 10
#define NN_SIZE 416
#define NN_MAX_CONF 0.5f
#define NN_ANCHOR_PER_LABER 3
#define NN_OUTSIZE0 10
#define NN_OUTSIZE1 20
#define NN_OUTSIZE2 40

static float Sigmoid(float x)
{
    return 1.f / (1.f + expf(-x));
}

void PostProcessPerLayer(float *output, int layers, int layer_idx, int class_count, float *anchors) {
    
    int branch, anchor_x, anchor_y, anchor_c, ptr_idx;
    int anchor_w = layers;
    int anchor_h = layers;
    const int ptr_offset = anchor_w * anchor_h;

    float* x_ptr, * y_ptr, * w_ptr, * h_ptr, * obj_ptr, * cls_ptr;
    float ret_w, ret_h, ret_x, ret_y;
    float pred_w, pred_h, pred_x, pred_y;
    float* anchor_ptr = output;
    float* layer_anchors;

    int numobj = 0;
    float conf;

    for (branch = 0; branch < NN_ANCHORS_PER_LAYER; branch++)
    {
        x_ptr = anchor_ptr;
        y_ptr = x_ptr + ptr_offset;
        w_ptr = y_ptr + ptr_offset;
        h_ptr = w_ptr + ptr_offset;
        obj_ptr = h_ptr + ptr_offset;

        layer_anchors = &anchors[layer_idx * 2 * NN_ANCHORS_PER_LAYER + 2 * branch];
        for (anchor_y = 0; anchor_y < anchor_h; anchor_y++)
        {
            for (anchor_x = 0; anchor_x < anchor_w; anchor_x++)
            {
                ptr_idx = anchor_y * anchor_w + anchor_x;
                printf("obj ptr : %lf \n", Sigmoid(obj_ptr[ptr_idx]));
                if (Sigmoid(obj_ptr[ptr_idx]) > 0.1)
                {
                    for (anchor_c = 0; anchor_c < class_count; anchor_c++)
                    {
                        cls_ptr = obj_ptr + ptr_offset * (anchor_c + 1);
                      
                        if (Sigmoid(cls_ptr[ptr_idx]) > NN_MAX_CONF) 
                        {
                            if (numobj >= FSNET_OUTPUT_MAX_OBJ)
                                goto LOOP_EXIT;
                            ret_w = expf(w_ptr[ptr_idx]) * layer_anchors[0];
                            ret_h = expf(h_ptr[ptr_idx]) * layer_anchors[1];
                            ret_x = (anchor_x + Sigmoid(x_ptr[ptr_idx])) / anchor_w;
                            ret_y = (anchor_y + Sigmoid(y_ptr[ptr_idx])) / anchor_h;
                            conf = Sigmoid(obj_ptr[ptr_idx]) * Sigmoid(cls_ptr[ptr_idx]);
                           
                            pred_w = (int)(ret_w * NN_SIZE);
                            pred_h = (int)(ret_h * NN_SIZE);
                            pred_x = (int)(((ret_x)-(ret_w / 2)) * NN_SIZE);
                            pred_y = (int)(((ret_y)-(ret_h / 2)) * NN_SIZE);
           
                            printf("# # # # # # # Info # # # # # # \n");
                            printf("rect x : %lf \n", pred_x);
                            printf("rect y : %lf \n", pred_y);
                            printf("rect w : %lf \n", pred_w);
                            printf("rect h : %lf \n", pred_h);
                            printf("rect class : %d \n", anchor_c);
                            printf("rect conf : %lf \n", conf);
                            printf("# # # # # # # # # # # # # # # \n");
                            

                            numobj++;
                        }
                    }
                }
            }
        }
        anchor_ptr += ptr_offset * (5 + class_count);
    }

LOOP_EXIT:
    return;
}

void FS_PostProcessing_YOLO(float* output1, float* output2, float* output3) {

    PostProcessPerLayer(output1, NN_OUTSIZE0, 2, NN_CLASS, yolo_anchors);
    PostProcessPerLayer(output2, NN_OUTSIZE1, 1, NN_CLASS, yolo_anchors);
    PostProcessPerLayer(output3, NN_OUTSIZE2, 0, NN_CLASS, yolo_anchors);
};


int main(void) {
    float* output1, * output2, * output3;
    char output1_filename[200];
    char output2_filename[200];
    char output3_filename[200];
    
    snprintf(output1_filename, 200, "./dump/branch_10x10x75.bin");
    snprintf(output2_filename, 200, "./dump/branch_20x20x75.bin");
    snprintf(output3_filename, 200, "./dump/branch_40x40x75.bin");
    
    output1 = (float*)calloc(10 * 10 * 75, sizeof(float));
    output2 = (float*)calloc(20 * 20 * 75, sizeof(float));
    output3 = (float*)calloc(40 * 40 * 75, sizeof(float));

    FILE* output1_bin = fopen(output1_filename, "rb");
    FILE* output2_bin = fopen(output2_filename, "rb");
    FILE* output3_bin = fopen(output3_filename, "rb");
  
    fread(output1, 10 * 10 * 75, sizeof(float), output1_bin);
    fread(output2, 20 * 20 * 75, sizeof(float), output2_bin);
    fread(output3, 40 * 40 * 75, sizeof(float), output3_bin);

    PostProcessing_YOLO(output1, output2, output3);

    fclose(output1_bin);
    fclose(output2_bin);
    fclose(output3_bin);
    free(output1);
    free(output2);
    free(output3);

    return 0;