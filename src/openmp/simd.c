
#include <omp.h>
#include <stdio.h>
#include <stdbool.h>
#include <sys/time.h>
//#include "data.h"
//#include "./no_backup/asia_data.h"
//#include "./no_backup/wilt_data.h"
#include "./no_backup/tretail_data.h"

//extern int batch_sz;
//extern int N_for_threads;
//extern int N_layers;
//extern int *actual_layer_len;
//extern int tot_layer_len;

int par_for(int batch_sz, int N_for_threads, int N_layers, int *layer_len, int tot_layer_len, 
            int * cum_layer_len,
            float *res, bool *op, int *ptr_0, int *ptr_1) {

  //int cum_layer_len = 0;
  for (int l=0; l< N_layers; l++) {
    //printf("%d\n", l);
    #pragma omp parallel 
    {
      #pragma omp for
        for (int t = 0; t< N_for_threads; t++) {
          //printf("%d %d\n", l, t);
          for (int layer_l = 0; layer_l< layer_len[t*N_layers + l]; layer_l++) {
            //printf("%d %d %d\n", l, t, layer_l);
            int cum_layer_l = cum_layer_len[t*(N_layers + 1) + l] + layer_l;
            int idx=cum_layer_l + t* tot_layer_len;
            //printf("cum_layer_l: %d, idx: %d\n", cum_layer_l, idx);
            #pragma omp simd simdlen(32)
            for (int b = 0; b< batch_sz; b++) {
              float in_0= res[ptr_0[idx] * batch_sz + b];
              float in_1= res[ptr_1[idx] * batch_sz + b];
              res[ptr_out[idx] * batch_sz + b]= op[idx]? in_0 * in_1 : in_0 + in_1;
              //printf("l, t, layer_l, b: %d, %d, %d, %d\n", l, t, layer_l, b);
            }
            //printf("l, t, layer_l: %d, %d, %d\n", l, t, layer_l);
          }
        }
    }
    //cum_layer_len += layer_len[l];
  }
}


int main() {
  //float * res;
  //bool* op;
  //int * ptr_0;
  //int * ptr_1;
  par_for(batch_sz, N_for_threads, N_layers, layer_len, tot_layer_len, 
            cum_layer_len,
            res,op,ptr_0,ptr_1);

  int n_iter= 1e3;
  struct timeval start, end;
  gettimeofday(&start, NULL);

  for(int i=0; i< n_iter; i++) {
    par_for(batch_sz, N_for_threads, N_layers, layer_len, tot_layer_len, 
            cum_layer_len,
            res,op,ptr_0,ptr_1);
  }
  gettimeofday(&end, NULL);

  float delta = ((end.tv_sec  - start.tv_sec) * 1000000u + 
           end.tv_usec - start.tv_usec) / 1.e6;
  
  //for(int b=0; b< batch_sz; b++) {
  for(int b=0; b< 8; b++) {
    printf("results %f, actual: %f\n", res[(n_tot-1)*batch_sz + b], golden_val);
  }
  //printf("results %f, actual: %f\n", res[0], golden_val);
  printf("%f s, batch_sz= %d, n_iter= %d\n", delta, batch_sz, n_iter);
  printf("%f s per batch\n", delta/n_iter);
}

