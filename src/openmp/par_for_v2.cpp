
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>
//#include "data.h"
#include "../.././no_backup/openmp/tretail_TWO_WAY_PARTITION_TWO_WAY_LIMIT_LAYERS_CPU_FINE_4.c"
/* #include "/esat/puck1/users/nshah/cpu_openmp/tretail_1024threads_512batch.c" */
//#include "/esat/puck1/users/nshah/cpu_openmp/ad_1threads_1batch.c"
//#include "./no_backup/wilt_data.h"
//#include "./no_backup/tretail_data.h"

//extern int batch_sz;
//extern int N_for_threads;
//extern int N_layers;
//extern int *actual_layer_len;
//extern int tot_layer_len;

int par_for(const int n_iter, const int batch_sz, const int N_for_threads, const int N_layers, const int *layer_len, const int tot_layer_len, 
            const int * cum_layer_len,
            float *res, const bool *op, const int *ptr_0, const int *ptr_1) {

  int t;
  #pragma omp parallel private(t) 
  {
    t = omp_get_thread_num();
    //int cum_layer_len = 0;
    for(int i=0; i< n_iter; i++) {
      for (int l=0; l< N_layers; l++) {
      //printf("%d\n", l);
        //printf("%d %d\n", l, t);
        for (int layer_l = 0; layer_l< layer_len[t*N_layers + l]; layer_l++) {
          //printf("%d %d %d\n", l, t, layer_l);
          int cum_layer_l = cum_layer_len[t*(N_layers + 1) + l] + layer_l;
          int idx=cum_layer_l + t* tot_layer_len;
          //printf("cum_layer_l: %d, idx: %d\n", cum_layer_l, idx);
          /* #pragma omp simd */ 
          /* for (int b = 0; b< batch_sz; b++) { */
          /*   float in_0= res[ptr_0[idx] * batch_sz + b]; */
          /*   float in_1= res[ptr_1[idx] * batch_sz + b]; */
          /*   res[ptr_out[idx] * batch_sz + b]= op[idx]? in_0 * in_1 : in_0 + in_1; */
          /*   //printf("l, t, layer_l, b: %d, %d, %d, %d\n", l, t, layer_l, b); */
          /* } */
          float in_0= res[ptr_0[idx]];
          float in_1= res[ptr_1[idx]];
          res[ptr_out[idx]]= op[idx]? in_0 * in_1 : in_0 + in_1;
          //printf("l, t, layer_l: %d, %d, %d\n", l, t, layer_l);
        }
      }
    }
    //cum_layer_len += layer_len[l];
  }
}


int main(int argc, char *argv[]) {
  //printf("Total threads: %d\n", omp_get_num_threads());
  //float * res;
  //bool* op;
  //int * ptr_0;
  //int * ptr_1;
  //omp_set_num_threads(N_for_threads);

  omp_set_num_threads(N_for_threads);

  par_for(1, batch_sz, N_for_threads, N_layers, layer_len, tot_layer_len, 
            cum_layer_len,
            res,op,ptr_0,ptr_1);

  
  int n_iter= 1e5;
  n_iter = atoi(argv[1]);
  struct timeval start, end;
  double start_time, end_time;
  gettimeofday(&start, NULL);

  //printf("Total threads: %d\n", omp_get_num_threads());
  start_time = omp_get_wtime();
  par_for(n_iter, batch_sz, N_for_threads, N_layers, layer_len, tot_layer_len, 
          cum_layer_len,
          res,op,ptr_0,ptr_1);
  end_time = omp_get_wtime();
  gettimeofday(&end, NULL);

  float delta_prev = ((end.tv_sec  - start.tv_sec) * 1000000u + 
           end.tv_usec - start.tv_usec) / 1.e6;
  
  float delta= end_time - start_time;
  
  //for(int b=0; b< batch_sz; b++) {
  for(int b=0; b< 2; b++) {
    printf("results,%f,actual,%f,", res[(n_tot-1)*batch_sz + b], golden_val);
  }
  //printf("results %f, actual: %f\n", res[0], golden_val);
  printf("total_time,%fs,batch_sz,%d,n_iter,%d,", delta, batch_sz, n_iter);
  printf("per_batch,%fs,", delta/n_iter);
  printf("per_inference,%fs,\n", delta/(n_iter*batch_sz));

  printf("N_layers,%d,N_for_threads,%d,n_iter,%d,n_compute,%d,tim_per_inference(us),%f,throughput(MOPS),%f,result,%f,golden_val,%f\n", 
      N_layers, N_for_threads, n_iter, n_compute,
      (delta/n_iter)*1.e6,
      (1.0 * n_compute * n_iter)/(delta * 1.e6),
      res[(n_tot-1)*batch_sz],
      golden_val
      );
}

