
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>
#include "stdatomic.h"
//#include "data.h"
//#include "/esat/puck1/users/nshah/cpu_openmp/tretail_1024threads_512batch.c"
/* #include "/esat/puck1/users/nshah/cpu_openmp/HB_494_bus_TWO_WAY_PARTITION_TWO_WAY_LIMIT_LAYERS_CPU_2.c" */
/* #include "/esat/puck1/users/nshah/cpu_openmp/HB_orani678_LAYER_WISE_ALAP_CPU_2.c" */
#include "/esat/puck1/users/nshah/cpu_openmp/HB_orani678_TWO_WAY_PARTITION_TWO_WAY_LIMIT_LAYERS_CPU_32.c"
/* #include "/esat/puck1/users/nshah/cpu_openmp/GHS_indef_bratu3d_TWO_WAY_PARTITION_TWO_WAY_LIMIT_LAYERS_CPU_24.c" */
/* #include "/esat/puck1/users/nshah/cpu_openmp/Bai_rdb5000_TWO_WAY_PARTITION_TWO_WAY_LIMIT_LAYERS_CPU_16.c" */
/* #include "/esat/puck1/users/nshah/cpu_openmp/HB_bcsstk23_TWO_WAY_PARTITION_TWO_WAY_LIMIT_LAYERS_CPU_8.c" */

volatile int global_sense = 0;
_Atomic(int) count;
static unsigned int global_number_threads;

void gtmp_init(int number_threads){
    global_number_threads = number_threads;
    /* count = number_threads; */
    atomic_store_explicit(&count, number_threads, memory_order_relaxed);
    printf("setting global_number_threads: %d, count: %d\n", global_number_threads, count);
}

void gtmp_barrier(int tid){
    int  my_sense = global_sense^1;
    /* printf("global_sense is %d, my sense is: %d, count: %d \n",global_sense, my_sense, count); */
    if(atomic_fetch_sub_explicit(&count, 1, memory_order_relaxed)==1){
    /* if(__sync_fetch_and_sub(&count,1)==1){ */
        /* count = global_number_threads; */
        atomic_store_explicit(&count, global_number_threads, memory_order_relaxed);
        global_sense = my_sense;
        /* printf("global_sense is %d\n",global_sense); */
    }else{
        //spin
        /* printf("Wait in thread %d\n", tid); */
        while(my_sense != global_sense);
    }
}

int single_thread(const int n_iter, const int N_for_threads, const int t, const int N_layers, const int *layer_len, 
            const int * cum_layer_len, const int* ptr_offset,  const int* output_offset,
            float *res, const bool *op, const int *ptr_0, const int *ptr_1) {
  for(int i=0; i< n_iter; i++) {
    for (int l=0; l< N_layers; l++) {
      //printf("%d\n", l);
        //printf("%d %d\n", l, t);
      int cum_layer_offset= cum_layer_len[t*(N_layers + 1) + l];
      int ptr_o= ptr_offset[t];
      int out_o= output_offset[t];
      for (int layer_l = 0; layer_l< layer_len[t*N_layers + l]; layer_l++) {
        //printf("%d %d %d\n", l, t, layer_l);
        int cum_layer_l = cum_layer_offset + layer_l;
        int ptr_idx= cum_layer_l + ptr_o;
        //printf("cum_layer_l: %d, idx: %d\n", cum_layer_l, idx);
        float in_0= res[ptr_0[ptr_idx]];
        float in_1= res[ptr_1[ptr_idx]];

        int out_idx= cum_layer_l + out_o;
        res[out_idx]= op[ptr_idx]? in_0 * in_1 : in_0 + in_1;
          //printf("l, t, layer_l, b: %d, %d, %d, %d\n", l, t, layer_l, b);
        //printf("l, t, layer_l: %d, %d, %d\n", l, t, layer_l);

        /* #pragma omp master */
        /* { */
        /*   while(shared_mutex != (1<<N_for_threads) - 1); // until all thread bits set to 1 */
        /*   shared_mutex = 0; */
        /* } */
      }
      /* #pragma omp barrier */
      /* atomic_fetch_sub_explicit(&count, 1, memory_order_relaxed); */
      /* atomic_fetch_sub(&count, 1); */
      printf("entering barrier, Layer: %d, thread: %d\n", l, t); 
      #pragma omp flush
      /* gtmp_barrier(t); */
      /* #pragma omp flush */
      /* printf("Layer: %d, thread: %d\n", l, t); */ 
      /* #pragma omp master */
      /* { */
      /* } */
    }
  }
}
int par_func(const int n_iter, const int N_for_threads, const int N_layers, const int *layer_len, 
            const int * cum_layer_len, const int* ptr_offset,  const int* output_offset,
            float *res, const bool *op, const int *ptr_0, const int *ptr_1) {

  omp_set_num_threads(N_for_threads);
  int th_id;
  /* #pragma omp parallel private(th_id) shared(global_sense, count) */
  #pragma omp parallel private(th_id) 
  {
    th_id = omp_get_thread_num();
    single_thread(n_iter, N_for_threads, th_id, N_layers, layer_len, 
            cum_layer_len, ptr_offset, output_offset,
            res,op,ptr_0,ptr_1);
  }
}

int par_for(const int n_iter, const int N_for_threads, const int N_layers, const int *layer_len, 
            const int * cum_layer_len, const int* ptr_offset,  const int* output_offset,
            float *res, const bool *op, const int *ptr_0, const int *ptr_1) {

  double barrier_start_time[N_for_threads];
  double barrier_end_time[N_for_threads];
  double total_barrier_time[N_for_threads];

  for(int i=0; i< N_for_threads; i++) {
    barrier_start_time[i] = 0;
    barrier_end_time[i] = 0;
    total_barrier_time[i] = 0;
  }
  for(int i=0; i< n_iter; i++) {
    for (int l=0; l< N_layers; l++) {
      //printf("%d\n", l);
      #pragma omp parallel for
      for (int t = 0; t< N_for_threads; t++) {
        barrier_end_time[t] = omp_get_wtime();
        /* printf("barrier time of thread %d: %fus\n",t, 1.e6* (barrier_end_time[t] - barrier_start_time[t])); */
        if(barrier_start_time[t] != 0) {
          total_barrier_time[t] += barrier_end_time[t] - barrier_start_time[t];
        }
        //printf("%d %d\n", l, t);
        int cum_layer_offset= cum_layer_len[t*(N_layers + 1) + l];
        int ptr_o= ptr_offset[t];
        int out_o= output_offset[t];
        for (int layer_l = 0; layer_l< layer_len[t*N_layers + l]; layer_l++) {
          //printf("%d %d %d\n", l, t, layer_l);
          int cum_layer_l = cum_layer_offset + layer_l;
          int ptr_idx= cum_layer_l + ptr_o;
          //printf("cum_layer_l: %d, idx: %d\n", cum_layer_l, idx);
          float in_0= res[ptr_0[ptr_idx]];
          float in_1= res[ptr_1[ptr_idx]];

          int out_idx= cum_layer_l + out_o;
          res[out_idx]= op[ptr_idx]? in_0 * in_1 : in_0 + in_1;
            //printf("l, t, layer_l, b: %d, %d, %d, %d\n", l, t, layer_l, b);
          //printf("l, t, layer_l: %d, %d, %d\n", l, t, layer_l);
        }
        barrier_start_time[t]= omp_get_wtime();
        /* printf("Layer: %d, thread: %d\n", l, t); */ 
      }
      //cum_layer_len += layer_len[l];
    }
  }

  /* for (int t = 0; t< N_for_threads; t++) { */
  /*   printf("Total time spent on barrier: %f\n", total_barrier_time[t]); */
  /* } */
}

int scatter_leaves(int n_leaves, int* l_mem_idx_ls, float* l_val_ls, float* res){
  #pragma omp parallel 
  {
    #pragma omp for
      for (int l = 0; l< n_leaves; l++) {
        res[l_mem_idx_ls[l]] = l_val_ls[l];
      }
  }
}

int main(int argc, char *argv[]) {
  //printf("Total threads: %d\n", omp_get_num_threads());
  //float * res;
  //bool* op;
  //int * ptr_0;
  //int * ptr_1;
  //omp_set_num_threads(N_for_threads);
  //
  printf("Scattering leaves\n");
  scatter_leaves(n_leaves, l_mem_idx_ls, l_val_ls, res);
  printf("Done scattering\n");

  omp_set_num_threads(N_for_threads);
  gtmp_init(N_for_threads);

  par_func(1, N_for_threads, N_layers, layer_len, 
            cum_layer_len, ptr_offset, output_offset,
            res,op,ptr_0,ptr_1);
  
  int n_iter= 1e5;
  n_iter = atoi(argv[1]);
  struct timeval start, end;
  gettimeofday(&start, NULL);

  //printf("Total threads: %d\n", omp_get_num_threads());
  par_func(n_iter, N_for_threads, N_layers, layer_len, 
            cum_layer_len, ptr_offset, output_offset,
            res,op,ptr_0,ptr_1);
  
  gettimeofday(&end, NULL);

  float delta = ((end.tv_sec  - start.tv_sec) * 1000000u + 
           end.tv_usec - start.tv_usec) / 1.e6;
  
  for(int b=0; b< 1; b++) {
    printf("results,%f,actual,%f\n", res[head_node_idx], golden_val);
  }
  //printf("results %f, actual: %f\n", res[0], golden_val);
  printf("per_inference,%fs,\n", delta/n_iter);
  printf("throughput,%f MOPS\n", (1.0 * n_compute * n_iter)/(delta * 1.e6));
}

