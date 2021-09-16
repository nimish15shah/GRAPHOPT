
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>
#include <atomic>

/* #define DEBUG */

/* #define NORMAL_PRINT */

int P = 32;
std::atomic<int> bar __attribute__ ((aligned (256))) {0}; // Counter of threads, faced barrier.
/* std::atomic<int> passed {0}; // Counter of threads, faced barrier. */
volatile int passed __attribute__ ((aligned (256)))  = 0; // Number of barriers, passed by all threads.
volatile int dummy __attribute__ ((aligned (256)))  = 0; 

void barrier_wait(int t)
{
    /* int passed_old = passed.load(std::memory_order_relaxed); */
    int passed_old = passed;

    if(bar.fetch_add(1) == (P - 1))
    {
        // The last thread, faced barrier.
        bar = 0;
        // Synchronize and store in one operation.
        /* passed.store(passed_old + 1, std::memory_order_release); */
        /* printf("release : %d\n", t); */
        passed = passed_old + 1;
        /* printf("Done"); */
    }
    else
    {
        // Not the last thread. Wait others.
        /* while(passed.load(std::memory_order_relaxed) == passed_old) {}; */
        while(passed == passed_old) {};
        // Need to synchronize cache with other threads, passed barrier.
        /* std::atomic_thread_fence(std::memory_order_acquire); */
        /* printf("Wait"); */
    }
}

/* #include "stdatomic.h" */
//#include "data.h"
//#include "/esat/puck1/users/nshah/cpu_openmp/tretail_1024threads_512batch.c"
/* #include "/esat/puck1/users/nshah/cpu_openmp/GHS_indef_c-59_TWO_WAY_PARTITION_TWO_WAY_LIMIT_LAYERS_CPU_COARSE_6.c" */
#include "/esat/puck1/users/nshah/cpu_openmp/Boeing_crystm02_LAYER_WISE_ALAP_CPU_COARSE_6.c"
/* #include "/esat/puck1/users/nshah/cpu_openmp/HB_orani678_TWO_WAY_PARTITION_TWO_WAY_LIMIT_LAYERS_CPU_COARSE_2.c" */
/* #include "/esat/puck1/users/nshah/cpu_openmp/GHS_psdef_torsion1_TWO_WAY_PARTITION_TWO_WAY_LIMIT_LAYERS_CPU_COARSE_1.c" */
/* #include "/esat/puck1/users/nshah/cpu_openmp/GHS_indef_bratu3d_TWO_WAY_PARTITION_TWO_WAY_LIMIT_LAYERS_CPU_24.c" */
/* #include "/esat/puck1/users/nshah/cpu_openmp/Bai_rdb5000_TWO_WAY_PARTITION_TWO_WAY_LIMIT_LAYERS_CPU_16.c" */
/* #include "/esat/puck1/users/nshah/cpu_openmp/HB_bcsstk23_TWO_WAY_PARTITION_TWO_WAY_LIMIT_LAYERS_CPU_2.c" */
/* #include "/esat/puck1/users/nshah/cpu_openmp/HB_bcsstk16_TWO_WAY_PARTITION_TWO_WAY_LIMIT_LAYERS_CPU_24.c" */
/* #include "/esat/puck1/users/nshah/cpu_openmp/Bai_dw8192_TWO_WAY_PARTITION_TWO_WAY_LIMIT_LAYERS_CPU_8.c" */

int single_thread(const int n_iter, const int N_for_threads, const int t, const int N_layers, 
            const int *layer_len, const int * cum_layer_len, 
            const int *ptr_in, const int* ptr_offset,  
             const int *op_len, const int* thread_offset,
            const float *nnz, const int * nnz_offset,
            const float *b_val, 
            float *res) {

  double barrier_start_time[N_layers];
  double barrier_end_time[N_layers];
  double total_barrier_time[N_layers];
  double total_execution_time[N_layers];
  double tot_op_layer[N_layers];
  double start_time;

  for(int i=0; i< n_iter; i++) {
    int nnz_ptr= nnz_offset[t];
    int ptr_in_ptr= ptr_offset[t]; // ptr to the ptr_in data structure
    int op_ptr= thread_offset[t];

    for (int l=0; l< N_layers; l++) {
#ifdef DEBUG
      start_time = omp_get_wtime();
      if (  op_ptr != thread_offset[t] + cum_layer_len[t*(N_layers + 1) +l]) {
        printf("wrong op_ptr: op_ptr:%d, cum_layer_len: %d\n",  op_ptr, thread_offset[t] + cum_layer_len[t*(N_layers + 1) + l]);
        exit(1);
      } 
#endif
      for (int layer_l = 0; layer_l< layer_len[t*N_layers + l]; layer_l++) {
        const int curr_op_len= op_len[op_ptr];
        float l_res= b_val[op_ptr];
        #pragma omp simd
        for (int i= 0; i< curr_op_len - 1; i++) {
          l_res += nnz[nnz_ptr + i] * res[ptr_in[ptr_in_ptr + i]];
        }
        res[op_ptr] = l_res / nnz[nnz_ptr + curr_op_len - 1];

        ptr_in_ptr += curr_op_len;
        nnz_ptr += curr_op_len;

        op_ptr++;
      }
#ifdef DEBUG
      total_execution_time[l] = (omp_get_wtime() - start_time) *1.e6;
      start_time = omp_get_wtime();
#endif
      #pragma omp barrier
      /* #pragma omp flush */
      /* barrier_wait(t); */
#ifdef DEBUG
      total_barrier_time[l] = (omp_get_wtime() - start_time) *1.e6;
#endif
      /* #pragma omp flush */
    }
  }
#ifdef DEBUG
  #pragma omp critical
  {
    double total=0;
    for (int l=0; l< N_layers; l++) {
      printf("thread: %d, layer: %d, layer_len: %d, exec: %fus, barrier: %fus\n", t, l, layer_len[t*N_layers + l], total_execution_time[l], total_barrier_time[l]);
      total += total_execution_time[l];
      total += total_barrier_time[l];
    }
    printf("thread: %d, total: %f\n", t, total);
  }
#endif
}
int par_func(const int n_iter, const int N_for_threads, const int N_layers, 
            const int *layer_len, const int * cum_layer_len, 
            const int *ptr_in, const int* ptr_offset,  
             const int *op_len, const int* thread_offset,
            const float *nnz, const int *nnz_offset,
            const float *b_val,
            float *res) {

  omp_set_num_threads(N_for_threads);
  int t;
  #pragma omp parallel private(t) shared(passed)
  /* #pragma omp parallel private(t) */ 
  {
    t = omp_get_thread_num();
    single_thread(n_iter, N_for_threads, t, N_layers, 
            layer_len,  cum_layer_len, 
            ptr_in,  ptr_offset,  
             op_len,  thread_offset,
            nnz, nnz_offset,
            b_val,
            res);
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
  omp_set_num_threads(N_for_threads);
  P= N_for_threads;

  par_func(1000, N_for_threads, N_layers, 
          layer_len,  cum_layer_len, 
          ptr_in,  ptr_offset,  
           op_len,  thread_offset,
          nnz, nnz_offset,
          b_val,
          res);
#ifdef NORMAL_PRINT
  for(int b=0; b< 1; b++) {
    printf("first results,%f,actual,%f\n", res[head_node_idx], golden_val);
  }
#endif
  
  int n_iter= 1e5;
  n_iter = atoi(argv[1]);
  struct timeval start, end;
  double start_time, end_time;

  gettimeofday(&start, NULL);

  start_time = omp_get_wtime();
  //printf("Total threads: %d\n", omp_get_num_threads());
  par_func(n_iter, N_for_threads, N_layers, 
          layer_len,  cum_layer_len, 
          ptr_in,  ptr_offset,  
           op_len,  thread_offset,
          nnz, nnz_offset,
          b_val,
          res);
  end_time = omp_get_wtime();
  gettimeofday(&end, NULL);

  float delta = ((end.tv_sec  - start.tv_sec) * 1000000u + 
           end.tv_usec - start.tv_usec) / 1.e6;
  
#ifdef NORMAL_PRINT
  for(int b=0; b< 1; b++) {
    printf("results,%f,actual,%f\n", res[head_node_idx], golden_val);
  }

  //printf("results %f, actual: %f\n", res[0], golden_val);
  printf("per_inference,%fs,\n", delta/n_iter);
  printf("per_inference according to other timer,%fs,\n", (end_time - start_time)/n_iter);
  printf("throughput,%f MOPS\n", (1.0 * n_compute * n_iter)/(delta * 1.e6));
  printf("throughput other timer,%f MOPS\n", (1.0 * n_compute * n_iter)/((end_time - start_time) * 1.e6));
#endif

  printf("N_layers,%d,N_for_threads,%d,n_iter,%d,n_compute,%d,tim_per_inference(us),%f,throughput(MOPS),%f,result,%f,golden_val,%f\n", 
      N_layers, N_for_threads, n_iter, n_compute,
      (delta/n_iter)*1.e6,
      (1.0 * n_compute * n_iter)/(delta * 1.e6),
      res[head_node_idx],
      golden_val
      );
}

