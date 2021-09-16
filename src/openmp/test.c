
#include <omp.h>
#include <stdio.h>
#include <sys/time.h>

float a[2048*32]= {3};
float b[2048*32]= {4};
int main() {
  //omp_set_num_threads(4);
  
  // to get total number of threads
  //omp_get_thread_num();
  
  //#pragma omp parallel 
  //{
  //  int ID= omp_get_thread_num();;
  //  printf("hello 0 (%d)\n", ID);
  //  #pragma omp barrier
  //  printf("hello 1 (%d)\n", ID);
  //}
  //printf("all done!\n");

  int count= 2048*32;
  int i=0;
  int j=0;
  struct timeval start, end;
  gettimeofday(&start, NULL);

  //#pragma omp parallel 
  //{
    for(j=0; j<100000; j++) {
    //printf("%d\n", j);
    #pragma omp simd 
    //#pragma omp for
    for(i=0; i<count; i++) {
      a[i]= a[i] + b[i];
    }
    }
  //}
  printf("Here");
  gettimeofday(&end, NULL);

  float delta = ((end.tv_sec  - start.tv_sec) * 1000000u + 
           end.tv_usec - start.tv_usec) / 1.e6;
  printf("%f s\n", delta);
}
