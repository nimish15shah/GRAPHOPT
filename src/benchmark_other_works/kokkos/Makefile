
FPATH ?=/esat/puck1/users/nshah/cpu_gpu_parallel/SuiteSparse/ssget/MM_LU_factors/sample_L.mtx
ITER ?=1
THREADS ?= 2

kokkos_run:
	echo $(FPATH)
	echo ITER= $(ITER)
	OMP_NUM_THREADS=$(THREADS) KMP_AFFINITY=granularity=fine,compact,1 OMP_PROC_BIND=true /users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Software_tools/kokkos-kernels/build_perf_test/perf_test/sparse/sparse_sptrsv --offset 1 -lf $(FPATH) --loop $(ITER)
