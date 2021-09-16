
FPATH ?= /esat/puck1/users/nshah/cpu_gpu_parallel/SuiteSparse/ssget/MM_LU_factors/sample_L.mtx

p2p_run:
	OMP_NUM_THREADS=12 KMP_AFFINITY=granularity=fine,compact,1 /users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Software_tools/graphopt_benchmark/SpMP/test/trsv_test $(FPATH)
