
SUITESPARSE= /esat/puck1/users/nshah/cpu_gpu_parallel/SuiteSparse
CC = gcc
OPTIMIZATION = -Ofast
# do "make library" to recompile SuiteSparse with new optimization and compiler

include $(SUITESPARSE)/SuiteSparse_config/SuiteSparse_config.mk
# "make config" to check all the paths and configurations


I = -I$(SUITESPARSE)/include

CS = $(LDFLAGS) -L$(SUITESPARSE)/lib -lcxsparse $(LDLIBS)

default: library cxsparse_trsv

cs_demo1: cs_demo1.c Makefile
	$(CC) $(CF) $(I) -o cs_demo1 cs_demo1.c $(CS)

cxsparse_trsv: cxsparse_trsv.c Makefile
	$(CC) $(CF) $(I) -o cxsparse_trsv cxsparse_trsv.c $(CS)

FPATH ?= /esat/puck1/users/nshah/cpu_gpu_parallel/SuiteSparse/CXSparse/Matrix/bcsstk01 
THREADS ?= 1
ITER ?= 1
cxsparse_trsv_run:
	./cxsparse_trsv $(FPATH) $(THREADS) $(ITER) 

umfpack_run:
	$(SUITESPARSE)/UMFPACK/Demo/my_graphopt_benchmark  $(FPATH) $(THREADS) $(ITER) 

library:
	( cd $(SUITESPARSE)/CXSparse/Lib ; $(MAKE) -B CC=$(CC) OPTIMIZATION=$(OPTIMIZATION) )
	( cd $(SUITESPARSE)/SuiteSparse_config ; $(MAKE) -B CC=$(CC) OPTIMIZATION=$(OPTIMIZATION) )

# restore SuiteSparse by recompiling with default compiler and optimization options
default_library:
	( cd $(SUITESPARSE)/CXSparse/Lib ; $(MAKE) -B )
	( cd $(SUITESPARSE)/SuiteSparse_config ; $(MAKE) -B )
