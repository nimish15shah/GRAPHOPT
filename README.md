# GRAPHOPT: constrained-optimization-based parallelization of irregular graphs for CPU multithreading
Constrained-optimization based parallelization of irregular graphs

Commands to run the full flow (i.e., superlayer generation, openMP code generation, parallel execution):

`python main.py --tmode full --net <Workload name> --threads <Number of target threads> --cir_type <Type of the workload>`
 
 Examples,

`python main.py --tmode full --net mnist --threads 4 --cir_type psdd`

`python main.py --tmode full --net HB/bcspwr01 --threads 4 --cir_type sptrsv`

Alternately, the following make commands run sample workloads:

`make psdd_sample`

`make sptrsv_sample`

`make all`

Log of the runtime of the parallelized OpenMP code will be created at `./log/run_log_openmp`

Log of the superlayer generation time will be created at `./log/superlayer_gen_time_log`


## Dependencies
Required python packages can be installed as follows:

`pip install -r requirements.txt`

Google OR-Tools with the FlatZinc support has to be separately installed from https://developers.google.com/optimization/install#flatzinc

The PATH and LD_LIBRARY_PATH enviornment variables also have to be updated to point to the OR-Tools location.

The c++ code is compiled with GCC 4.8.5 in our setup.

## Workloads
A few sample workloads are provided in the `./workloads/` directory. To reproduce all the results from the paper, sparse matrices can be downloaded from the SuiteSparse matrix collection https://sparse.tamu.edu/ .

## System requirements
Our experiments are performed on CentOS 7. Around 300GB disk storage is needed for the workloads (not provided with this repo), to reproduce all the experiments from the paper. Google OR-Tools can use parallel CPU threads for super layer generation, hence more CPU threads can be used to reduce the superlayer generation time.

If you use this repository, please cite out work:

<a id="1">[1]</a>
Shah, Nimish, Wannes Meert, and Marian Verhelst. "GRAPHOPT: constrained-optimization-based parallelization of irregular graphs." IEEE Transactions on Parallel and Distributed Systems (2022).
