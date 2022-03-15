# GRAPHOPT
Constrained-optimization based parallelization of irregular graphs

Commands to run the main core algorithms:

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


# Dependencies
Required python packages can be installed from the requirements.txt file with the pip as follows:
`pip install -r requirements.txt`

Google OR-Tools has to be separately installed with the FlatZinc support from https://developers.google.com/optimization/install#flatzinc
The PATH and LD_LIBRARY_PATH enviornment variables also have to be updated to point to the OR-Tools location.

The c++ code is compiled with GCC 4.8.5 in our setup.

If you use this repository, please cite out work:

<a id="1">[1]</a>
Shah, N., Meert, W. and Verhelst, M., 2021. GRAPHOPT: constrained optimization-based parallelization of irregular graphs. arXiv preprint arXiv:2105.01976.
