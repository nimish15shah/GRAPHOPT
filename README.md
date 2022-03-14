# GRAPHOPT
Constrained-optimization based parallelization of irregular graphs

Commands to run the main core algorithms:

`python main.py --tmode gen_super_layers --net <Workload name> --threads <Number of target threads> --cir_type <Type of the workload>`
 
 Examples,
`python main.py --tmode gen_super_layers --net mnist --threads 4 --cir_type psdd`

Log of the runtime of the parallelized OpenMP code will be created at `./log/run_log_openmp`
Log of superlayer generation time will be created at `./log/superlayer_gen_time_log`

If you use this repository, please cite out work:

<a id="1">[1]</a>
Shah, N., Meert, W. and Verhelst, M., 2021. GRAPHOPT: constrained optimization-based parallelization of irregular graphs. arXiv preprint arXiv:2105.01976.
