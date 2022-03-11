include "globals.mzn";

int: N;
int: n_CU;
set of int: NODES = 1..N;

int: N_done;
set of int: done_NODES = 1..N_done;

array[NODES] of set of NODES: one_predecessors;
array[NODES] of set of NODES: predecessors;

int: N_edges;
array[1..N_edges, 1..2] of int: edges;
%array[1..N_edges] of var 0..1: local_edge; 
var 0..N_edges: tot_local_edges; 

array[done_NODES] of 1..n_CU: done_CU; 

array[NODES] of var 1..n_CU: curr_CU; 
array[NODES] of var 0..n_CU: curr_CU_active; 
array[NODES] of var 0..1: node_active; 

array[1..n_CU] of var 0..(N+1): mapped_per_CU;
array[1..n_CU] of var 0..(N+1): mapped_per_CU_active;



% intra-barrier constraint
constraint forall(n in NODES) (
      forall (p in one_predecessors[n]) (
        curr_CU[p] == curr_CU[n]
      )
  );

constraint forall(n in NODES) (
  node_active[n] = forall (p in predecessors[n]) (curr_CU[p] == curr_CU[n] /\
  curr_CU_active[p] != 0)
);
constraint forall(n in NODES) (
  curr_CU_active[n]= node_active[n] * curr_CU[n]
);

% count mapped
constraint forall(cu in 1..n_CU) (
    mapped_per_CU[cu] = count(curr_CU, cu)
  );

constraint forall(cu in 1..n_CU) (
    mapped_per_CU_active[cu] = count(curr_CU_active, cu)
  );

% get local edges
constraint tot_local_edges = sum(e in 1..N_edges) (done_CU[edges[e,1]] ==
curr_CU_active[edges[e,2]]);

% symmetry break
%constraint curr_CU[1] = 1;

% objective
var int: obj;
%constraint obj= sum(sort(mapped_per_CU)[1..32]);
constraint obj= 8*min(mapped_per_CU_active) + tot_local_edges;
%constraint obj= 100*min(mapped_per_CU_active) + 10*tot_local_edges + sum(mapped_per_CU_active);
%constraint obj= 2*sum(mapped_per_CU_active) + min(mapped_per_CU_active) + tot_local_edges;
%constraint obj= 2*sum(mapped_per_CU_active);

ann: search_ann1;
%search_ann1 = int_search(curr_CU, first_fail, indomain_random);
%search_ann1 = int_search(done_CU, input_order, indomain_max);

solve :: seq_search ([search_ann1]) maximize obj;
%solve maximize obj;

output [ "obj = \(obj)\n",
%      "curr_CU_active= \(curr_CU_active) \n",
      "mapped_per_CU= \(mapped_per_CU) \n",
      "mapped_per_CU_active= \(mapped_per_CU_active) \n",
%      "node_active= \(node_active) \n",
      "tot_local_edges= \(tot_local_edges) \n",
    ];
