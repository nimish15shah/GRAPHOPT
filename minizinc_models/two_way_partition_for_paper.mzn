
%%%%% input parameters %%%%
int: n_V; % number of nodes in graph G
set of int: V = 1..n_V;

int: max_node_w; % maximum node weight in current G
array[V] of 1..max_node_w: node_w;

int: n_E;
array[1..n_E, 1..2] of int: E; % edges (an array (instead of a set) of tuples, for ease of modelling)

int: n_Vin; 
set of int: Vin = 1..n_Vin; % source nodes on incoming edges

int: n_Ein;
array[1..n_Ein, 1..2] of int: Ein;

array[Vin] of 1..2: PARTin; 

%%%%% decision variables %%%%

array[V] of var 0..2: PART; 
var 0..(max_node_w * n_V): PART_1_size;
var 0..(max_node_w * n_V): PART_2_size;

array[1..n_Ein] of var bool: Ein_crossings; 

%%%%% constraints %%%%
% acyclic and data-dependency constraint
constraint forall (e in 1..n_E) (
    let { int : src = E [e, 1]; % local variables for readability
          int : dst = E [e, 2];
       } in

    PART [dst] = PART [src] \/
    PART [dst] = 0
  );
 
% partition sizes constraint
constraint PART_1_size = sum ( [ node_w[v] | v in V where PART [v] == 1 ] );
constraint PART_2_size = sum ( [ node_w[v] | v in V where PART [v] == 2 ] );

% inter-thread communication constraint
constraint  forall (e in 1..n_Ein) (
    let { int : src = Ein [e, 1]; % local variables for readability
          int : dst = Ein [e, 2];
       } in
    Ein_crossings [e] =
    ( PART [dst] != 0 /\
      PART [dst] != PARTin [src]
    )
  );

%%%%% objective %%%%
solve maximize 10 * min (PART_1_size, PART_2_size) - sum ( Ein_crossings );

/* output [ "obj = \(obj)\n", */
/* %      "curr_CU_active= \(curr_CU_active) \n", */
/*       "mapped_per_CU_active= \(mapped_per_CU_active) \n", */
/* %      "node_active= \(node_active) \n", */
/*       "tot_local_edges= \(tot_local_edges) \n", */
/*     ]; */
