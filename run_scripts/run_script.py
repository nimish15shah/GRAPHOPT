
import os
import sys
import argparse

def main(argv=None):
  parser = argparse.ArgumentParser(description='Regress over a range of EXP and MNT len')
  parser.add_argument('--net', type=str, choices=['alarm', 'alarm_2', 'asia', 'cancer', 'cancer_2', 'barley', 'win95pts', 'win95pts_2', 'hailfinder', 'hepar2', 'andes', 'andes_2', 'pigs', 'pigs_2' ,'link', 'munin', 'munin_2', 'HAR_TAN', 'HAR_NaiveBayes26F', 'HAR_TAN26F', 'UIWADS_NaiveBayes3F_class1' , 'UIWADS_NaiveBayes3F_class2', 'UNIMIB_NB_window10', 'UNIMIB_TAN_window10', 'mnist' , 'uci_har_nb', 'test_net'], default='asia', help='Enter the name of the network to be analysed')
  parser.add_argument('--cir_type', type=str, choices=['ac','log'], default='ac', help='Specify the type of circuit to be read. Default= ac')
  parser.add_argument('--tmode', type=str, choices= ['try', 'float_add_opt', 'hw_tree_blocks', 'max_FixPt_err_query', 'output_min_max', 'munin_single_query_verify', 'eval_float_error', 'eval_fixed_error', 'adaptively_optimize_for_Fxpt','exhaust_search_for_max_error', 'dot_file_to_visualize_ac', 'float_add_optimization_exhaustive', 'generate_ASIP_cfile', 'vectorize_inputs_of_building_blocks', 'ac_eval_testset_error', 'scheduling_for_gather', 'hw_tree_blocks_all_nets' ] , help='test mode')
  
  args = parser.parse_args(argv)
  run(args)


def run(args):
  
  if args.tmode == 'hw_tree_blocks':
    fitness_incr= 5.0
    for step in range(1,6):
      run_cmd= "python main.py " + str(args.net) + ' -t --tmode ' + str(args.tmode) + ' --cir_type ' + str(args.cir_type) + ' --targs 5 4 2 ' + str(step*fitness_incr) + ' ALL >> ./LOG/run_log' + '_'+ str(args.net)
      print run_cmd
      ret= os.system(run_cmd)
  
  elif args.tmode == 'hw_tree_blocks_all_nets':
    networks= ['bnetflix_psdd', 'bbc_psdd','book_psdd', 'kdd', 'msnbc', 'baudio', 'cpu', 'eeg_eye', 'bank_note', 'bio_response']
#    networks= ['asia']
    fitness_incr= 5.0
    for net in networks:
      for step in range(2,3):
        run_cmd= "python main.py " + str(net) + ' -t --tmode hw_tree_blocks --cir_type ' + str(args.cir_type) + ' --targs 5 1 1 ' + str(step*fitness_incr) + ' ALL >> ./LOG/run_log' + '_'+ str(net)
        print run_cmd
        ret= os.system(run_cmd)

  elif args.tmode == 'scheduling_for_gather':
#    networks= ['bnetflix_psdd', 'bbc_psdd','bio_response']
    networks= ['bnetflix_psdd', 'bbc_psdd','book_psdd', 'kdd', 'msnbc', 'baudio', 'cpu', 'eeg_eye', 'bank_note', 'bio_response']
    fitness_incr= 5.0
    for net in networks:
      for step in range(2,3):
        run_cmd= "python main.py " + str(net) + ' -t --tmode scheduling_for_gather --cir_type ' + str(args.cir_type) + ' --targs 5 1 1 ' + str(step*fitness_incr) + ' ALL >> ./LOG/run_log' + '_'+ str(net)
        print run_cmd
        ret= os.system(run_cmd)

  else:
    assert 0

if __name__ == "__main__":
  sys.exit(main())
