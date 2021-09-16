

out_prefix="/esat/puck1/users/nshah/cpu_openmp/compiled_bin/"
batch_sz_ls=(512 256 128 64 32 16 8 4 2 1)
n_threads_ls=(512 256 128 64 32 16 8 4 2 1)
partition_datasets=("ad" "baudio" "bbc" "bnetflix" "book" "c20ng" "cr52" "cwebkb" "jester" "kdd" "mnist" "msnbc" "msweb" "nltcs" "pumsb_star" "tretail")
for n_threads in ${n_threads_ls[@]}; do
  for batch_sz in ${batch_sz_ls[@]}; do
    for net in ${partition_datasets[@]}; do
      cmd="${out_prefix}${net}_${n_threads}threads_${batch_sz}batch.out 1000"
      echo -n "${net},${n_threads},${batch_sz},"
      eval $cmd
      echo ""
    done
  done
done
