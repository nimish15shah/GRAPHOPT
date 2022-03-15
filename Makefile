
threads?=2

psdd_sample:
	python main.py --tmode gen_super_layers --net mnist --threads $(threads) --cir_type psdd

sptrsv_sample:
	python main.py --tmode gen_super_layers --net HB/bcspwr01 --threads $(threads) --cir_type sptrsv

all:
	python main.py --tmode gen_super_layers --net HB/bcspwr01 --threads $(threads) --cir_type sptrsv
	python main.py --tmode gen_super_layers --net HB/bcsstm02 --threads $(threads) --cir_type sptrsv
	python main.py --tmode gen_super_layers --net HB/bcsstm05 --threads $(threads) --cir_type sptrsv
	python main.py --tmode gen_super_layers --net HB/bcsstm22 --threads $(threads) --cir_type sptrsv
	python main.py --tmode gen_super_layers --net HB/can_24  --threads $(threads) --cir_type sptrsv
	python main.py --tmode gen_super_layers --net HB/can_62  --threads $(threads) --cir_type sptrsv
	python main.py --tmode gen_super_layers --net HB/ibm32   --threads $(threads) --cir_type sptrsv
	python main.py --tmode gen_super_layers --net tretail    --threads $(threads) --cir_type psdd
	python main.py --tmode gen_super_layers --net mnist      --threads $(threads) --cir_type psdd
	python main.py --tmode gen_super_layers --net nltcs      --threads $(threads) --cir_type psdd
	python main.py --tmode gen_super_layers --net kdd        --threads $(threads) --cir_type psdd
	python main.py --tmode gen_super_layers --net msnbc      --threads $(threads) --cir_type psdd
	python main.py --tmode gen_super_layers --net msweb      --threads $(threads) --cir_type psdd
	python main.py --tmode gen_super_layers --net ad         --threads $(threads) --cir_type psdd
	python main.py --tmode gen_super_layers --net baudio     --threads $(threads) --cir_type psdd
	python main.py --tmode gen_super_layers --net bbc        --threads $(threads) --cir_type psdd
	python main.py --tmode gen_super_layers --net bnetflix   --threads $(threads) --cir_type psdd
	python main.py --tmode gen_super_layers --net book       --threads $(threads) --cir_type psdd
	python main.py --tmode gen_super_layers --net c20ng      --threads $(threads) --cir_type psdd
	python main.py --tmode gen_super_layers --net cr52       --threads $(threads) --cir_type psdd
	python main.py --tmode gen_super_layers --net cwebkb     --threads $(threads) --cir_type psdd
	python main.py --tmode gen_super_layers --net jester     --threads $(threads) --cir_type psdd

