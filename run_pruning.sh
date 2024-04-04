#!/bin/bash

source ~/garuda_env/bin/activate

module_name="mobilenet_v3"
models=(
	"MobileNetV3Small"
	"MobileNetV3Large"
)

sparsities=(
	"0.25"
	"0.50"
	"0.75"
)
runners=(
	"CPU"
	"GPU"
)

for runner in "${runners[@]}"
do
	for model in "${models[@]}"
	do
		for sparsity in "${sparsities[@]}"
		do
			log_output="./results/pruning/DEF_${runner}_${model}_${sparsity}_sparsity.txt"
			for i in {1..5} 
			do
				pruning_options=(
					--module_name ${module_name}
					--model_name ${model}
					--target_sparsity ${sparsity}
					--run_num "${i}"
					--tuner "${runner}" 
				)

				# Run pruning in python
				python3 pyscripts/pruning/prune.py "${pruning_options[@]}" >> $log_output
			done
		done
	done
done
