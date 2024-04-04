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

for model in "${models[@]}"
do
	for sparsity in "${sparsities[@]}" 
	do
		log_output="./results/pruning/latencies/pruned_keras/DEF_${model}_${sparsity}_sparsity.txt"

		for i in {1..10}
		do
		    options=(
			--module_name ${module_name}
			--model_name ${model}
			--sparsity ${sparsity})

		    python3 pyscripts/latencies/pruned_keras_latencies.py "${options[@]}" >> $log_output
		    printf "\n----------------\n" >> $log_output
		done	
	done
done

