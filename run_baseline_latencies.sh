#!/bin/bash

source ~/garuda_env/bin/activate

module_name="mobilenet_v3"
models=(
	"MobileNetV3Small"
	"MobileNetV3Large"
)

for model in "${models[@]}"
do
	log_output="./results/generate/latencies/DEF_${model}.txt"

	for i in {1..10}
	do
		options=(
			--module_name "${module_name}"
			--model_name "${model}"	
		)

		python3 pyscripts/latencies/baseline_latencies.py "${options[@]}" >> $log_output 	
	done
done
