#!/bin/bash

source ~/garuda_env/bin/activate

# MobileNets
module_name="mobilenet_v3"
models=(
	"MobileNetV3Small"
	"MobileNetV3Large"
)

for model in "${models[@]}"
do
	log_output="./results/ptq/tflite/DEF_${model}.txt"
	for i in {1..5}
	do	
		args=(
			--module_name ${module_name}
			--model_name ${model}
			--run_num "${i}"
		)

		# convert to TFLite
		python3 pyscripts/postquant.py "${args[@]}" >> $log_output
		printf "\n----------------------------\n" >> $log_output
	done
done
