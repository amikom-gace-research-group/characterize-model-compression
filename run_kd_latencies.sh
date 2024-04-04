#!/bin/bash

source ~/garuda_env/bin/activate

module_name="mobilenet_v3"
student_model="MobileNetV3Small"
teacher_models=(
	"MobileNetV3Large"
)

for teacher in "${teacher_models[@]}"
do
	log_output="./results/kd/latencies/DEF_${teacher}.txt"

	for i in {1..10}
	do
		options=(
			--student_module_name "${module_name}"
			--student_model_name "${student_model}"
			--teacher_model_name "${teacher}")

		python3 pyscripts/latencies/kd_latencies.py "${options[@]}" >> $log_output 	
	done
done
