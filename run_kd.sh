#!/bin/bash

source ~/garuda_env/bin/activate

model_arch="MobileNetV3Small"
module_name="mobilenet_v3"
teachers=(
	"MobileNetV3Large"
	# "DenseNet121"
	# "DenseNet169"
	# "DenseNet201"
	# "EfficientNetB0"
	# "EfficientNetB1"
	# "EfficientNetB2"
	# "EfficientNetB3"
	# "EfficientNetB4"
	# "EfficientNetB5"	
)
tuners=(
	"CPU"
	"GPU"
)	

for teacher in "${teachers[@]}"
do
	for tuner in "${tuners[@]}"
	do
		tuning="$(tr [A-Z] [a-z] <<< "$tuner")"
		txt_output_location="./results/kd/${tuning}/DEF_${teacher}_${model_arch}.txt"
		for i in {1..5} 
		do
		    general_args=(
			# module name / preprocessing options
			--module_name ${module_name}
			# model name
			--student_model ${model_arch}
			# run number
			--run_num ${i}
			# teacher model
			--teacher ${teacher}
			--tuner ${tuner}
			)

			python3 pyscripts/kd/kd.py "${general_args[@]}" >> $txt_output_location
			printf "\n----------------------------\n" >> $txt_output_location
		done
	done
done
