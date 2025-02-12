#!/bin/bash

methods=("GramNet" "CNNDetect" "GANID" "DMID" "UnivFD" "RINE" "PatchCraft")
processing_methods=("resize" "centercrop" "tencrop" "texture_crop")
batch_sizes=(128 128 16 16)

for method in "${methods[@]}"; do
    if [[ "$method" == "CNNDetect" ]]; then
        for i in "${!processing_methods[@]}"; do
            for parameter in "0.1" "0.5"; do
                python val.py --batch_size "${batch_sizes[i]}" --processing_method "${processing_methods[i]}" --method "$method" --parameter "$parameter"
            done
        done
    elif [[ "$method" == "RINE" ]]; then
        for i in "${!processing_methods[@]}"; do
            for parameter in "4" "ldm"; do
                python val.py --batch_size "${batch_sizes[i]}" --processing_method "${processing_methods[i]}" --method "$method" --parameter "$parameter"
            done
        done
    elif [[ "$method" == "GANID" ]]; then
        for i in "${!processing_methods[@]}"; do
            for parameter in "progan" "stylegan2"; do
                python val.py --batch_size "${batch_sizes[i]}" --processing_method "${processing_methods[i]}" --method "$method" --parameter "$parameter"
            done
        done
    else
        for i in "${!processing_methods[@]}"; do
            python val.py --batch_size "${batch_sizes[i]}" --processing_method "${processing_methods[i]}" --method "$method"
        done
    fi
done
