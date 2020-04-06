#!/bin/bash

ORI_BINARY_DIR=$1

MOD_BINARY_DIR=$3

OUTPUT_DIR=$5

# Sample command to match binaries with the two directories. This script will find the binaries with the same name for matching.
 
# ./src/analysis_in_batch.sh /home/DeepBinDiff/input/coreutils/binaries/coreutils-5.93-O0/ /home/DeepBinDiff/input/coreutils/binaries/coreutils-5.93-O3/ output/

echo "start analyzing binaries in batch!"
echo "Binary directories: $ORI_BINARY_DIR and $MOD_BINARY_DIR"

if [ "$ORI_BINARY_DIR" != "" ]; then
	if [[ $ORI_BINARY_DIR == */ ]];
	then
		ORI_BINARY_DIR=${ORI_BINARY_DIR::-1}
	fi
else
	echo "Forget ORI_BINARY_DIR directory?"
	exit
fi

if [ "$MOD_BINARY_DIR" != "" ]; then
	if [[ $MOD_BINARY_DIR == */ ]];
	then
		MOD_BINARY_DIR=${MOD_BINARY_DIR::-1}
	fi
else
	echo "Forget MOD_BINARY_DIR directory?"
	exit
fi


if [ "$OUTPUT_DIR" != "" ]; then
	if [[ $OUTPUT_DIR == */ ]];
	then
		OUTPUT_DIR=${OUTPUT_DIR::-1}
	fi
else
	echo "Forget output directory?"
	exit
fi


for file in $ORI_BINARY_DIR/*
do
    rm $OUTPUT_DIR/*
    filename=$(basename $file)
    echo "\n\nanalyzing file: $filename"
    python3 src/deepbindiff.py --input1 $file --input2 $MOD_BINARY_DIR/$filename --outputDir $OUTPUT_DIR > log${filename}
done
