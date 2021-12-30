#!/bin/bash

DIR=work_dirs/pretrained
mkdir -p $DIR

curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1D-Bp1I3AhymPDiR2GWVcQusF_o9q9lVL" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {{print $NF}}' ./cookie`&id=1D-Bp1I3AhymPDiR2GWVcQusF_o9q9lVL" -o $DIR/LQM_r50_fpn_1x.pth
rm cookie

curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1T4e9qkT4w5Rbue_nbba9Thrje4E6Q5um" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {{print $NF}}' ./cookie`&id=1T4e9qkT4w5Rbue_nbba9Thrje4E6Q5um" -o $DIR/LQM_r101_fpn_2x.pth
rm cookie

curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1o79wRRhEe3HwiWwxHIG69VGjk5RIpGQ2" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {{print $NF}}' ./cookie`&id=1o79wRRhEe3HwiWwxHIG69VGjk5RIpGQ2" -o $DIR/LQM_r101_dcnv2_fpn_2x.pth
rm cookie

curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1hfD0cGvJd2ZO2FiVmFzoQEGBjjVuRB52" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {{print $NF}}' ./cookie`&id=1hfD0cGvJd2ZO2FiVmFzoQEGBjjVuRB52" -o $DIR/LQM_x101_fpn_2x.pth
rm cookie

curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1TAsQntoprgIYXiFCA1BJJ2IOpY3MdmpX" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {{print $NF}}' ./cookie`&id=1TAsQntoprgIYXiFCA1BJJ2IOpY3MdmpX" -o $DIR/LQM_x101_dcnv2_fpn_2x.pth
rm cookie
