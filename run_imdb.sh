#!/bin/bash

cecho(){
	RED="\033[0;31m"
	YELLOW="\033[1;33m"
	NC="\033[0m" # No Color
	printf "${!1}${2} ${NC}\n"
}

gpu=${1:-0}
data="imdb"
we=50
ade=20
v="ratio_0.001 mam mdm"

python make_graph.py --dataset ${data}

for seed in 3 6 9 12 15
do
	cecho "RED" "Start Dataset=${data} Seed=${seed} Num=20 View=$v"
	python train.py --dataset ${data} --seed ${seed} --contrastive --view ${v} --lam 30 --lr 0.001 --gpu ${gpu} --num 20 --adjust_epoch ${ade} --warm_epoch ${we}

	cecho "RED" "Start Dataset=${data} Seed=${seed} Num=40 View=$v"
	python train.py --dataset ${data} --seed ${seed} --contrastive --view ${v} --lam 60 --lr 0.001 --gpu ${gpu} --num 40 --adjust_epoch ${ade} --warm_epoch ${we}

	cecho "RED" "Start Dataset=${data} Seed=${seed} Num=60 View=$v"
	python train.py --dataset ${data} --seed ${seed} --contrastive --view ${v} --lam 8 --lr 0.001 --gpu ${gpu} --num 60 --adjust_epoch ${ade} --warm_epoch ${we}
done


