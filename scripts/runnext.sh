export TRANSFORMERS_VERBOSITY=error
bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/run_llavanext.sh 1.5
bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/run_llavanext.sh 1.0
bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/run_llavanext.sh 0.9

bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanext.sh 0.9
bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanext.sh 0.8
bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanext.sh 0.7
bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanext.sh 0.6