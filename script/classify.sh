#! /bin/sh

GPU=$1
RUN=$2
DATE=$3
STEP=$4

run()
{

IMAGE_SIZE=$1
IN_CHANNELS=$2
MODEL=$3
DATASET=$4
ENCODER=$5
CONDITIONING=$6
POOLING=$7
CONTEXT=$8
EVALUATION=$9
PATCH_SIZE=$10
SAMPLE_SIZE=$11
BATCH_SIZE=$12


MODEL_FLAGS="--image_size ${IMAGE_SIZE} --in_channels ${IN_CHANNELS} --num_channels 64 --model ${MODEL} 
--model_path ${DATASET}_${MODEL}_${ENCODER}_${CONDITIONING}_${POOLING}_sigma_${CONTEXT}/run-${DATE}/ema_0.995_${STEP}.pt --learn_sigma True"
SAMPLE_FLAGS="--batch_size ${BATCH_SIZE} --batch_size_eval ${BATCH_SIZE} --num_samples 10000 --timestep_respacing 250 
--mode_evaluation ${EVALUATION}"
ENCODER_FLAGS="--patch_size ${PATCH_SIZE} --encoder_mode ${ENCODER} --sample_size ${SAMPLE_SIZE} --dataset ${DATASET} 
--mode_conditioning ${CONDITIONING} --pool ${POOLING} --mode_context ${CONTEXT}"

CUDA_VISIBLE_DEVICES=$GPU \
python classify.py $MODEL_FLAGS $SAMPLE_FLAGS $ENCODER_FLAGS

}

# Run config
case $RUN in
    # omniglot_back_eval
    vfsddpm_omniglot_vit_lag_meanpatch_sigma_outdistro)
      run 28 1 vfsddpm omniglot_back_eval vit lag mean_patch deterministic out-distro 7 5 128
      ;;
    vfsddpm_omniglot_vitset_lag_none_sigma_outdistro)
      run 28 1 vfsddpm omniglot_back_eval vit_set lag none deterministic out-distro 7 5 128
      ;;
    
    #_____________________________________________________________________________#
    ## VfsDGM out-distro conditional sampling
    
    # vit film mean
    vfsddpm_cifar100_vit_film_mean_sigma_outdistro)
      run 32 3 vfsddpm cifar100 vit film mean deterministic out-distro 8 5 128
      ;;
    vfsddpm_cifar100mix_vit_film_mean_sigma_outdistro)
      run 32 3 vfsddpm cifar100mix vit film mean deterministic out-distro 8 5 128
      ;;
    vfsddpm_celeba_vit_film_mean_sigma_outdistro)
      run 64 3 vfsddpm celeba vit film mean deterministic out-distro 16 5 64
      ;;
    
    # unet film mean
    vfsddpm_cifar100_unet_film_mean_sigma_outdistro)
      run 32 3 vfsddpm cifar100 unet film mean deterministic out-distro 8 5 64
      ;;
    vfsddpm_cifar100mix_unet_film_mean_sigma_outdistro)
      run 32 3 vfsddpm cifar100mix unet film mean deterministic out-distro 8 5 64
      ;;
    vfsddpm_celeba_unet_film_mean_sigma_outdistro)
      run 64 3 vfsddpm celeba unet film mean deterministic out-distro 16 5 32
      ;;

    # vit lag meanpatch
    vfsddpm_cifar100_vit_lag_meanpatch_sigma_outdistro)
      run 32 3 vfsddpm cifar100 vit lag mean_patch deterministic out-distro 8 5 128
      ;;
    vfsddpm_cifar100mix_vit_lag_meanpatch_sigma_outdistro)
      run 32 3 vfsddpm cifar100mix vit lag mean_patch deterministic out-distro 8 5 128
      ;;
    vfsddpm_celeba_vit_lag_meanpatch_sigma_outdistro)
      run 64 3 vfsddpm celeba vit lag mean_patch deterministic out-distro 16 5 64
      ;;

    # vitset lag none
    vfsddpm_cifar100_vitset_lag_none_sigma_outdistro)
      run 32 3 vfsddpm cifar100 vit_set lag none deterministic out-distro 8 5 128
      ;;  
    vfsddpm_cifar100mix_vitset_lag_none_sigma_outdistro)
      run 32 3 vfsddpm cifar100mix vit_set lag none deterministic out-distro 8 5 128
      ;;
    vfsddpm_celeba_vitset_lag_none_sigma_outdistro)
      run 64 3 vfsddpm celeba vit_set lag none deterministic out-distro 8 5 64
      ;;
esac
