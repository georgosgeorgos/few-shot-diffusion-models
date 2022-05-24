#! /bin/sh

GPU=$1
RUN=$2
DATE=$3
STEP=$4
NSAMPLES=$5

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
SAMPLING=$9
PATCH_SIZE=$10
SAMPLE_SIZE=$11
BATCH_SIZE=$12

# use ema model for sampling
MODEL_FLAGS="--image_size ${IMAGE_SIZE} --in_channels ${IN_CHANNELS} --num_channels 64 
--context_channels 256 --sample_size 5 --model ${MODEL} 
--model_path ${DATASET}_${MODEL}_${ENCODER}_${CONDITIONING}_${POOLING}_sigma_${CONTEXT}/run-${DATE}/ema_0.995_${STEP}.pt  --learn_sigma True"
SAMPLE_FLAGS="--batch_size ${BATCH_SIZE} --batch_size_eval ${BATCH_SIZE} --num_samples ${NSAMPLES} --timestep_respacing 250 
--mode_conditional_sampling ${SAMPLING} --dataset ${DATASET}"
ENCODER_FLAGS="--patch_size ${PATCH_SIZE} --encoder_mode ${ENCODER} --sample_size ${SAMPLE_SIZE} 
--mode_conditioning ${CONDITIONING} --pool ${POOLING}  --mode_context ${CONTEXT}"

CUDA_VISIBLE_DEVICES=$GPU \
python sample_conditional.py $MODEL_FLAGS $SAMPLE_FLAGS $ENCODER_FLAGS

}

# Run config
case $RUN in

    #_____________________________________________________________________________#
    ## DDPM unconditional #########################################################

    # save reference batch in-distro
    ddpm_cifar100_sigma_indistro)
      run 32 3 ddpm cifar100 None None None None in-distro 1 5 128
      ;;
    ddpm_cifar100mix_sigma_indistro)
      run 32 3 ddpm cifar100mix None None None None in-distro 1 5 128
      ;;
    ddpm_minimagenet_sigma_indistro)
      run 32 3 ddpm minimagenet None None None None in-distro 1 5 128
      ;;
    ddpm_celeba_sigma_indistro)
      run 64 3 ddpm celeba None None None None in-distro 1 5 64
      ;;
      
    # save reference batch out-distro
    ddpm_cifar100_sigma_outdistro)
      run 32 3 ddpm cifar100 None None None None out-distro 1 5 128
      ;;
    ddpm_cifar100mix_sigma_outdistro)
      run 32 3 ddpm cifar100mix None None None None out-distro 1 5 128
      ;;
    ddpm_minimagenet_sigma_outdistro)
      run 32 3 ddpm minimagenet None None None None out-distro 1 5 128
      ;;
    ddpm_celeba_sigma_outdistro)
      run 64 3 ddpm celeba None None None None out-distro 1 5 64
      ;;
    
    #_____________________________________________________________________________#
    ## VfsDGM out-distro conditional sampling #####################################

    # vit film mean
    vfsddpm_cifar100_vit_film_mean_sigma_outdistro)
      run 32 3 vfsddpm cifar100 vit film mean deterministic out-distro 8 5 128
      ;;
    vfsddpm_cifar100mix_vit_film_mean_sigma_outdistro)
      run 32 3 vfsddpm cifar100mix vit film mean deterministic out-distro 8 5 128
      ;;
    vfsddpm_minimagenet_vit_film_mean_sigma_outdistro)
      run 32 3 vfsddpm minimagenet vit film mean deterministic out-distro 8 5 128
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
    vfsddpm_minimagenet_unet_film_mean_sigma_outdistro)
      run 32 3 vfsddpm minimagenet unet film mean deterministic out-distro 8 5 64
      ;;
    vfsddpm_celeba_unet_film_mean_sigma_outdistro)
      run 64 3 vfsddpm celeba unet film mean deterministic out-distro 16 5 32
      ;;

    # vit lag meanpatch
    vfsddpm_cifar100_vit_lag_meanpatch_sigma_outdistro)
      run 32 3 vfsddpm cifar100 vit lag mean_patch deterministic out-distro 8 5 128
      ;;
    vfsddpm_cifar100_vit_lag_agg_sigma_outdistro)
      run 32 3 vfsddpm cifar100 vit lag agg deterministic out-distro 8 5 128
      ;;
    vfsddpm_cifar100mix_vit_lag_meanpatch_sigma_outdistro)
      run 32 3 vfsddpm cifar100mix vit lag mean_patch deterministic out-distro 8 5 128
      ;;
    vfsddpm_minimagenet_vit_lag_meanpatch_sigma_outdistro)
      run 32 3 vfsddpm minimagenet vit lag mean_patch deterministic out-distro 8 5 128
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
    vfsddpm_minimagenet_vitset_lag_none_sigma_outdistro)
      run 32 3 vfsddpm minimagenet vit_set lag none deterministic out-distro 8 5 128
      ;;
    vfsddpm_celeba_vitset_lag_none_sigma_outdistro)
      run 64 3 vfsddpm celeba vit_set lag none deterministic out-distro 8 5 64
      ;;

    # unet film mean variational
    vfsddpm_cifar100_unet_film_mean_vi_outdistro)
      run 32 3 vfsddpm cifar100 unet film mean variational out-distro 8 5 64
      ;;
    vfsddpm_cifar100mix_unet_film_mean_vi_outdistro)
      run 32 3 vfsddpm cifar100mix unet film mean variational out-distro 8 5 64
      ;;
    vfsddpm_minimagenet_unet_film_mean_vi_outdistro)
      run 32 3 vfsddpm minimagenet unet film mean variational out-distro 8 5 64
      ;;
    vfsddpm_celeba_unet_film_mean_vi_outdistro)
      run 64 3 vfsddpm celeba unet film mean variational out-distro 16 5 32
      ;;

    # vit lag meanpatch variational_discrete
    vfsddpm_cifar100_vit_lag_meanpatch_vid_outdistro)
      run 32 3 vfsddpm cifar100 vit lag mean_patch variational_discrete out-distro 8 5 128
      ;;
    vfsddpm_cifar100mix_vit_lag_meanpatch_vid_outdistro)
      run 32 3 vfsddpm cifar100mix vit lag mean_patch variational_discrete out-distro 8 5 128
      ;;
    vfsddpm_minimagenet_vit_lag_meanpatch_vid_outdistro)
      run 32 3 vfsddpm minimagenet vit lag mean_patch variational_discrete out-distro 8 5 128
      ;;
    vfsddpm_celeba_vit_lag_meanpatch_vid_outdistro)
      run 64 3 vfsddpm celeba vit lag mean_patch variational_discrete out-distro 16 5 64
      ;;

    ## VfsDGM in-distro conditional sampling #####################################################

    # vit film mean
    vfsddpm_cifar100_vit_film_mean_sigma_indistro)
      run 32 3 vfsddpm cifar100 vit film mean deterministic in-distro 8 5 128
      ;;
    vfsddpm_cifar100mix_vit_film_mean_sigma_indistro)
      run 32 3 vfsddpm cifar100mix vit film mean deterministic in-distro 8 5 128
      ;;
    vfsddpm_minimagenet_vit_film_mean_sigma_indistro)
      run 32 3 vfsddpm minimagenet vit film mean deterministic in-distro 8 5 128
      ;;
    vfsddpm_celeba_vit_film_mean_sigma_indistro)
      run 64 3 vfsddpm celeba vit film mean deterministic in-distro 16 5 64
      ;;
    
    # unet film mean 
    vfsddpm_cifar100_unet_film_mean_sigma_indistro)
      run 32 3 vfsddpm cifar100 unet film mean deterministic in-distro 8 5 64
      ;;
    vfsddpm_cifar100mix_unet_film_mean_sigma_indistro)
      run 32 3 vfsddpm cifar100mix unet film mean deterministic in-distro 8 5 64
      ;;
    vfsddpm_minimagenet_unet_film_mean_sigma_indistro)
      run 32 3 vfsddpm minimagenet unet film mean deterministic in-distro 8 5 64
      ;;
    vfsddpm_celeba_unet_film_mean_sigma_indistro)
      run 64 3 vfsddpm celeba unet film mean deterministic in-distro 16 5 32
      ;;

    # vit lag meanpatch
    vfsddpm_cifar100_vit_lag_meanpatch_sigma_indistro)
      run 32 3 vfsddpm cifar100 vit lag mean_patch deterministic in-distro 8 5 128
      ;;
    vfsddpm_cifar100_vit_lag_agg_sigma_indistro)
      run 32 3 vfsddpm cifar100 vit lag agg deterministic in-distro 8 5 128
      ;;
    vfsddpm_cifar100mix_vit_lag_meanpatch_sigma_indistro)
      run 32 3 vfsddpm cifar100mix vit lag mean_patch deterministic in-distro 8 5 128
      ;;
    vfsddpm_minimagenet_vit_lag_meanpatch_sigma_indistro)
      run 32 3 vfsddpm minimagenet vit lag mean_patch deterministic in-distro 8 5 128
      ;;
    vfsddpm_celeba_vit_lag_meanpatch_sigma_indistro)
      run 64 3 vfsddpm celeba vit lag mean_patch deterministic in-distro 16 5 64
      ;;

    # vit-set lag none
    vfsddpm_cifar100_vitset_lag_none_sigma_indistro)
      run 32 3 vfsddpm cifar100 vit_set lag none deterministic in-distro 8 5 128
      ;;  
    vfsddpm_cifar100mix_vitset_lag_none_sigma_indistro)
      run 32 3 vfsddpm cifar100mix vit_set lag none deterministic in-distro 8 5 128
      ;;
    vfsddpm_minimagenet_vitset_lag_none_sigma_indistro)
      run 32 3 vfsddpm minimagenet vit_set lag none deterministic in-distro 8 5 128
      ;;
    vfsddpm_celeba_vitset_lag_none_sigma_indistro)
      run 64 3 vfsddpm celeba vit_set lag none deterministic in-distro 16 5 64
      ;;

    # unet film mean variational
    vfsddpm_cifar100_unet_film_mean_vi_indistro)
      run 32 3 vfsddpm cifar100 unet film mean variational in-distro 8 5 64
      ;;
    vfsddpm_cifar100mix_unet_film_mean_vi_indistro)
      run 32 3 vfsddpm cifar100mix unet film mean variational in-distro 8 5 64
      ;;
    vfsddpm_minimagenet_unet_film_mean_vi_indistro)
      run 32 3 vfsddpm minimagenet unet film mean variational in-distro 8 5 64
      ;;
    vfsddpm_celeba_unet_film_mean_vi_indistro)
      run 64 3 vfsddpm celeba unet film mean variational in-distro 16 5 32
      ;;

    # vit lag meanpatch variational_discrete
    vfsddpm_cifar100_vit_lag_meanpatch_vid_indistro)
      run 32 3 vfsddpm cifar100 vit lag mean_patch variational_discrete in-distro 8 5 128
      ;;
    vfsddpm_cifar100mix_vit_lag_meanpatch_vid_indistro)
      run 32 3 vfsddpm cifar100mix vit lag mean_patch variational_discrete in-distro 8 5 128
      ;;
    vfsddpm_minimagenet_vit_lag_meanpatch_vid_indistro)
      run 32 3 vfsddpm minimagenet vit lag mean_patch variational_discrete in-distro 8 5 128
      ;;
    vfsddpm_celeba_vit_lag_meanpatch_vid_indistro)
      run 64 3 vfsddpm celeba vit lag mean_patch variational_discrete in-distro 16 5 64
      ;;

esac
