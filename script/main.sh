#! /bin/sh

GPU=$1
RUN=$2

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
PATCH_SIZE=$9
SAMPLE_SIZE=$10
BATCH_SIZE=$11
BATCH_SIZE_EVAL=$12


MODEL_FLAGS="--image_size ${IMAGE_SIZE} --in_channels ${IN_CHANNELS} --num_channels 64 
--context_channels 256 --dropout 0.2 --num_res_blocks 2 --model ${MODEL} --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size ${BATCH_SIZE} --batch_size_eval ${BATCH_SIZE_EVAL} --dataset ${DATASET} --ema_rate 0.995"
ENCODER_FLAGS="--patch_size ${PATCH_SIZE} --encoder_mode ${ENCODER} --sample_size ${SAMPLE_SIZE} 
--mode_conditioning ${CONDITIONING} --pool ${POOLING} --mode_context ${CONTEXT}"

CUDA_VISIBLE_DEVICES=$GPU \
python main.py $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $ENCODER_FLAGS

}

# Run config
case $RUN in
    # ddpm
    ddpm_omniglot_sigma)
      run 28 1 ddpm omniglot_back_eval None None None None 1 5 32 64
      ;;
    ddpm_cifar100_sigma)
      run 32 3 ddpm cifar100 None None None None 1 5 32 64
      ;;
    ddpm_minimagenet_sigma)
      run 32 3 ddpm minimagenet None None None None 1 5 32 64
      ;;
    ddpm_cifar100mix_sigma)
      run 32 3 ddpm cifar100mix None None None None 1 5 32 64
      ;;
    ddpm_celeba_sigma)
      run 64 3 ddpm celeba None None None None 1 5 8 16
      ;;

    # vit film mean
    vfsddpm_cifar100_vit_film_mean_sigma)
      run 32 3 vfsddpm cifar100 vit film mean deterministic 8 5 32 64
      ;;
    vfsddpm_minimagenet_vit_film_mean_sigma)
      run 32 3 vfsddpm minimagenet vit film mean deterministic 8 5 32 64
      ;;
    vfsddpm_cifar100mix_vit_film_mean_sigma)
      run 32 3 vfsddpm cifar100mix vit film mean deterministic 8 5 32 64
      ;;
    vfsddpm_celeba_vit_film_mean_sigma)
      run 64 3 vfsddpm celeba vit film mean deterministic 16 5 8 16
      ;;

    # unet film mean
    vfsddpm_cifar100_unet_film_mean_sigma)
      run 32 3 vfsddpm cifar100 unet film mean deterministic 8 5 32 64
      ;;
    vfsddpm_minimagenet_unet_film_mean_sigma)
      run 32 3 vfsddpm minimagenet unet film mean deterministic 8 5 32 64
      ;;
    vfsddpm_cifar100mix_unet_film_mean_sigma)
      run 32 3 vfsddpm cifar100mix unet film mean deterministic 8 5 32 64
      ;;
    vfsddpm_celeba_unet_film_mean_sigma)
      run 64 3 vfsddpm celeba unet film mean deterministic 16 5 8 16
      ;;

    # vit lag mean-patch
    vfsddpm_omniglot_vit_lag_meanpatch_sigma)
      run 28 1 vfsddpm omniglot_back_eval vit lag mean_patch deterministic 7 5 32 128
      ;;
    vfsddpm_cifar100_vit_lag_meanpatch_sigma)
      run 32 3 vfsddpm cifar100 vit lag mean_patch deterministic 8 5 32 64
      ;;

    vfsddpm_cifar100_vit_lag_agg_sigma)
      run 32 3 vfsddpm cifar100 vit lag agg deterministic 8 5 32 64
      ;;
    vfsddpm_cifar100_vit_lag_none_sigma)
      run 32 3 vfsddpm cifar100 vit lag none deterministic 8 5 24 32
      ;;
    
    vfsddpm_minimagenet_vit_lag_meanpatch_sigma)
      run 32 3 vfsddpm minimagenet vit lag mean_patch deterministic 8 5 32 64
      ;;
    vfsddpm_cifar100mix_vit_lag_meanpatch_sigma)
      run 32 3 vfsddpm cifar100mix vit lag mean_patch deterministic 8 5 32 64
      ;;
    vfsddpm_celeba_vit_lag_meanpatch_sigma)
      run 64 3 vfsddpm celeba vit lag mean_patch deterministic 16 5 8 16
      ;;

    # vit-set lag none
    vfsddpm_omniglot_vitset_lag_none_sigma)
      run 28 1 vfsddpm omniglot_back_eval vit_set lag none deterministic 7 5 32 64
      ;;
    vfsddpm_cifar100_vitset_lag_none_sigma)
      run 32 3 vfsddpm cifar100 vit_set lag none deterministic 8 5 32 64
      ;;
    vfsddpm_minimagenet_vitset_lag_none_sigma)
      run 32 3 vfsddpm minimagenet vit_set lag none deterministic 8 5 32 64
      ;;
    vfsddpm_cifar100mix_vitset_lag_none_sigma)
      run 32 3 vfsddpm cifar100mix vit_set lag none deterministic 8 5 32 64
      ;;
    vfsddpm_celeba_vitset_lag_none_sigma)
      run 64 3 vfsddpm celeba vit_set lag none deterministic 16 5 8 16
      ;;

    # unet fil mean variational
    vfsddpm_cifar100_unet_film_mean_vi)
      run 32 3 vfsddpm cifar100 unet film mean variational 8 5 32 64
      ;;
    vfsddpm_cifar100mix_unet_film_mean_vi)
      run 32 3 vfsddpm cifar100mix unet film mean variational 8 5 32 64
      ;;
    vfsddpm_celeba_unet_film_mean_vi)
      run 64 3 vfsddpm celeba unet film mean variational 16 5 8 16
      ;;

    # vit lag mean-patch variational_discrete
    vfsddpm_cifar100_vit_lag_meanpatch_vid)
      run 32 3 vfsddpm cifar100 vit lag mean_patch variational_discrete 8 5 32 64
      ;;
    vfsddpm_cifar100mix_vit_lag_meanpatch_vid)
      run 32 3 vfsddpm cifar100mix vit lag mean_patch variational_discrete 8 5 32 64
      ;;
    vfsddpm_celeba_vit_lag_meanpatch_vid)
      run 64 3 vfsddpm celeba vit lag mean_patch variational_discrete 16 5 8 16
      ;;
esac

