#! /bin/sh

GPU=$1

# fix the reference batch from DDPM (for in-distro and out-distro)

CUDA_VISIBLE_DEVICES=$GPU \


python metrics/metrics.py /scratch/gigi/fsddpm/cifar100_ddpm_sigma/sampling-conditional-in-distro-2022-04-11-01-22-01-001184/full_samples_conditional_10000x32x32x3_in-distro_5.npz /scratch/gigi/fsddpm/cifar100_vfsddpm_unet_film_mean_sigma_variational/sampling-conditional-in-distro-2022-04-09-04-06-06-885714/full_samples_conditional_10000x32x32x3_in-distro_5.npz

python metrics/metrics.py /scratch/gigi/fsddpm/cifar100_ddpm_sigma/sampling-conditional-out-distro-2022-04-11-01-22-01-001184/full_samples_conditional_10000x32x32x3_out-distro_5.npz /scratch/gigi/fsddpm/cifar100_vfsddpm_unet_film_mean_sigma_variational/sampling-conditional-out-distro-2022-04-09-04-06-06-885714/full_samples_conditional_10000x32x32x3_out-distro_5.npz


python metrics/metrics.py /scratch/gigi/fsddpm/cifar100_ddpm_sigma/sampling-conditional-in-distro-2022-04-11-01-22-01-001184/full_samples_conditional_10000x32x32x3_in-distro_5.npz /scratch/gigi/fsddpm/cifar100_vfsddpm_vit_lag_mean_patch_sigma_variational_discrete/sampling-conditional-in-distro-2022-04-09-11-23-25-847438/full_samples_conditional_10000x32x32x3_in-distro_5.npz

python metrics/metrics.py /scratch/gigi/fsddpm/cifar100_ddpm_sigma/sampling-conditional-out-distro-2022-04-11-01-22-01-001184/full_samples_conditional_10000x32x32x3_out-distro_5.npz /scratch/gigi/fsddpm/cifar100_vfsddpm_vit_lag_mean_patch_sigma_variational_discrete/sampling-conditional-out-distro-2022-04-09-11-23-25-847438/full_samples_conditional_10000x32x32x3_out-distro_5.npz


