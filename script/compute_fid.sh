#! /bin/sh

GPU=$1

# fix the reference batch from DDPM (for in-distro and out-distro)

CUDA_VISIBLE_DEVICES=$GPU \

# python metrics/metrics.py /scratch/gigi/fsddpm/cifar100_ddpm_sigma/sampling-conditional-in-distro-2022-04-11-01-22-01-001184/full_samples_conditional_10000x32x32x3_in-distro_5.npz /scratch/gigi/fsddpm/cifar100_ddpm_sigma/sampling-conditional-in-distro-2022-04-11-01-22-01-001184/full_samples_conditional_10000x32x32x3_in-distro_5.npz

# Inception Score: 8.37218189239502
# FID: 15.354394775017056
# sFID: 18.034955078352368
# Precision: 0.6595
# Recall: 0.5656

# python metrics/metrics.py /scratch/gigi/fsddpm/cifar100_ddpm_sigma/sampling-conditional-out-distro-2022-04-11-01-22-01-001184/full_samples_conditional_10000x32x32x3_out-distro_5.npz /scratch/gigi/fsddpm/cifar100_ddpm_sigma/sampling-conditional-out-distro-2022-04-11-01-22-01-001184/full_samples_conditional_10000x32x32x3_out-distro_5.npz

# Inception Score: 8.37218189239502
# FID: 62.838982113459224
# sFID: 28.909927156386857
# Precision: 0.5867
# Recall: 0.4002

# python metrics/metrics.py /scratch/gigi/fsddpm/cifar100_ddpm_sigma/sampling-conditional-in-distro-2022-04-11-01-22-01-001184/full_samples_conditional_10000x32x32x3_in-distro_5.npz /scratch/gigi/fsddpm/cifar100_vfsddpm_vit_lag_mean_patch_sigma_deterministic/sampling-conditional-in-distro-2022-04-11-01-20-21-864811/full_samples_conditional_10000x32x32x3_in-distro_5.npz

# Inception Score: 9.875879287719727
# FID: 10.217449816799558
# sFID: 17.488873822707433
# Precision: 0.7167
# Recall: 0.6523

# python metrics/metrics.py /scratch/gigi/fsddpm/cifar100_ddpm_sigma/sampling-conditional-out-distro-2022-04-11-01-22-01-001184/full_samples_conditional_10000x32x32x3_out-distro_5.npz /scratch/gigi/fsddpm/cifar100_vfsddpm_vit_lag_mean_patch_sigma_deterministic/sampling-conditional-out-distro-2022-04-11-01-20-21-864811/full_samples_conditional_10000x32x32x3_out-distro_5.npz

# Inception Score: 7.163098335266113
# FID: 35.071504747495794
# sFID: 20.952861448469548
# Precision: 0.623
# Recall: 0.5322

# python metrics/metrics.py /scratch/gigi/fsddpm/cifar100_ddpm_sigma/sampling-conditional-in-distro-2022-04-11-01-22-01-001184/full_samples_conditional_10000x32x32x3_in-distro_5.npz /scratch/gigi/fsddpm/cifar100_vfsddpm_vit_set_lag_sigma_deterministic/sampling-conditional-in-distro-2022-04-10-23-31-02-407130/full_samples_conditional_10000x32x32x3_in-distro_5.npz

# Inception Score: 9.195577621459961
# FID: 12.392369518464022
# sFID: 17.262240095253787
# Precision: 0.6857
# Recall: 0.5733

# python metrics/metrics.py /scratch/gigi/fsddpm/cifar100_ddpm_sigma/sampling-conditional-out-distro-2022-04-11-01-22-01-001184/full_samples_conditional_10000x32x32x3_out-distro_5.npz /scratch/gigi/fsddpm/cifar100_vfsddpm_vit_set_lag_sigma_deterministic/sampling-conditional-out-distro-2022-04-10-23-31-02-407130/full_samples_conditional_10000x32x32x3_out-distro_5.npz

# Inception Score: 7.407537460327148
# FID: 40.71449108546966
# sFID: 22.122053921638553
# Precision: 0.5737
# Recall: 0.4457

# python metrics/metrics.py /scratch/gigi/fsddpm/cifar100_ddpm_sigma/sampling-conditional-in-distro-2022-04-11-01-22-01-001184/full_samples_conditional_10000x32x32x3_in-distro_5.npz /scratch/gigi/fsddpm/cifar100_vfsddpm_vit_film_mean_sigma_deterministic/sampling-conditional-in-distro-2022-04-11-03-55-29-267118/full_samples_conditional_10000x32x32x3_in-distro_5.npz

# Inception Score: 9.256915092468262
# FID: 13.343766828753303
# sFID: 21.32856044049072
# Precision: 0.6761
# Recall: 0.5525

# python metrics/metrics.py /scratch/gigi/fsddpm/cifar100_ddpm_sigma/sampling-conditional-out-distro-2022-04-11-01-22-01-001184/full_samples_conditional_10000x32x32x3_out-distro_5.npz /scratch/gigi/fsddpm/cifar100_vfsddpm_vit_film_mean_sigma_deterministic/sampling-conditional-out-distro-2022-04-11-03-55-29-267118/full_samples_conditional_10000x32x32x3_out-distro_5.npz

# Inception Score: 8.190563201904297
# FID: 45.497751540955505
# sFID: 29.8680334721912
# Precision: 0.5383
# Recall: 0.4592

# python metrics/metrics.py /scratch/gigi/fsddpm/cifar100_ddpm_sigma/sampling-conditional-in-distro-2022-04-11-01-22-01-001184/full_samples_conditional_10000x32x32x3_in-distro_5.npz /scratch/gigi/fsddpm/cifar100_vfsddpm_unet_film_mean_sigma_deterministic/sampling-conditional-in-distro-2022-04-10-06-38-35-345748/full_samples_conditional_10000x32x32x3_in-distro_5.npz

# Inception Score: 9.4474515914917
# FID: 11.841791823425751
# sFID: 17.642722409435805
# Precision: 0.7047
# Recall: 0.5645


# python metrics/metrics.py /scratch/gigi/fsddpm/cifar100_ddpm_sigma/sampling-conditional-out-distro-2022-04-11-01-22-01-001184/full_samples_conditional_10000x32x32x3_out-distro_5.npz /scratch/gigi/fsddpm/cifar100_vfsddpm_unet_film_mean_sigma_deterministic/sampling-conditional-out-distro-2022-04-10-06-38-35-345748/full_samples_conditional_10000x32x32x3_out-distro_5.npz

# Inception Score: 7.148565769195557
# FID: 38.499689370791884
# sFID: 22.212370257523162
# Precision: 0.5501
# Recall: 0.4674


#____________________________________________________________________
#python metrics/metrics.py /scratch/gigi/fsddpm/cifar100mix_ddpm_sigma/sampling-conditional-in-distro-2022-04-06-11-21-16-368778/full_samples_conditional_10000x32x32x3_in-distro_5.npz /scratch/gigi/fsddpm/cifar100mix_ddpm_sigma/sampling-conditional-in-distro-2022-04-06-11-21-16-368778/full_samples_conditional_10000x32x32x3_in-distro_5.npz

#python metrics/metrics.py /scratch/gigi/fsddpm/cifar100mix_ddpm_sigma/sampling-conditional-out-distro-2022-04-06-11-21-16-368778/full_samples_conditional_10000x32x32x3_out-distro_5.npz /scratch/gigi/fsddpm/cifar100mix_ddpm_sigma/sampling-conditional-out-distro-2022-04-06-11-21-16-368778/full_samples_conditional_10000x32x32x3_out-distro_5.npz

#python metrics/metrics.py /scratch/gigi/fsddpm/cifar100mix_ddpm_sigma/sampling-conditional-in-distro-2022-04-06-11-21-16-368778/full_samples_conditional_10000x32x32x3_in-distro_5.npz /scratch/gigi/fsddpm/cifar100mix_vfsddpm_vit_lag_mean_patch_sigma_deterministic/sampling-conditional-in-distro-2022-04-07-23-10-49-276825/full_samples_conditional_10000x32x32x3_in-distro_5.npz

# python metrics/metrics.py /scratch/gigi/fsddpm/cifar100mix_ddpm_sigma/sampling-conditional-out-distro-2022-04-06-11-21-16-368778/full_samples_conditional_10000x32x32x3_out-distro_5.npz /scratch/gigi/fsddpm/cifar100mix_vfsddpm_vit_lag_mean_patch_sigma_deterministic/sampling-conditional-out-distro-2022-04-07-23-10-49-276825/full_samples_conditional_10000x32x32x3_out-distro_5.npz

#python metrics/metrics.py /scratch/gigi/fsddpm/cifar100mix_ddpm_sigma/sampling-conditional-in-distro-2022-04-06-11-21-16-368778/full_samples_conditional_10000x32x32x3_in-distro_5.npz /scratch/gigi/fsddpm/cifar100mix_vfsddpm_vit_set_lag_sigma_deterministic/sampling-conditional-in-distro-2022-04-07-23-10-38-337314/full_samples_conditional_10000x32x32x3_in-distro_5.npz

# python metrics/metrics.py /scratch/gigi/fsddpm/cifar100mix_ddpm_sigma/sampling-conditional-out-distro-2022-04-06-11-21-16-368778/full_samples_conditional_10000x32x32x3_out-distro_5.npz /scratch/gigi/fsddpm/cifar100mix_vfsddpm_vit_set_lag_sigma_deterministic/sampling-conditional-out-distro-2022-04-07-23-10-38-337314/full_samples_conditional_10000x32x32x3_out-distro_5.npz


# python metrics/metrics.py /scratch/gigi/fsddpm/cifar100mix_ddpm_sigma/sampling-conditional-in-distro-2022-04-06-11-21-16-368778/full_samples_conditional_10000x32x32x3_in-distro_5.npz /scratch/gigi/fsddpm/cifar100mix_vfsddpm_vit_film_mean_sigma_deterministic/sampling-conditional-in-distro-2022-04-07-23-13-38-645760/full_samples_conditional_10000x32x32x3_in-distro_5.npz

# python metrics/metrics.py /scratch/gigi/fsddpm/cifar100mix_ddpm_sigma/sampling-conditional-out-distro-2022-04-06-11-21-16-368778/full_samples_conditional_10000x32x32x3_out-distro_5.npz /scratch/gigi/fsddpm/cifar100mix_vfsddpm_vit_film_mean_sigma_deterministic/sampling-conditional-out-distro-2022-04-07-23-13-38-645760/full_samples_conditional_10000x32x32x3_out-distro_5.npz


# python metrics/metrics.py /scratch/gigi/fsddpm/cifar100mix_ddpm_sigma/sampling-conditional-in-distro-2022-04-06-11-21-16-368778/full_samples_conditional_10000x32x32x3_in-distro_5.npz /scratch/gigi/fsddpm/cifar100mix_vfsddpm_unet_film_mean_sigma_deterministic/sampling-conditional-in-distro-2022-04-06-15-37-22-379553/full_samples_conditional_10000x32x32x3_in-distro_5.npz

# python metrics/metrics.py /scratch/gigi/fsddpm/cifar100mix_ddpm_sigma/sampling-conditional-out-distro-2022-04-06-11-21-16-368778/full_samples_conditional_10000x32x32x3_out-distro_5.npz /scratch/gigi/fsddpm/cifar100mix_vfsddpm_unet_film_mean_sigma_deterministic/sampling-conditional-out-distro-2022-04-06-15-37-22-379553/full_samples_conditional_10000x32x32x3_out-distro_5.npz


#________________________________________________________________________________________________________________

python metrics/metrics.py /scratch/gigi/fsddpm/cifar100_ddpm_sigma/sampling-conditional-out-distro-2022-04-11-01-22-01-001184/full_samples_conditional_10000x32x32x3_out-distro_5_transfer_minimagenet.npz /scratch/gigi/fsddpm/cifar100_ddpm_sigma/sampling-conditional-out-distro-2022-04-11-01-22-01-001184/full_samples_conditional_10000x32x32x3_out-distro_5_transfer_minimagenet.npz

python metrics/metrics.py /scratch/gigi/fsddpm/cifar100_ddpm_sigma/sampling-conditional-out-distro-2022-04-11-01-22-01-001184/full_samples_conditional_10000x32x32x3_out-distro_5_transfer_minimagenet.npz /scratch/gigi/fsddpm/cifar100_vfsddpm_unet_film_mean_sigma_deterministic/sampling-conditional-out-distro-2022-04-10-06-38-35-345748/full_samples_conditional_10000x32x32x3_out-distro_5_transfer_minimagenet.npz

python metrics/metrics.py /scratch/gigi/fsddpm/cifar100_ddpm_sigma/sampling-conditional-out-distro-2022-04-11-01-22-01-001184/full_samples_conditional_10000x32x32x3_out-distro_5_transfer_minimagenet.npz /scratch/gigi/fsddpm/cifar100_vfsddpm_vit_film_mean_sigma_deterministic/sampling-conditional-out-distro-2022-04-11-03-55-29-267118/full_samples_conditional_10000x32x32x3_out-distro_5_transfer_minimagenet.npz

python metrics/metrics.py /scratch/gigi/fsddpm/cifar100_ddpm_sigma/sampling-conditional-out-distro-2022-04-11-01-22-01-001184/full_samples_conditional_10000x32x32x3_out-distro_5_transfer_minimagenet.npz /scratch/gigi/fsddpm/cifar100_vfsddpm_vit_set_lag_sigma_deterministic/sampling-conditional-out-distro-2022-04-10-23-31-02-407130/full_samples_conditional_10000x32x32x3_out-distro_5_transfer_minimagenet.npz

python metrics/metrics.py /scratch/gigi/fsddpm/cifar100_ddpm_sigma/sampling-conditional-out-distro-2022-04-11-01-22-01-001184/full_samples_conditional_10000x32x32x3_out-distro_5_transfer_minimagenet.npz /scratch/gigi/fsddpm/cifar100_vfsddpm_vit_lag_mean_patch_sigma_deterministic/sampling-conditional-out-distro-2022-04-11-01-20-21-864811/full_samples_conditional_10000x32x32x3_out-distro_5_transfer_minimagenet.npz


#________________________________________________________________________________________________________________

python metrics/metrics.py /scratch/gigi/fsddpm/minimagenet_ddpm_sigma/sampling-conditional-in-distro-2022-04-15-00-44-47-158288/full_samples_conditional_10000x32x32x3_in-distro_5.npz /scratch/gigi/fsddpm/minimagenet_ddpm_sigma/sampling-conditional-in-distro-2022-04-15-00-44-47-158288/full_samples_conditional_10000x32x32x3_in-distro_5.npz

python metrics/metrics.py /scratch/gigi/fsddpm/minimagenet_ddpm_sigma/sampling-conditional-out-distro-2022-04-15-00-44-47-158288/full_samples_conditional_10000x32x32x3_out-distro_5.npz /scratch/gigi/fsddpm/minimagenet_ddpm_sigma/sampling-conditional-out-distro-2022-04-15-00-44-47-158288/full_samples_conditional_10000x32x32x3_out-distro_5.npz

python metrics/metrics.py /scratch/gigi/fsddpm/minimagenet_ddpm_sigma/sampling-conditional-in-distro-2022-04-15-00-44-47-158288/full_samples_conditional_10000x32x32x3_in-distro_5.npz /scratch/gigi/fsddpm/minimagenet_vfsddpm_vit_lag_mean_patch_sigma_deterministic/sampling-conditional-in-distro-2022-04-15-00-45-06-710259/full_samples_conditional_10000x32x32x3_in-distro_5.npz

python metrics/metrics.py /scratch/gigi/fsddpm/minimagenet_ddpm_sigma/sampling-conditional-out-distro-2022-04-15-00-44-47-158288/full_samples_conditional_10000x32x32x3_out-distro_5.npz /scratch/gigi/fsddpm/minimagenet_vfsddpm_vit_lag_mean_patch_sigma_deterministic/sampling-conditional-out-distro-2022-04-15-00-45-06-710259/full_samples_conditional_10000x32x32x3_out-distro_5.npz



python metrics/metrics.py /scratch/gigi/fsddpm/minimagenet_ddpm_sigma/sampling-conditional-in-distro-2022-04-15-00-44-47-158288/full_samples_conditional_10000x32x32x3_in-distro_5.npz /scratch/gigi/fsddpm/minimagenet_vfsddpm_vit_film_mean_sigma_deterministic/sampling-conditional-in-distro-2022-04-16-11-28-33-292996/full_samples_conditional_10000x32x32x3_in-distro_5.npz

python metrics/metrics.py /scratch/gigi/fsddpm/minimagenet_ddpm_sigma/sampling-conditional-out-distro-2022-04-15-00-44-47-158288/full_samples_conditional_10000x32x32x3_out-distro_5.npz /scratch/gigi/fsddpm/minimagenet_vfsddpm_vit_film_mean_sigma_deterministic/sampling-conditional-out-distro-2022-04-16-11-28-33-292996/full_samples_conditional_10000x32x32x3_out-distro_5.npz


python metrics/metrics.py /scratch/gigi/fsddpm/minimagenet_ddpm_sigma/sampling-conditional-in-distro-2022-04-15-00-44-47-158288/full_samples_conditional_10000x32x32x3_in-distro_5.npz /scratch/gigi/fsddpm/minimagenet_vfsddpm_vit_set_lag_sigma_deterministic/sampling-conditional-in-distro-2022-04-17-08-04-05-332267/full_samples_conditional_10000x32x32x3_in-distro_5.npz

python metrics/metrics.py /scratch/gigi/fsddpm/minimagenet_ddpm_sigma/sampling-conditional-out-distro-2022-04-15-00-44-47-158288/full_samples_conditional_10000x32x32x3_out-distro_5.npz /scratch/gigi/fsddpm/minimagenet_vfsddpm_vit_set_lag_sigma_deterministic/sampling-conditional-out-distro-2022-04-17-08-04-05-332267/full_samples_conditional_10000x32x32x3_out-distro_5.npz


python metrics/metrics.py /scratch/gigi/fsddpm/minimagenet_ddpm_sigma/sampling-conditional-in-distro-2022-04-15-00-44-47-158288/full_samples_conditional_10000x32x32x3_in-distro_5.npz /scratch/gigi/fsddpm/minimagenet_vfsddpm_unet_film_mean_sigma_deterministic/sampling-conditional-in-distro-2022-04-18-00-06-35-858511/full_samples_conditional_10000x32x32x3_in-distro_5.npz

python metrics/metrics.py /scratch/gigi/fsddpm/minimagenet_ddpm_sigma/sampling-conditional-out-distro-2022-04-15-00-44-47-158288/full_samples_conditional_10000x32x32x3_out-distro_5.npz /scratch/gigi/fsddpm/minimagenet_vfsddpm_unet_film_mean_sigma_deterministic/sampling-conditional-out-distro-2022-04-18-00-06-35-858511/full_samples_conditional_10000x32x32x3_out-distro_5.npz



python metrics/metrics.py /scratch/gigi/fsddpm/cifar100_ddpm_sigma/sampling-conditional-out-distro-2022-04-11-01-22-01-001184/full_samples_conditional_10000x32x32x3_out-distro_5.npz /scratch/gigi/fsddpm/cifar100_ddpm_sigma/sampling-conditional-out-distro-2022-04-11-01-22-01-001184/full_samples_conditional_implicit_10000x32x32x3_out-distro_5.npz


python metrics/metrics.py /scratch/gigi/fsddpm/minimagenet_ddpm_sigma/sampling-conditional-out-distro-2022-04-15-00-44-47-158288/full_samples_conditional_10000x32x32x3_out-distro_5.npz /scratch/gigi/fsddpm/minimagenet_ddpm_sigma/sampling-conditional-out-distro-2022-04-15-00-44-47-158288/full_samples_conditional_implicit_10000x32x32x3_out-distro_5.npz 

python metrics/metrics.py /scratch/gigi/fsddpm/cifar100_ddpm_sigma/sampling-conditional-out-distro-2022-04-11-01-22-01-001184/full_samples_conditional_10000x32x32x3_out-distro_5_transfer_minimagenet.npz /scratch/gigi/fsddpm/cifar100_ddpm_sigma/sampling-conditional-out-distro-2022-04-11-01-22-01-001184/full_samples_conditional_implicit_10000x32x32x3_out-distro_5_transfer_minimagenet.npz
