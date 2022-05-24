# Few-Shot Diffusion Models (FSDM)


Under Review

* paper: `Few-Shot Diffusion Models`

## Set the env
```python
conda create -n fsdm python=3.6

git clone https://github.com/submission-neurips22/few-shot-diffusion-models

cd few-shot-diffusion-models

pip install -r requirements.txt
```

## Datasets
We train the models on small sets of dimension 2-20. 
Train/val/test sets use disjoint classes by default.

Binary:

* `Omniglot` (back_eval) - (1 x 28 x 28) - 964/97/659

RGB:

* `CIFAR100` - (3 x 32 x 32) - 60/20/20
* `CIFAR100mix` - (3 x 32 x 32) - 60/20/20
* `MinImageNet` - (3 x 32 x 32) - 64/16/20
* `CelebA` - (3 x 64 x 64) - 4444/635/1270

## Training

Train a DDPM on CIFAR100

```bash
sh script/run.sh gpu_num ddpm_cifar100 
```

Train a FSDM model on CIFAR100 dataset with ViT encoder, FiLM conditioning and MEAN aggregation

```bash
sh script/run.sh gpu_num vfsddpm_cifar100_vit_film_mean
```

Train a MODEL on DATASET with ENCODER, CONDITIONING and AGGREGATION

```bash
sh script/run.sh gpu_num {dddpm, cddpm, sddpm, addpm, vfsddpm}_{omniglot, cifar100, cifar100mix, minimagenet, cub, celeba}_{vit, unet}_{mean, lag, cls, sum_patch_mean}
```

## Sampling
Sample a FSDM model on CIFAR100 for new classes after 100K iterations 1000 samples

```bash
sh script/sample_conditional.sh gpu_num vfsddpm_cifar100_vit_film_mean_outdistro {date} 100000 1000
```


## Metrics
Compute FID, IS, Precision, Recall for a FSDM model on CIFAR100 new classes




## Acknoledgments

A lot of code and ideas borrowed from:

* https://github.com/openai/guided-diffusion
* https://github.com/lucidrains/vit-pytorch
* https://github.com/CompVis/latent-diffusion


