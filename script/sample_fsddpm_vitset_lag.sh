#! /bin/sh

# mnemosyne
# sh script/eval.sh 1 vfsddpm_omniglot_vitset_lag_none_sigma_outdistro 2022-04-04-17-26-17-064370 110000


# sh script/eval.sh 2 vfsddpm_cifar100_vitset_lag_none_sigma_outdistro 2022-04-10-23-31-02-407130 200000
# sh script/eval.sh 2 vfsddpm_cifar100_vitset_lag_none_sigma_indistro 2022-04-10-23-31-02-407130 200000

# sh script/eval.sh 2 vfsddpm_cifar100mix_vitset_lag_none_sigma_outdistro 2022-04-07-23-10-38-337314 100000
# sh script/eval.sh 2 vfsddpm_cifar100mix_vitset_lag_none_sigma_indistro 2022-04-07-23-10-38-337314 100000

# # sh script/eval.sh 2 vfsddpm_celeba_vitset_lag_none_sigma_outdistro 2022-04-09-06-32-29-893217 150000
# # sh script/eval.sh 2 vfsddpm_celeba_vitset_lag_none_sigma_indistro 2022-04-09-06-32-29-893217 150000

# #__________________________________________________________________________________________________________________________
# sh script/sample_conditional.sh 2 vfsddpm_cifar100_vitset_lag_none_sigma_outdistro 2022-04-10-23-31-02-407130 200000 10000
# sh script/sample_conditional.sh 2 vfsddpm_cifar100_vitset_lag_none_sigma_indistro 2022-04-10-23-31-02-407130 200000 10000

# sh script/sample_conditional.sh 2 vfsddpm_cifar100mix_vitset_lag_none_sigma_outdistro 2022-04-07-23-10-38-337314 100000 10000
# sh script/sample_conditional.sh 2 vfsddpm_cifar100mix_vitset_lag_none_sigma_indistro 2022-04-07-23-10-38-337314 100000 10000

# sh script/sample_conditional.sh 2 vfsddpm_celeba_vitset_lag_none_sigma_outdistro 2022-04-09-06-32-29-893217 150000 10000
# sh script/sample_conditional.sh 2 vfsddpm_celeba_vitset_lag_none_sigma_indistro 2022-04-09-06-32-29-893217 150000 10000


# sh script/eval.sh 3 vfsddpm_minimagenet_vitset_lag_none_sigma_outdistro 2022-04-17-08-04-05-332267 200000
# sh script/eval.sh 3 vfsddpm_minimagenet_vitset_lag_none_sigma_indistro 2022-04-17-08-04-05-332267 200000

#sh script/sample_conditional.sh 3 vfsddpm_minimagenet_vitset_lag_none_sigma_outdistro 2022-04-17-08-04-05-332267 200000 10000
#sh script/sample_conditional.sh 3 vfsddpm_minimagenet_vitset_lag_none_sigma_indistro 2022-04-17-08-04-05-332267 200000 10000


#sh script/distro.sh 5 vfsddpm_cifar100_vitset_lag_none_sigma_outdistro 2022-04-10-23-31-02-407130 200000
sh script/distro.sh 5 vfsddpm_cifar100_vitset_lag_none_sigma_indistro 2022-04-10-23-31-02-407130 200000