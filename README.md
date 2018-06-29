# Multi-Channel Normalized Cross-Correlation

This is the MATLAB implementation from our BMVC 2017 [Cross-Domain Forensic Shoeprint Matching](http://vision.ics.uci.edu/papers/KongSRF_BMVC_2017/KongSRF_BMVC_2017.pdf) and arXiv [Cross-Domain Image Matching with Deep Feature Maps](https://arxiv.org/abs/1804.02367) submissions.

## Getting Started
- Clone this repo:
```bash
git clone --recurse-submodules https://github.com/bkong/MCNCC
```
- Follow the instructions at http://www.vlfeat.org/matconvnet/install/ to install MatConvNet.

## Alignment Search Matching
- Download the dataset (e.g., fid300)
```bash
bash scripts/download_dataset.sh fid300
```
- Startup MATLAB
```bash
matlab
```
- Extract the ResNet-50 res2bx features by running the appropriate feature extraction function
```
>> gen_resnetfeats_fid300(2)
```
- Compute the MCNCC scores
```
>> alignment_search_eval_fid300(1:300, 2)
```
```1:300``` specifies which cropped crime scene images to evaluate against the reference images of FID-300. Because this is a slow process, you can evaluate just a subset of the crime scene images. Alternatively, you can manually distribute the workload by specifying different subsets on different machines/GPUs to accelerate the task.
- Generate a CMC plot comparing the MCNCC against the baselines
```
>> baseline_comparison_cmc_fid(2)
```
## No-alignment Search Matching
- Download the dataset (e.g., facades)
```bash
bash scripts/download_dataset.sh facades
```
- Startup MATLAB
```bash
matlab
```
- Extract the ResNet-50 res2bx features by running the appropriate feature extraction function
```
>> gen_resnetfeats_facades(2)
```
- Compute the MCNCC scores
```
>> no_search_eval_facades(2, 'mcncc')
```
```'mcncc'``` can be changed to any of these values ```{'cosine', 'euclidean', '3dncc', 'mcncc'}```
- Generate a CMC plot comparing the four correlation/distance metrics on the training set
```
>> baseline_comparison_cmc_facades(2, true)
```
The second parameter can be set to ```false``` to generate a CMC plot on the testing set.

## Citation
If you use this code for your research, please cite our paper.
```
@inproceedings{KongSRF_BMVC_2017,
    author = "Kong, Bailey and Supan{\vc}i{\vc}, James Steven and Ramanan, Deva and Fowlkes, Charless C.",
    title = "Cross-Domain Forensic Shoeprint Matching",
    booktitle = "British Machine Vision Conference (BMVC)",
    year = "2017"
}
```
