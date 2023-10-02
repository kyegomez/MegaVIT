[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# MegaVit
A simple implementation of a CLIP that splits up an image into quandrants and then gets the embeddings for each quandrant


[Paper Link](https://arxiv.org/pdf/2302.05442.pdf)

# Appreciation
* Lucidrains
* Agorians



# Install

# Usage

# Architecture

# Dataset Strategy
The paper trains ViT-22B on a version of the JFT dataset that has been extended to around 4 billion images. JFT is a large-scale dataset scraped from the internet, originally containing over 300 million images labeled with a hierarchical taxonomy of 30,000 categories. 

The authors do not provide full details on how the dataset was extended from the original JFT to 4 billion images. However, the goal seems to be creating a larger and more diverse training set to support scaling up the model size. Pre-training on larger datasets enables learning more robust and generalizable visual representations.

The authors evaluate ViT-22B on a comprehensive set of 39 datasets covering various domains like image classification, dense prediction tasks, video, and fairness benchmarks. Using such a diverse evaluation suite allows them to thoroughly assess the scalability and transferability of ViT-22B across different domains and data distributions.

Below is a table summarizing some of the key datasets used in the paper:

| Dataset | Domain | Images | Classes |
|-|-|-|-| 
| JFT (training set) | Internet images | ~4 billion | 30,000 |
| ImageNet | Natural images | 1.28M | 1000 |
| ImageNet-C | Corrupted ImageNet images | 1.28M | 1000 |  
| ImageNet-R | Hard ImageNet images | 30K | 200 |
| ImageNet-A | Adversarial ImageNet images | 7.5K | 200 |
| ObjectNet | Natural images | 113K | 113 |
| Cifar-10 | Tiny natural images | 60K | 10 |
| Cifar-100 | Tiny natural images | 60K | 100 | 
| ADE20K | Scene parsing | 25K | 150 |
| Kinetics-400 | Human action videos | 400K | 400 |
| CelebA | Celeb faces | 202K | 40 |


# License
MIT

# Citations

