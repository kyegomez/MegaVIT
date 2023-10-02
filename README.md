[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# MegaVit
The open source implementation of the model from "Scaling Vision Transformers to 22 Billion Parameters"



[Paper Link](https://arxiv.org/pdf/2302.05442.pdf)

# Appreciation
* Lucidrains
* Agorians



# Install
`pip install mega-vit`

# Usage
- Simple usage,
```python
import torch
from mega_vit.main import MegaVit

v = MegaVit(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(1, 3, 256, 256)

preds = v(img) # (1, 1000)
print(preds)
```

- Hyperparams as stated in paper:
```python
import torch
from mega_vit.main import MegaVit

v = ViT(
    image_size = 224,
    patch_size = 14,
    num_classes = 1000,
    dim = 6144,
    depth = 48,
    heads = 48,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(1, 3, 224, 224)

preds = v(img) # (1, 1000)
print(preds)
```

# Model Architecture
- Regular vit with new parallel layers, QK(Query/Key)Normalization, and omitted biases.

----
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
```
@misc{2302.05442,
Author = {Mostafa Dehghani and Josip Djolonga and Basil Mustafa and Piotr Padlewski and Jonathan Heek and Justin Gilmer and Andreas Steiner and Mathilde Caron and Robert Geirhos and Ibrahim Alabdulmohsin and Rodolphe Jenatton and Lucas Beyer and Michael Tschannen and Anurag Arnab and Xiao Wang and Carlos Riquelme and Matthias Minderer and Joan Puigcerver and Utku Evci and Manoj Kumar and Sjoerd van Steenkiste and Gamaleldin F. Elsayed and Aravindh Mahendran and Fisher Yu and Avital Oliver and Fantine Huot and Jasmijn Bastings and Mark Patrick Collier and Alexey Gritsenko and Vighnesh Birodkar and Cristina Vasconcelos and Yi Tay and Thomas Mensink and Alexander Kolesnikov and Filip Pavetić and Dustin Tran and Thomas Kipf and Mario Lučić and Xiaohua Zhai and Daniel Keysers and Jeremiah Harmsen and Neil Houlsby},
Title = {Scaling Vision Transformers to 22 Billion Parameters},
Year = {2023},
Eprint = {arXiv:2302.05442},
}
```

# Todo
- [ ] Add flash attention, with layernorm before attn, and then layernom for qk values,
- [ ] Basic training script on CIFAR,
- [ ] When using ViT-22B, similar to any large scale model, it is difficult to understand how the model arrived at a specific decision, which could lead to lack of
trust and accountability. Add in a mechanism to backtrack
- [ ] create logic to train the decoder for 300k steps with a batch size of 64 using Adam (Kingma and Ba, 2015) and clip the gradients to a global norm value of 0.05 to stabilize training. We linearly increase the learning rate for 2500 steps to 0.0002 (starting from 0) and then decay the learning rate with a cosine schedule (Loshchilov and Hutter, 2017) back to 0.