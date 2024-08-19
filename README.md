# Hierarchical Conditioning of Diffusion Models Using Tree-of-Life for Studying Species Evolution (ECCV 2024)
This repo is the official implementation of "Hierarchical Conditioning of Diffusion Models Using Tree-of-Life for Studying Species Evolution"


Accepted at ECCV 2024. [Project Page](https://imageomics.github.io/phylo-diffusion/) !

## Requirements
A suitable [conda](https://conda.io/) environment named `phylo_diffusion` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate phylo_diffusion
```

## Model Training
For training the models, we can use the following command:
```
python main.py --name <mddel_name> --logdir <path_to_logdir> --base <yaml_config_path> --postfix <file_postfix_name> -t True --gpus <comma-separated GPU indices>
```

We first need to train the base autoencoder model. This can be trained using the following command:

## Sampling Images

```
python scripts/trait_masking.py --config_path <path_to_config_file> --ckpt_path <path_to_saved_model> --node_dict <path_to_hierarchical_node_dict> --output_dir_name <output_dir_name>
```

## Citation
Our paper:

```
@article{khurana2024hierarchical,
  title={Hierarchical Conditioning of Diffusion Models Using Tree-of-Life for Studying Species Evolution},
  author={Khurana, Mridul and Daw, Arka and Maruf, M and Uyeda, Josef C and Dahdul, Wasila and Charpentier, Caleb and Bak{\i}{\c{s}}, Yasin and Bart Jr, Henry L and Mabee, Paula M and Lapp, Hilmar and others},
  journal={arXiv preprint arXiv:2408.00160},
  year={2024}
}
```

### Acknowledgments
The code base is borrowed from the original implementation of Latent Diffusion Models [1] available at [https://github.com/CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion). Also consider citing LDM.

### References

[1] Rombach, Robin, et al. "High-resolution image synthesis with latent diffusion models." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.
