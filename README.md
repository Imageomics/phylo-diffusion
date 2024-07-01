# HIER-Embed
Code Repository for HIER-Embed - ECCV submission

## Requirements
A suitable [conda](https://conda.io/) environment named `hier_embed` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate hier_embed
```

## Model Training
For training the models, we can use the following command:
```
python main.py --name <mddel_name> --logdir <path_to_logdir> --base <yaml_config_path> --postfix <file_postfix_name> -t True --gpus <comma-separated GPU indices>
```

We first need to train the base autoencoder model. This can be trained using the following command:

## Sampling Images

## Acknowledgments
The code base is heavily borrowed from the original implementation of Latent Diffusion Models [1] available at [https://github.com/CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion)

### References

[1] Rombach, Robin, et al. "High-resolution image synthesis with latent diffusion models." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.