import argparse, os, sys, glob
import torch
import pickle
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

# import colored_traceback.always


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def masking_embed(embedding, levels=1, embed_dim_to_mask=128):
    """
    size of embedding - nx1xd, n: number of samples, d - 512
    replacing the last 128*levels from the embedding
    """
    replace_size = embed_dim_to_mask*levels
    random_noise = torch.randn(embedding.shape[0], embedding.shape[1], replace_size)
    embedding[:, :, -replace_size:] = random_noise
    return embedding

def reorganize_dict_based_on_index(original_dict, index):
    new_dict = {}
    # if not (0 <= index < 4):
    #     return "Index out of range. Please provide an index between 0 and 3."

    for key, value in original_dict.items():
        try:
            selected_num = value[0].split(',')[index].strip()
            selected_num = int(selected_num)
        except IndexError:
            return "Index out of range for the data provided."

        if selected_num in new_dict:
            new_dict[selected_num].append(key)
        else:
            new_dict[selected_num] = [key]

    return new_dict


def sampling(model, sampler, list_of_codes, n_samples=1, scale=1.0, masked=False, levels=1, ancestry_level=1, num_levels=4):

    for node_representation in tqdm(list_of_codes):
        prompt = node_representation
        all_samples=list()
        with torch.no_grad():
            with model.ema_scope():
                uc = None
                if scale != 1.0:
                    uc = model.get_learned_conditioning(n_samples * [""])
                for n in trange(opt.n_iter, desc="Sampling"):
                    xc = n_samples * (prompt)
                    xc = [tuple(xc)]
                    print(prompt)
                    c = model.get_learned_conditioning({'class_to_node': xc})
                    embed_dim_to_mask = int(c.shape[-1]/num_levels)
                    if masked:
                        c = masking_embed(c, levels, embed_dim_to_mask)
                        print(f"**** Masked {levels} levels from last ****")
                    shape = [3, 64, 64]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                    conditioning=c,
                                                    batch_size=n_samples,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=uc,
                                                    eta=opt.ddim_eta)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                    all_samples.append(x_samples_ddim)
                    
        # individual images
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')


        class_number = int(prompt[0].split(',')[ancestry_level-1])
        class_name = f'class_{class_number}'
        save_dir_path = os.path.join(sample_path, f'level{ancestry_level}', class_name)
        os.makedirs(save_dir_path, exist_ok=True)
        for i in range(opt.n_samples):
            sample = grid[i]
            img = 255. * rearrange(sample, 'c h w -> h w c').cpu().numpy()
            img_arr = img.astype(np.uint8)
            Image.fromarray(img_arr).save(f'{save_dir_path}/{class_name}_img_{i}.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "--prompt",
    #     type=str,
    #     nargs="?",
    #     default="a painting of a virus monster playing guitar",
    #     help="the prompt to render"
    # )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        # default=0.0,
        default=1.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    # parser.add_argument(
    #     "--n_iter",
    #     type=int,
    #     default=1,
    #     help="sample this often",
    # )

    # parser.add_argument(
    #     "--H",
    #     type=int,
    #     default=256,
    #     help="image height, in pixel space",
    # )

    # parser.add_argument(
    #     "--W",
    #     type=int,
    #     default=256,
    #     help="image width, in pixel space",
    # )

    parser.add_argument(
        "--n_samples",
        type=int,
        # default=4,
        default=1,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--output_dir_name",
        type=str,
        default='default_file',
        help="name of folder",
    )

    parser.add_argument(
        "--postfix",
        type=str,
        default='',
        help="name of folder",
    )

    # parser.add_argument(
    #     "--levels",
    #     type=int,
    #     default=1,
    #     help="haw many levels to mask going from 4 to 1",
    # )

    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--config_path", 
        type=str,
        default='/fastscratch/mridul/new_diffusion_models/ldm/cub_conditioning/2024-01-31T21-25-36_HLE_f4_epoch185_batch8_lr5e-6/configs/2024-01-31T21-25-36-project.yaml',
        help="Path to config_file",
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default='/fastscratch/mridul/new_diffusion_models/ldm/cub_conditioning/2024-01-31T21-25-36_HLE_f4_epoch185_batch8_lr5e-6/checkpoints/epoch=000113.ckpt',
        help="Path to ckpt_path",
    )

    parser.add_argument(
        "--node_dict",
        type=str,
        default='/projects/ml4science/mridul/ldm/data/cub/cub_class_to_ancestral_level.pkl',
        help="Path to ckpt_path",
    )

    opt = parser.parse_args()


    with open(opt.node_dict, 'rb') as pickle_file:
        class_to_node_dict = pickle.load(pickle_file)

    # sorted the dictionary according to key
    class_to_node_dict = {key: class_to_node_dict[key] for key in sorted(class_to_node_dict)}

    # get number of discrete phylogenetic levels
    num_levels = len(list(class_to_node_dict.values())[0][0].split(','))

    # for each ancestral levels, create speceis groupings at each ancestral level
    ancestral_level_species_grouping = []
    for i in range(num_levels-1):
        ancestral_level_species_grouping.append(reorganize_dict_based_on_index(class_to_node_dict, index=i))

    def get_ancester_hier_embed_list(ancestor_dict):
        ancestor_species_names = [value[0] for _, value in ancestor_dict.items()]
        ancestor_list = [class_to_node_dict[x] for x in ancestor_species_names]
        return ancestor_list

    # Get list of ancestral level HIER-Embed nodes at each level for species to be generated after masking
    ancestral_level_hier_nodes = []
    for ancestor in ancestral_level_species_grouping:
        ancestral_level_hier_nodes.append(get_ancester_hier_embed_list(ancestor))


    # Load the config and the model
    config = OmegaConf.load(opt.config_path)
    model = load_model_from_config(config, opt.ckpt_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    sample_path = os.path.join(outpath, opt.output_dir_name)
    os.makedirs(sample_path, exist_ok=True)

    # Sample for Trait Masking for each level.
    for i in range(num_levels-1):
        ancestral_lvl = num_levels-1-i
        sampling(model, sampler, ancestral_level_hier_nodes[ancestral_lvl-1], opt.n_samples, masked=True, levels=i+1, ancestry_level=ancestral_lvl, num_levels=num_levels)

