
import torch

EPS=1e-10

def get_CosineDistance_matrix(features):
    if features.dim() >2:
        features = features.reshape(features.shape[0], -1)

    features_norm = features / (EPS + features.norm(dim=1)[:, None])
    ans = torch.mm(features_norm, features_norm.transpose(0,1))

    # We want distance, not similarity.
    ans = torch.add(-ans, 1.)

    return ans

def aggregatefrom_specimen_to_species(sorted_class_names_according_to_class_indx, specimen_distance_matrix, z_size, channels):
    unique_sorted_class_names_according_to_class_indx = sorted(set(sorted_class_names_according_to_class_indx))

    species_dist_matrix = torch.zeros(len(unique_sorted_class_names_according_to_class_indx), channels, z_size, z_size)
    for indx_i, i in enumerate(unique_sorted_class_names_according_to_class_indx):
        class_i_indices = [idx for idx, element in enumerate(sorted_class_names_according_to_class_indx) if element == i] 
        species_dist_matrix[indx_i] = torch.mean(specimen_distance_matrix[class_i_indices,:], dim=0, keepdim=True)
    
    return species_dist_matrix