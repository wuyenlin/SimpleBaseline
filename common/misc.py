import torch
import numpy as np


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)



def vectorize(gt_3d) -> torch.tensor:
    """
    process gt_3d (17,3) into a (16,4) that contains bone vector and length
    :return bone_info: [unit bone vector (,3) + bone length (,1)]
    """
    indices = (
        (0,7), (7,8), (8,9), (9,10),  # spine + head
        (8,11), (11,12), (12,13), 
        (8,14), (14,15), (15,16), # arms
        (0,4), (4,5), (5,6),
        (0,1), (1,2), (2,3), # legs
    )

    num_bones = len(indices)
    gt_3d_tensor = gt_3d if torch.is_tensor(gt_3d) \
                    else torch.from_numpy(gt_3d)

    bone_info = torch.zeros([num_bones, 4], requires_grad=False) # (16, 4)
    for i in range(num_bones):
        vec = gt_3d_tensor[indices[i][1],:] - gt_3d_tensor[indices[i][0],:]
        vec_len = torch.linalg.norm(vec)
        unit_vec = vec/vec_len
        bone_info[i,:3], bone_info[i,3] = unit_vec, vec_len
    return bone_info
