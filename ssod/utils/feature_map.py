import torch
import torch.nn.functional as F


def mlvl_feature_map_project(mlvl_feature_maps, homography_matrix_list, ori_shapes, img_shapes):
    thetas = []
    dtype = mlvl_feature_maps[0].dtype
    for homography_matrix, ori_shape, img_shape in zip(homography_matrix_list, ori_shapes, img_shapes):
        h1, w1 = ori_shape
        h2, w2 = img_shape
        m1 = homography_matrix.new_tensor([[2.0 / w1, 0.0, -1.0],
                                           [0.0, 2.0 / h1, -1.0],
                                           [0.0, 0.0, 1.0]])
        m2 = homography_matrix.new_tensor([[2.0 / w2, 0.0, -1.0],
                                           [0.0, 2.0 / h2, -1.0],
                                           [0.0, 0.0, 1.0]])
        theta = (m2 @ homography_matrix @ m1.inverse()).inverse()
        thetas.append(theta)
    theta = torch.stack(thetas)[:, :2, :]
    projected_feature_maps = []
    for feature_map in mlvl_feature_maps:
        grid = F.affine_grid(theta, feature_map.size(), align_corners=True)
        maps = F.grid_sample(feature_map.float(), grid, mode='nearest', align_corners=True)
        projected_feature_maps.append(maps.to(dtype))
    return projected_feature_maps
