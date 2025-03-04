import math
import torch.nn.functional as F


def mlvl_feature_map_project(mlvl_feature_maps,
                             feat_scales,
                             homography_matrix_list,
                             ori_shapes,
                             img_shapes,
                             batch_input_shape):
    assert len(mlvl_feature_maps) == len(feat_scales)
    num_images = mlvl_feature_maps[0].shape[0]
    dtype = mlvl_feature_maps[0].dtype

    thetas = []
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
        thetas.append(theta[:2, :])

    mlvl_projected_maps = []
    for feature_maps, feat_scale in zip(mlvl_feature_maps, feat_scales):
        batch_h = math.ceil(batch_input_shape[0] / feat_scale)
        batch_w = math.ceil(batch_input_shape[1] / feat_scale)
        feat_dim = feature_maps.size(1)

        projected_maps = feature_maps.new_zeros((num_images, feat_dim, batch_h, batch_w))
        for img_id in range(num_images):
            feat_h1 = math.ceil(ori_shapes[img_id][0] // feat_scale)
            feat_w1 = math.ceil(ori_shapes[img_id][1] // feat_scale)
            feat_h2 = math.ceil(img_shapes[img_id][0] // feat_scale)
            feat_w2 = math.ceil(img_shapes[img_id][1] // feat_scale)
            feat1 = feature_maps[img_id, :, :feat_h1, :feat_w1].unsqueeze(0)
            grid = F.affine_grid(thetas[img_id].unsqueeze(0), [1, feat_dim, feat_h2, feat_w2], align_corners=True)
            feat2 = F.grid_sample(feat1.float(), grid, mode='nearest', align_corners=True)
            projected_maps[img_id, :, :feat_h2, :feat_w2] = feat2[0].type(dtype)
        mlvl_projected_maps.append(projected_maps)
    return mlvl_projected_maps
