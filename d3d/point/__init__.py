import torch

def aligned_scatter(coordinates, feature_map, method="drop"):
    '''
    Gather the values given coordinates in feature_map.
    
    :param feature_map: should have shape B x C x D1 x D2... x Dm, the number of remaining dimensions (m) should be consistent with coordinate dimension
    :param coordinates: should have shape N x (m+1), it should contains batch index at the first dimension
    :param method:
        drop - directly convert decimal coordinates to integers;
        mean - mean of surrounding values
        weighted - bilinear interpolation of surrounding values
        max - maximum of surrounding values
    :return: extracted features with shape N x C
    '''
    if method == "drop":
        coordinates = coordinates.long()
        _, ndim = coordinates.shape
        assert len(feature_map.shape) == ndim + 1

        indexing = (coordinates[:, 0], slice(None)) + tuple(coordinates[:, i] for i in range(1, ndim))
        return feature_map[indexing]
    else:
        raise NotImplementedError()
