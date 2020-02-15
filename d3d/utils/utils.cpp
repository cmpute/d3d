#include <unordered_map>
#include <array>
#include <tuple>

#include <torch/extension.h>
#include <torch/types.h>
#include <iostream>

using namespace std;
using namespace at;

typedef tuple<int, int, int> coord_t;

struct coord_hash : public std::unary_function<coord_t, std::size_t>
{
    std::size_t operator()(const coord_t& k) const
    {
        constexpr int p = 997; // match this to the range of k
        size_t ret = std::get<0>(k);
        ret = ret * p + std::get<1>(k);
        ret = ret * p + std::get<2>(k);
        return ret;
    }
};
 
struct coord_equal : public std::binary_function<coord_t, coord_t, bool>
{
    bool operator()(const coord_t& v0, const coord_t& v1) const
    {
        return (
            std::get<0>(v0) == std::get<0>(v1) &&
            std::get<1>(v0) == std::get<1>(v1) &&
            std::get<2>(v0) == std::get<2>(v1)
        );
    }
};

typedef unordered_map<coord_t, int, coord_hash, coord_equal> coord_m; // XXX: Consider use robin-map
 
/*
 * Voxelize point cloud
 * 
 * \param[in] points Original point cloud with certain features. The first three columns have to be xyz coordinates
 * \param[in] voxel_shape The shape of the voxel grid
 * \param[in] voxel_bound The boundary of the voxel grid, in format [xmin, xmax, ymin, ymax, zmin, zmax]
 * \param[in] max_points Maximum number of points per voxel
 * \param[in] max_voxels Maximum number of voxels
 * \param[out] voxels Generated features for each voxel
 * \param[out] coords Calculated coordinates for generated voxels
 * \param[out] aggregates Aggregated feature of each voxel, only applicable when ReductionType != 0
 * \param[out] voxel_pmask Point mask of each voxel
 * \param[out] voxel_npoints Number of points in each voxel (Note: this could be larger than max_npoints)
 */
template <int ReductionType>
int voxelize_3d_templated(
    const Tensor points, const Tensor voxel_shape, const Tensor voxel_bound,
    const int max_points, const int max_voxels,
    Tensor voxels, Tensor coords, Tensor aggregates,
    Tensor voxel_pmask, Tensor voxel_npoints
)
{
    auto points_ = points.accessor<float, 2>();
    auto voxel_shape_ = voxel_shape.accessor<int, 1>();
    auto voxel_bound_ = voxel_bound.accessor<float, 1>();
    auto voxels_ = voxels.accessor<float, 3>();
    auto coords_ = coords.accessor<int, 2>();
    auto aggregates_ = aggregates.accessor<float, 2>();
    auto voxel_pmask_ = voxel_pmask.accessor<bool, 2>();
    auto voxel_npoints_ = voxel_npoints.accessor<int, 1>();

    // calculate voxel sizes and other variables
    float voxel_size_[3];
    for (int d = 0; d < 3; ++d)
        voxel_size_[d] = (voxel_bound_[(d<<1) | 1] - voxel_bound_[d<<1]) / voxel_shape_[d];
    auto npoints = points.sizes().at(0);
    auto nfeatures = points.sizes().at(1);

    coord_m voxel_idmap;
    int coord_temp[3];
    int nvoxels = 0;
    for (int i = 0; i < npoints; ++i)
    {
        // calculate voxel-wise coordinates
        bool out_of_range = false;
        for (int d = 0; d < 3; ++d)
        {
            int idx = int((points_[i][d] - voxel_bound_[d << 1]) / voxel_size_[d]);
            if (idx < 0 || idx >= voxel_shape_[d])
            {
                out_of_range = true;
                break;
            }
            coord_temp[d] = idx;
        }
        if (out_of_range) continue;

        // assign voxel
        int voxel_idx;
        auto coord_tuple = make_tuple(coord_temp[0], coord_temp[1], coord_temp[2]);
        auto voxel_iter = voxel_idmap.find(coord_tuple);
        if (voxel_iter == voxel_idmap.end())
        {
            if (nvoxels >= max_voxels)
                continue;

            voxel_idx = nvoxels++;
            voxel_idmap[coord_tuple] = voxel_idx;
            for (int d = 0; d < 3; ++d)
                coords_[voxel_idx][d] = coord_temp[d];
        }
        else
            voxel_idx = voxel_iter->second;

        // assign voxel mask and original features
        int n = voxel_npoints_[voxel_idx]++;
        if (n < max_points)
        {
            voxel_pmask_[voxel_idx][n] = true;
            for (int d = 0; d < nfeatures; ++d)
                voxels_[voxel_idx][n][d] = points_[i][d];
        }

        // reduce features
        for (int d = 0; d < nfeatures; ++d)
        {
            switch (ReductionType)
            {
            case 1: // mean
                aggregates_[voxel_idx][d] += points_[i][d];
                break;

            case 2: // max, aggregates should be initialized as positive huge numbers
                aggregates_[voxel_idx][d] = max(aggregates_[voxel_idx][d], points_[i][d]);
                break;

            case 3: // min, aggregates should be initialized as negative huge numbers
                aggregates_[voxel_idx][d] = min(aggregates_[voxel_idx][d], points_[i][d]);
                break;
            
            case 0:
            default:
                break;
            }
        }
    }

    // if aggregate is mean, divide them by the numbers now
    if (ReductionType == 1)
        for (int i = 0; i < nvoxels; i++)
            for (int d = 0; d < nfeatures; d++)
                aggregates_[i][d] /= voxel_npoints_[i];

    return nvoxels;
}

/*
 * This function process constant parameters input and call corresponding implementations
 */
int voxelize_3d(
    const Tensor points, const Tensor voxel_shape, const Tensor voxel_bound,
    const int max_points, const int max_voxels, const int reduction_type,
    Tensor voxels, Tensor coords, Tensor aggregates,
    Tensor voxel_pmask, Tensor voxel_npoints
)
{
    switch (reduction_type)
    {
    case 0:
        return voxelize_3d_templated<0>(points, voxel_shape, voxel_bound, max_points, max_voxels,
            voxels, coords, aggregates, voxel_pmask, voxel_npoints);
    case 1:
        return voxelize_3d_templated<1>(points, voxel_shape, voxel_bound, max_points, max_voxels,
            voxels, coords, aggregates, voxel_pmask, voxel_npoints);
    case 2:
        return voxelize_3d_templated<2>(points, voxel_shape, voxel_bound, max_points, max_voxels,
            voxels, coords, aggregates, voxel_pmask, voxel_npoints);
    case 3:
        return voxelize_3d_templated<3>(points, voxel_shape, voxel_bound, max_points, max_voxels,
            voxels, coords, aggregates, voxel_pmask, voxel_npoints);
    default:
        return 0;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("voxelize_3d", &voxelize_3d, "3D voxelization of tensor");
}
