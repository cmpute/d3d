#include "d3d/voxel/voxelize.h"

#include <unordered_map>
#include <array>
#include <tuple>
#include <vector>
#include <iostream>
#include <limits>

using namespace std;
using namespace at;
using namespace pybind11::literals;

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


template <int ReductionType>
py::dict voxelize_3d_templated(
    const Tensor points, const Tensor voxel_shape, const Tensor voxel_bound,
    const int max_points, const int max_voxels
)
{

    auto points_ = points.accessor<float, 2>();
    auto voxel_shape_ = voxel_shape.accessor<int, 1>();
    auto voxel_bound_ = voxel_bound.accessor<float, 1>();

    Tensor voxels = torch::zeros({max_voxels, max_points, points_.size(1)}, torch::dtype(torch::kFloat32));
    Tensor coords = torch::empty({max_voxels, 3}, torch::dtype(torch::kLong));
    Tensor voxel_pmask = torch::empty({max_voxels, max_points}, torch::dtype(torch::kBool));
    Tensor voxel_npoints = torch::zeros(max_voxels, torch::dtype(torch::kInt32));
    auto voxels_ = voxels.accessor<float, 3>();
    auto coords_ = coords.accessor<int64_t, 2>();
    auto voxel_pmask_ = voxel_pmask.accessor<bool, 2>();
    auto voxel_npoints_ = voxel_npoints.accessor<int, 1>();

    // initialize aggregated values
    Tensor aggregates;
    switch(ReductionType)
    {
        case 0:
            aggregates = torch::empty({0, 0});
            break;
        case 1:
            aggregates = torch::zeros({max_voxels, points_.size(1)}, torch::dtype(torch::kFloat32));
            break;
        case 2:
            aggregates = torch::full({max_voxels, points_.size(1)}, -numeric_limits<float>::infinity(), torch::dtype(torch::kFloat32));
            break;
        case 3:
            aggregates = torch::full({max_voxels, points_.size(1)}, numeric_limits<float>::infinity(), torch::dtype(torch::kFloat32));
            break;
    }
    auto aggregates_ = aggregates.accessor<float, 2>();

    // calculate voxel sizes and other variables
    float voxel_size_[3];
    for (int d = 0; d < 3; ++d)
        voxel_size_[d] = (voxel_bound_[(d<<1) | 1] - voxel_bound_[d<<1]) / voxel_shape_[d];
    auto npoints = points_.size(0);
    auto nfeatures = points_.size(1);

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

    // remove unused voxel entries
    if (ReductionType > 0)
        return py::dict("voxels"_a=voxels.slice(0, 0, nvoxels),
            "coords"_a=coords.slice(0, 0, nvoxels),
            "voxel_pmask"_a=voxel_pmask.slice(0, 0, nvoxels),
            "voxel_npoints"_a=voxel_npoints.slice(0, 0, nvoxels),
            "aggregates"_a=aggregates.slice(0, 0, nvoxels)
        );
    else
        return py::dict("voxels"_a=voxels.slice(0, 0, nvoxels),
            "coords"_a=coords.slice(0, 0, nvoxels),
            "voxel_pmask"_a=voxel_pmask.slice(0, 0, nvoxels),
            "voxel_npoints"_a=voxel_npoints.slice(0, 0, nvoxels)
        );
}

py::dict voxelize_3d(
    const Tensor points, const Tensor voxel_shape, const Tensor voxel_bound,
    const int max_points, const int max_voxels, const int reduction_type
)
{
    #define _VOXELIZE_3D_REDUCTION_TYPE_CASE(n) case n: \
        return voxelize_3d_templated<n>(points, voxel_shape, voxel_bound, max_points, max_voxels)
    switch (reduction_type)
    {
    _VOXELIZE_3D_REDUCTION_TYPE_CASE(0);
    _VOXELIZE_3D_REDUCTION_TYPE_CASE(1);
    _VOXELIZE_3D_REDUCTION_TYPE_CASE(2);
    _VOXELIZE_3D_REDUCTION_TYPE_CASE(3);
    default:
        throw py::value_error("Unsupported reduction type in voxelization!");
    }
    #undef _VOXELIZE_3D_REDUCTION_TYPE_CASE
}

py::dict voxelize_3d_sparse(
    const Tensor points, const Tensor voxel_shape, const Tensor voxel_bound,
    const torch::optional<int> max_points, const torch::optional<int> max_voxels
)
{
    auto points_ = points.accessor<float, 2>();
    auto voxel_shape_ = voxel_shape.accessor<int, 1>();
    auto voxel_bound_ = voxel_bound.accessor<float, 1>();

    // fill points mapping with -1 (unassigned)
    Tensor points_mapping = torch::full(points_.size(0), -1, torch::dtype(torch::kLong));
    auto points_mapping_ = points_mapping.accessor<int64_t, 1>();

    // calculate voxel sizes and other variables
    float voxel_size_[3];
    for (int d = 0; d < 3; ++d)
        voxel_size_[d] = (voxel_bound_[(d<<1) | 1] - voxel_bound_[d<<1]) / voxel_shape_[d];
    auto npoints = points_.size(0);
    auto nfeatures = points_.size(1);

    coord_m voxel_idmap;
    vector<int> voxel_npoints;
    vector<Tensor> coords_list;
    int coord_temp[3];
    int nvoxels = 0;
    bool need_mask = false;
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
        if (out_of_range)
        {
            need_mask = true;
            continue;
        }

        // assign voxel
        int voxel_idx;
        auto coord_tuple = make_tuple(coord_temp[0], coord_temp[1], coord_temp[2]);
        auto voxel_iter = voxel_idmap.find(coord_tuple);
        if (voxel_iter == voxel_idmap.end())
        {
            voxel_idx = nvoxels++;
            voxel_idmap[coord_tuple] = voxel_idx;
            voxel_npoints.push_back(1);
            coords_list.push_back(torch::tensor({coord_temp[0], coord_temp[1], coord_temp[2]}, torch::kLong));
        }
        else
        {
            voxel_idx = voxel_iter->second;
            voxel_npoints[voxel_idx] += 1;
        }
        points_mapping_[i] = voxel_idx;
    }

    // process mask
    Tensor points_filtered = points;
    Tensor masked;
    if (need_mask)
    {
        masked = torch::where(points_mapping >= 0)[0];
        points_filtered = torch::index_select(points, 0, masked);
        points_mapping = torch::index_select(points_mapping, 0, masked);
    }
    else
    {
        masked = torch::arange(npoints);
    }

    return py::dict(
        "points"_a=points_filtered,
        "points_mask"_a=masked,
        "points_mapping"_a=points_mapping,
        "coords"_a=torch::stack(coords_list),
        "voxel_npoints"_a=torch::tensor(voxel_npoints, torch::kInt32));
}
