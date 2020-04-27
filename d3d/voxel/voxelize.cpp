#include "d3d/voxel/voxelize.h"

#include <unordered_map>
#include <unordered_set>
#include <array>
#include <tuple>
#include <vector>
#include <iostream>
#include <limits>

using namespace std;
using namespace at;
using namespace pybind11::literals;
template <typename T> using toptional = torch::optional<T>;

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


template <ReductionType Reduction>
py::dict voxelize_3d_dense_templated(
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
    switch(Reduction)
    {
        case ReductionType::NONE:
            aggregates = torch::empty({0, 0});
            break;
        case ReductionType::MEAN:
            aggregates = torch::zeros({max_voxels, points_.size(1)}, torch::dtype(torch::kFloat32));
            break;
        case ReductionType::MAX:
            aggregates = torch::full({max_voxels, points_.size(1)}, -numeric_limits<float>::infinity(), torch::dtype(torch::kFloat32));
            break;
        case ReductionType::MIN:
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
            switch (Reduction)
            {
            case ReductionType::MEAN: // mean
                aggregates_[voxel_idx][d] += points_[i][d];
                break;

            case ReductionType::MAX: // max, aggregates should be initialized as positive huge numbers
                aggregates_[voxel_idx][d] = max(aggregates_[voxel_idx][d], points_[i][d]);
                break;

            case ReductionType::MIN: // min, aggregates should be initialized as negative huge numbers
                aggregates_[voxel_idx][d] = min(aggregates_[voxel_idx][d], points_[i][d]);
                break;
            
            case ReductionType::NONE:
            default:
                break;
            }
        }
    }

    // if aggregate is mean, divide them by the numbers now
    if (Reduction == ReductionType::MEAN)
        for (int i = 0; i < nvoxels; i++)
            for (int d = 0; d < nfeatures; d++)
                aggregates_[i][d] /= voxel_npoints_[i];

    // remove unused voxel entries
    if (Reduction != ReductionType::NONE)
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

py::dict voxelize_3d_dense(
    const Tensor points, const Tensor voxel_shape, const Tensor voxel_bound,
    const int max_points, const int max_voxels, const ReductionType reduction_type
)
{
    #define _VOXELIZE_3D_REDUCTION_TYPE_CASE(n) case n: \
        return voxelize_3d_dense_templated<n>(points, voxel_shape, voxel_bound, max_points, max_voxels)
    switch (reduction_type)
    {
    _VOXELIZE_3D_REDUCTION_TYPE_CASE(ReductionType::NONE);
    _VOXELIZE_3D_REDUCTION_TYPE_CASE(ReductionType::MEAN);
    _VOXELIZE_3D_REDUCTION_TYPE_CASE(ReductionType::MIN);
    _VOXELIZE_3D_REDUCTION_TYPE_CASE(ReductionType::MAX);
    default:
        throw py::value_error("Unsupported reduction type in voxelization!");
    }
    #undef _VOXELIZE_3D_REDUCTION_TYPE_CASE
}

py::dict voxelize_3d_sparse(
    const Tensor points, const Tensor voxel_shape, const Tensor voxel_bound,
    const toptional<int> max_points, const toptional<int> max_voxels
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

py::dict voxelize_sparse(
    const Tensor points, const Tensor voxel_size, const toptional<int> ndim
)
{
    auto points_ = points.accessor<float, 2>();
    auto voxel_size_ = voxel_size.accessor<float, 1>();
    int ndim_ = ndim.value_or(3);

    auto npoints = points_.size(0);
    Tensor points_mapping = torch::empty(npoints, torch::dtype(torch::kLong));
    auto points_mapping_ = points_mapping.accessor<int64_t, 1>();

    coord_m voxel_idmap; // coord to voxel id
    vector<int> voxel_npoints; // number of points per voxel
    vector<Tensor> coords_list; // list of coordinates
    int coord_temp[ndim_]; // array to store temporary coordinate
    int nvoxels = 0; // number of total voxels
    for (int i = 0; i < npoints; ++i)
    {
        // calculate voxel-wise coordinates
        for (int d = 0; d < ndim_; ++d)
            coord_temp[d] = floor(points_[i][d] / voxel_size_[d]);

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

    return py::dict(
        "points_mapping"_a=points_mapping,
        "coords"_a=torch::stack(coords_list),
        "voxel_npoints"_a=torch::tensor(voxel_npoints, torch::kInt32)
    );
}

py::dict voxelize_filter(
    const Tensor feats, const Tensor points_mapping,
    const Tensor coords, const Tensor voxel_npoints, const toptional<Tensor> coords_bound,
    const toptional<int> min_points, const toptional<int> max_points, const toptional<int> max_voxels,
    const toptional<MaxPointsFilterType> max_points_filter, const toptional<MaxVoxelsFilterType> max_voxels_filter
)
{
    auto feats_ = feats.accessor<float, 2>();
    auto coords_ = coords.accessor<int64_t, 2>();
    auto voxel_npoints_ = voxel_npoints.accessor<int, 1>();
    auto points_mapping_ = points_mapping.accessor<int64_t, 1>();
    Tensor coord_bound_default = coords_bound.value_or(torch::empty({0, 0}, torch::dtype(torch::kFloat)));
    auto coords_bound_ = coord_bound_default.accessor<int64_t, 2>();
    int nvoxels = coords_.size(0);
    int ndim = coords_.size(1);
    int npoints = feats_.size(0);
    int nfeats = feats_.size(1);

    // default values
    MaxPointsFilterType max_points_filter_ = max_points_filter.value_or(MaxPointsFilterType::NONE);
    MaxVoxelsFilterType max_voxels_filter_ = max_voxels_filter.value_or(MaxVoxelsFilterType::NONE);
    if (max_voxels_filter != MaxVoxelsFilterType::NONE && !max_voxels.has_value())
        throw py::value_error("Must specify maximum voxel count to filter voxels!");
    int max_voxels_ = max_voxels.value();
    if (max_points_filter != MaxPointsFilterType::NONE && !max_points.has_value())
        throw py::value_error("Must specify maximum points per voxel to filter points!");
    int max_points_ = max_points.value();
    int min_points_ = min_points.value_or(0);

    // filter out voxels at first
    std::unordered_map<int, int> voxel_indices;
    int nvoxel_filtered = 0;
    auto test_bound = [&](int i) -> bool{
        bool out_of_range = false;
        for (int d = 0; d < ndim; d++)
            if (coords_[i][d] < coords_bound_[d][0] ||
                coords_[i][d] >= coords_bound_[d][1])
                {
                    out_of_range = true;
                    break;
                }
        return out_of_range;
    };
    switch (max_voxels_filter_)
    {
        case MaxVoxelsFilterType::NONE:
            for (int i = 0; i < nvoxels; i++)
            {
                if (voxel_npoints_[i] < min_points_) // skip voxels with few points
                    continue;
                if (coords_bound.has_value() && test_bound(i))
                    continue;
                voxel_indices[i] = nvoxel_filtered++;
            }
            break;
        case MaxVoxelsFilterType::TRIM:
            for (int i = 0; i < nvoxels; i++)
            {
                if (nvoxel_filtered >= max_voxels_)
                    break;
                if (voxel_npoints_[i] < min_points_) // skip voxels with few points
                    continue;
                if (coords_bound.has_value() && test_bound(i))
                    continue;
                voxel_indices[i] = nvoxel_filtered++;
            }
            break;
        case MaxVoxelsFilterType::DESCENDING:
            auto order = torch::argsort(voxel_npoints, 0, true);
            auto ordered_ = order.accessor<int, 1>();
            for (int i = 0; i < nvoxels; i++)
            {
                if (nvoxel_filtered >= max_voxels_)
                    break;
                if (voxel_npoints_[ordered_[i]] < min_points_)
                    break; // skip voxels with few points
                if (coords_bound.has_value() && test_bound(ordered_[i]))
                    continue;
                voxel_indices[ordered_[i]] = nvoxel_filtered++;
            }
            break;
    }

    // filter coordinates
    Tensor coords_filtered = torch::empty({nvoxel_filtered, ndim}, coords.options());
    auto coords_filtered_ = coords_filtered.accessor<int64_t, 2>();
    for (auto iter = voxel_indices.begin(); iter != voxel_indices.end(); iter++)
        for (int d = 0; d < ndim; d++)
            coords_filtered_[iter->second][d] = coords_[iter->first][d];

    // filter out points, first create output tensors
    Tensor points_mapping_filtered = torch::empty_like(points_mapping);
    Tensor voxel_npoints_filtered = torch::zeros({nvoxel_filtered}, voxel_npoints.options());
    auto points_mapping_filtered_ = points_mapping_filtered.accessor<int64_t, 1>();
    auto voxel_npoints_filtered_ = voxel_npoints_filtered.accessor<int, 1>();

    switch(max_points_filter_)
    {
        case MaxPointsFilterType::NONE:
            for (int i = 0; i < npoints; i++)
            {
                auto new_id = voxel_indices.find(points_mapping_[i]);
                if (new_id != voxel_indices.end())
                {
                    int vid = new_id->second;
                    points_mapping_filtered_[i] = vid;
                    voxel_npoints_filtered_[vid]++;
                }
                else points_mapping_filtered_[i] = -1;
            }
            break;

        case MaxPointsFilterType::TRIM:
            for (int i = 0; i < npoints; i++)
            {
                auto new_id = voxel_indices.find(points_mapping_[i]);
                if (new_id != voxel_indices.end())
                {
                    int vid = new_id->second;
                    if (voxel_npoints_filtered_[vid] >= max_points_)
                    {
                        points_mapping_filtered_[i] = -1;
                        continue;
                    }
                    points_mapping_filtered_[i] = vid;
                    voxel_npoints_filtered_[vid]++;
                }
                else points_mapping_filtered_[i] = -1;
            }
            break;

        case MaxPointsFilterType::FARTHEST_SAMPLING:
            throw py::value_error("Farthest Sampling not implemented!");
            break;
    }

    Tensor masked = torch::where(points_mapping_filtered >= 0)[0];
    Tensor feats_filtered = torch::index_select(feats, 0, masked);
    points_mapping_filtered = torch::index_select(points_mapping_filtered, 0, masked);
    return py::dict(
        "points"_a=feats_filtered, // FIXME: use consistent name for point features (`points` or `feats`)
        "points_mask"_a=masked,
        "points_mapping"_a=points_mapping_filtered,
        "voxel_npoints"_a=voxel_npoints_filtered,
        "coords"_a=coords_filtered
    );
}