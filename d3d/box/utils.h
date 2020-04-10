#pragma once

#include <torch/extension.h>

py::list rbox_2d_crop(torch::Tensor cloud, torch::Tensor boxes);
