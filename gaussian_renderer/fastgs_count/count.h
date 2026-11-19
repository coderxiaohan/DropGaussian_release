#pragma once

#include <torch/extension.h>
#include <memory>

// Buffers are produced and decoded by the same compilation of the local
// rasterizer. None of these buffers is exposed through the Python interface.
struct FastGSCountContext {
    torch::Tensor image, radii;
    torch::Tensor geometry, binning, frame;
    int point_count = 0, rendered_count = 0, width = 0, height = 0;

    torch::Tensor count(const torch::Tensor& mask) const;
};

std::shared_ptr<FastGSCountContext> render_fastgs_context(
    const torch::Tensor& background, const torch::Tensor& xyz,
    const torch::Tensor& colors, const torch::Tensor& opacity,
    const torch::Tensor& scales, const torch::Tensor& rotations,
    const torch::Tensor& covariance, const torch::Tensor& view,
    const torch::Tensor& projection, double tan_fovx, double tan_fovy,
    int64_t height, int64_t width, const torch::Tensor& sh, int64_t degree,
    const torch::Tensor& camera_center, bool debug);
