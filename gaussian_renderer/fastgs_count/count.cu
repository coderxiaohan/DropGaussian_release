// FastGS pixel-count criterion, independently integrated into the existing
// local rasterizer's buffers. Original criterion:
// https://github.com/fastgs/FastGS/blob/fast-dropgaussian/submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/forward.cu
// Geometry/binning/rendering are linked from this repository's Inria rasterizer
// sources under their existing license; those sources are not modified here.

#include "count.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <climits>
#include <cmath>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer_impl.h"

namespace {

// The reused source launches on CUDA's default stream. Explicit event edges
// preserve correctness when the caller uses a non-default PyTorch stream.
void wait_for(cudaStream_t destination, cudaStream_t source) {
    if (destination == source) return;
    cudaEvent_t event;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    C10_CUDA_CHECK(cudaEventRecord(event, source));
    C10_CUDA_CHECK(cudaStreamWaitEvent(destination, event, 0));
    C10_CUDA_CHECK(cudaEventDestroy(event));
}

void check_float(const torch::Tensor& tensor, const torch::Device& device, const char* name) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == device, name, " must be on the model CUDA device");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32 && tensor.is_contiguous(), name, " must be contiguous float32");
}

const float* optional_float(const torch::Tensor& tensor) {
    return tensor.numel() ? tensor.data_ptr<float>() : nullptr;
}

std::function<char*(size_t)> buffer_allocator(torch::Tensor& tensor) {
    return [&tensor](size_t bytes) {
        tensor.resize_({static_cast<int64_t>(bytes)});
        return reinterpret_cast<char*>(tensor.data_ptr<uint8_t>());
    };
}

// Each thread handles one masked pixel and replays its sorted splat list.
// Unlike alpha*T attribution, every actual blend contributes exactly ONE.
__global__ void count_pixels(
    const uint2* ranges, const uint32_t* point_list,
    const float2* means, const float4* conic_opacity,
    const bool* mask, int width, int height, int* counts) {
    const int x = blockIdx.x * BLOCK_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_Y + threadIdx.y;
    if (x >= width || y >= height || !mask[y * width + x]) return;
    const int tiles_x = (width + BLOCK_X - 1) / BLOCK_X;
    const uint2 range = ranges[blockIdx.y * tiles_x + blockIdx.x];
    float transmittance = 1.0f;
    for (uint32_t entry = range.x; entry < range.y; ++entry) {
        const uint32_t id = point_list[entry];
        const float2 mean = means[id];
        const float4 conic = conic_opacity[id];
        const float dx = mean.x - static_cast<float>(x);
        const float dy = mean.y - static_cast<float>(y);
        const float power = -0.5f * (conic.x * dx * dx + conic.z * dy * dy) - conic.y * dx * dy;
        if (power > 0.0f) continue;
        const float alpha = min(0.99f, conic.w * exp(power));
        if (alpha < 1.0f / 255.0f) continue;
        const float next = transmittance * (1.0f - alpha);
        if (next < 0.0001f) break;
        atomicAdd(counts + id, 1);
        transmittance = next;
    }
}

}  // namespace

std::shared_ptr<FastGSCountContext> render_fastgs_context(
    const torch::Tensor& background, const torch::Tensor& xyz,
    const torch::Tensor& colors, const torch::Tensor& opacity,
    const torch::Tensor& scales, const torch::Tensor& rotations,
    const torch::Tensor& covariance, const torch::Tensor& view,
    const torch::Tensor& projection, double tan_fovx, double tan_fovy,
    int64_t height, int64_t width, const torch::Tensor& sh, int64_t degree,
    const torch::Tensor& camera_center, bool debug) {
    TORCH_CHECK(xyz.is_cuda(), "FastGS pixel counting requires CUDA tensors");
    const auto device = xyz.device();
    check_float(xyz, device, "xyz");
    TORCH_CHECK(xyz.dim() == 2 && xyz.size(1) == 3 && xyz.size(0) <= INT_MAX, "xyz must have shape [N,3]");
    TORCH_CHECK(height > 0 && width > 0 && height <= INT_MAX && width <= INT_MAX && height * width <= INT_MAX,
                "image dimensions exceed the int32 pixel-count interface");
    TORCH_CHECK(degree >= 0 && degree <= 3, "SH degree must lie in [0,3]");
    TORCH_CHECK(std::isfinite(tan_fovx) && std::isfinite(tan_fovy) && tan_fovx > 0 && tan_fovy > 0,
                "field of view must be finite and positive");
    for (const auto& item : std::vector<std::pair<const torch::Tensor*, const char*>>{
             {&background, "background"}, {&colors, "colors"}, {&opacity, "opacity"},
             {&scales, "scales"}, {&rotations, "rotations"}, {&covariance, "covariance"},
             {&view, "view"}, {&projection, "projection"}, {&sh, "sh"}, {&camera_center, "camera_center"}}) {
        check_float(*item.first, device, item.second);
    }
    const int point_count = static_cast<int>(xyz.size(0));
    TORCH_CHECK(background.numel() == 3 && camera_center.numel() == 3, "background and camera center must be RGB/XYZ triples");
    TORCH_CHECK(view.numel() == 16 && projection.numel() == 16, "camera matrices must contain 16 elements");
    TORCH_CHECK(opacity.numel() == point_count, "opacity must have one value per point");
    TORCH_CHECK(covariance.numel() == point_count * int64_t(6) ||
                (scales.numel() == point_count * int64_t(3) && rotations.numel() == point_count * int64_t(4)),
                "provide [N,6] covariance or [N,3] scales and [N,4] rotations");
    TORCH_CHECK(colors.numel() == point_count * int64_t(3) ||
                (sh.dim() == 3 && sh.size(0) == point_count && sh.size(1) >= (degree + 1) * (degree + 1) && sh.size(2) == 3),
                "provide [N,3] colors or sufficient SH coefficients [N,K,3]");

    c10::cuda::CUDAGuard device_guard(device);
    const auto caller = c10::cuda::getCurrentCUDAStream(device.index());
    const auto stream = c10::cuda::getDefaultCUDAStream(device.index());
    wait_for(stream.stream(), caller.stream());
    c10::cuda::CUDAStreamGuard stream_guard(stream);
    auto context = std::make_shared<FastGSCountContext>();
    context->point_count = point_count;
    context->height = static_cast<int>(height);
    context->width = static_cast<int>(width);
    context->image = torch::zeros({NUM_CHANNELS, height, width}, xyz.options());
    context->radii = torch::zeros({point_count}, xyz.options().dtype(torch::kInt32));
    context->geometry = torch::empty({0}, xyz.options().dtype(torch::kUInt8));
    context->binning = torch::empty({0}, xyz.options().dtype(torch::kUInt8));
    context->frame = torch::empty({0}, xyz.options().dtype(torch::kUInt8));
    if (point_count > 0) {
        context->rendered_count = CudaRasterizer::Rasterizer::forward(
            buffer_allocator(context->geometry), buffer_allocator(context->binning), buffer_allocator(context->frame),
            point_count, static_cast<int>(degree), sh.numel() ? static_cast<int>(sh.size(1)) : 0,
            background.data_ptr<float>(), context->width, context->height,
            xyz.data_ptr<float>(), optional_float(sh), optional_float(colors), opacity.data_ptr<float>(),
            optional_float(scales), 1.0f, optional_float(rotations), optional_float(covariance),
            view.data_ptr<float>(), projection.data_ptr<float>(), camera_center.data_ptr<float>(),
            static_cast<float>(tan_fovx), static_cast<float>(tan_fovy), false,
            context->image.data_ptr<float>(), context->radii.data_ptr<int>(), debug);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        context->image.copy_(background.reshape({3, 1, 1}).expand({3, height, width}));
    }
    wait_for(caller.stream(), stream.stream());
    // These outputs are allocated on the default stream but consumed on the
    // caller stream. Protect their storage if the context is released while
    // caller-stream work is still queued.
    c10::cuda::CUDACachingAllocator::recordStream(context->image.storage().data_ptr(), caller);
    c10::cuda::CUDACachingAllocator::recordStream(context->radii.storage().data_ptr(), caller);
    return context;
}

torch::Tensor FastGSCountContext::count(const torch::Tensor& mask) const {
    TORCH_CHECK(mask.is_cuda() && mask.device() == image.device(), "error mask must be on the render CUDA device");
    TORCH_CHECK(mask.scalar_type() == torch::kBool && mask.is_contiguous() &&
                mask.dim() == 2 && mask.size(0) == height && mask.size(1) == width,
                "error mask must be contiguous bool [H,W]");
    c10::cuda::CUDAGuard device_guard(image.device());
    const auto caller = c10::cuda::getCurrentCUDAStream(image.device().index());
    const auto stream = c10::cuda::getDefaultCUDAStream(image.device().index());
    wait_for(stream.stream(), caller.stream());
    c10::cuda::CUDAStreamGuard stream_guard(stream);
    auto counts = torch::zeros({point_count}, radii.options());
    if (point_count > 0 && rendered_count > 0) {
        char* geometry_pointer = reinterpret_cast<char*>(geometry.data_ptr<uint8_t>());
        char* binning_pointer = reinterpret_cast<char*>(binning.data_ptr<uint8_t>());
        char* frame_pointer = reinterpret_cast<char*>(frame.data_ptr<uint8_t>());
        const auto geom = CudaRasterizer::GeometryState::fromChunk(geometry_pointer, point_count);
        const auto bins = CudaRasterizer::BinningState::fromChunk(binning_pointer, rendered_count);
        const auto img = CudaRasterizer::ImageState::fromChunk(frame_pointer, width * height);
        const dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y);
        const dim3 block(BLOCK_X, BLOCK_Y);
        count_pixels<<<grid, block, 0, stream.stream()>>>(
            img.ranges, bins.point_list, geom.means2D, geom.conic_opacity,
            mask.data_ptr<bool>(), width, height, counts.data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    wait_for(caller.stream(), stream.stream());
    c10::cuda::CUDACachingAllocator::recordStream(counts.storage().data_ptr(), caller);
    return counts;
}
