#include "count.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    pybind11::class_<FastGSCountContext, std::shared_ptr<FastGSCountContext>>(module, "RenderContext")
        .def_readonly("image", &FastGSCountContext::image)
        .def_readonly("radii", &FastGSCountContext::radii)
        .def("count", &FastGSCountContext::count);
    module.def("render", &render_fastgs_context);
}
