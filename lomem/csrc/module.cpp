#include "cutils.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("bool_to_uint8", &bool_to_uint8, "pack bool tensor into uint8 tensor");
    m.def("uint8_to_bool", &uint8_to_bool, "unpack uint8 tensor into bool tensor");
    m.def("regelu2_fw", &regelu2_fw, "the forward pass of ReGELU2");
    m.def("regelu2_bw", &regelu2_bw, "the backward pass of ReGELU2");
    m.def("resilu2_fw", &resilu2_fw, "the forward pass of ReSiLU2");
    m.def("resilu2_bw", &resilu2_bw, "the backward pass of ReSiLU2");
    m.def("layer_norm_fw", &layer_norm_fw, "the forward pass of LayerNorm");
    m.def("layer_norm_bw", &layer_norm_bw, "the backward pass of LayerNorm");
    m.def("rms_norm_fw", &rms_norm_fw, "the forward pass of RMSNorm");
    m.def("rms_norm_bw", &rms_norm_bw, "the backward pass of RMSNorm");
}