// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once

#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

// Interface for Python
at::Tensor PSROIAlign_forward(const at::Tensor& input,
                            const at::Tensor& rois,
                            const float spatial_scale,
                            const int pooled_height,
                            const int pooled_width,
                            const int sampling_ratio,
                            const int out_dim) {
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return PSROIAlign_forward_cuda(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, out_dim);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not compiled with CPU support");
  //return PSROIAlign_forward_cpu(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio);
}

at::Tensor PSROIAlign_backward(const at::Tensor& grad,
                             const at::Tensor& rois,
                             const float spatial_scale,
                             const int pooled_height,
                             const int pooled_width,
                             const int batch_size,
                             const int channels,
                             const int height,
                             const int width,
                             const int sampling_ratio,
                             const int out_dim) {
  if (grad.type().is_cuda()) {
#ifdef WITH_CUDA
    return PSROIAlign_backward_cuda(grad, rois, spatial_scale, pooled_height, pooled_width, batch_size, channels, height, width, sampling_ratio, out_dim);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

