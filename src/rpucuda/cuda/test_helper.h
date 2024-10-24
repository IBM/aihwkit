/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */

#pragma once

namespace RPU {

namespace test_helper {

template <typename T>
int debugKernelUpdateW(
    T *weights,
    uint32_t *x_counts,
    int x_size,
    uint32_t *d_counts,
    int d_size,
    int nK32,
    T dw_min,
    T dw_min_std,
    T bound,
    T *timings);

template <typename T>
int debugKernelUpdateWBatch(
    T *weights,
    uint32_t *x_counts,
    int x_size,
    uint32_t *d_counts,
    int d_size,
    int nK32,
    int m_batch,
    bool trans,
    T dw_min,
    T dw_min_std,
    T bound,
    int kernel_type,
    T *timings);

template <typename T>
int debugKernelUpdateWBatchShared(
    T *weights,
    uint32_t *x_counts,
    int x_size,
    uint32_t *d_counts,
    int d_size,
    int K,
    int m_batch,
    bool trans,
    T dw_min,
    T dw_min_std,
    T bound,
    int kernel_type,
    T *timings);
} // namespace test_helper
} // namespace RPU
  // namespace RPU
