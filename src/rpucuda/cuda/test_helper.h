/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
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
