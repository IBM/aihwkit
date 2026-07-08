/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */

#pragma once

#ifndef RPU_CUB_NS_QUALIFIER
#ifndef CUB_NS_QUALIFIER
#undef CUB_NS_PREFIX
#undef CUB_NS_POSTFIX
#define CUB_NS_PREFIX namespace RPU {
#define CUB_NS_POSTFIX }
#define CUB_NS_QUALIFIER ::RPU::cub
#define RPU_CUB_NS_QUALIFIER RPU::cub::
#else
#define RPU_CUB_NS_QUALIFIER CUB_NS_QUALIFIER::
#endif
#endif

#include <cub/cub.cuh>

// -----------------------------------------------------------------------------
// CCCL 3.0 (shipped with CUDA 13.0+) removed the CUB "fancy" iterators
// cub::TransformInputIterator and cub::CountingInputIterator. They are
// superseded by the equivalent Thrust iterators, which CUB itself now consumes.
// Provide thin compatibility aliases (matching the old CUB template signatures)
// inside the cub namespace so the existing call sites keep compiling on both old
// and new toolkits. The aliases are only defined when the real symbols are gone.
// See the CCCL 3.0 migration guide: https://nvidia.github.io/cccl/
// -----------------------------------------------------------------------------
#if defined(CUB_VERSION) && (CUB_VERSION >= 300000)
#include <cstddef>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

CUB_NAMESPACE_BEGIN

// cub::TransformInputIterator<ValueType, ConversionOp, InputIteratorT> applied
// op to the dereferenced input and returned ValueType by value. Map that onto
// thrust::transform_iterator, pinning its reference/value types to ValueType so
// the semantics are identical regardless of the functor's declared result type.
template <
    typename ValueType,
    typename ConversionOp,
    typename InputIteratorT,
    typename OffsetT = ptrdiff_t>
using TransformInputIterator =
    ::thrust::transform_iterator<ConversionOp, InputIteratorT, ValueType, ValueType>;

template <typename ValueType, typename OffsetT = ptrdiff_t>
using CountingInputIterator = ::thrust::counting_iterator<ValueType>;

CUB_NAMESPACE_END
#endif
