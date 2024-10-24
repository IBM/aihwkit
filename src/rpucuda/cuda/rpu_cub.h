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
