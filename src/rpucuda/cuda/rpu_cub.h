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
