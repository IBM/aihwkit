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

#include "bit_line_maker.h"
#include "chopped_weight_output.h"
#include "cuda_util.h"
#include "pwu_kernel.h"
#include "pwu_kernel_parameter_base.h"
#include "rpu_pulsed_meta_parameter.h"
#include "rpucuda_pulsed_device.h"

namespace RPU {

#define DEFINE_PWU_KERNEL_PARAMETER(NAME, BASE, RUN_BODY)                                          \
  class PWUKernelParameter##NAME : public PWUKernelParameter##BASE<T> {                            \
                                                                                                   \
  public:                                                                                          \
    using PWUKernelParameter##BASE<T>::PWUKernelParameter##BASE;                                   \
                                                                                                   \
    void                                                                                           \
    run(cudaStream_t s,                                                                            \
        T *dev_weights,                                                                            \
        int m_batch,                                                                               \
        const BitLineMaker<T> *blm,                                                                \
        PulsedRPUDeviceCuda<T> *rpucuda_device,                                                    \
        const PulsedUpdateMetaParameter<T> &up,                                                    \
        curandState_t *dev_states,                                                                 \
        int one_sided = 0,                                                                         \
        uint32_t *x_counts_chunk = nullptr,                                                        \
        uint32_t *d_counts_chunk = nullptr,                                                        \
        const ChoppedWeightOutput<T> *cwo = nullptr) override {                                    \
      RUN_BODY;                                                                                    \
    };                                                                                             \
  }

#define DEFINE_PWU_KERNEL_BASE(NAME, CTOR_BODY)                                                    \
  template <typename T> class PWUKernelParameter##NAME : public PWUKernelParameterBase<T> {        \
                                                                                                   \
  public:                                                                                          \
    using PWUKernelParameterBase<T>::print;                                                        \
    using PWUKernelParameterBase<T>::run;                                                          \
                                                                                                   \
    PWUKernelParameter##NAME(                                                                      \
        CudaContextPtr construction_context,                                                       \
        int x_size,                                                                                \
        int d_size,                                                                                \
        int m_batch,                                                                               \
        int nK32,                                                                                  \
        int use_bo64,                                                                              \
        bool out_trans_in,                                                                         \
        const PulsedUpdateMetaParameter<T> &up,                                                    \
        std::string update_name)                                                                   \
        : PWUKernelParameterBase<T>(                                                               \
              construction_context,                                                                \
              x_size,                                                                              \
              d_size,                                                                              \
              m_batch,                                                                             \
              nK32,                                                                                \
              use_bo64,                                                                            \
              out_trans_in,                                                                        \
              up,                                                                                  \
              update_name + "/" + #NAME) {                                                         \
      CTOR_BODY;                                                                                   \
    }                                                                                              \
  }

#define RPU_SWITCH_TRANS_TEMPLATE_OS(                                                              \
    NUM_T, ONE_SIDED, COUNT_T, X_TRANS, D_TRANS, S, NBLOCKS, NTHREADS, SHARED_MEM, KERNEL, ARGS)   \
  if (X_TRANS & D_TRANS) {                                                                         \
    KERNEL<NUM_T, ONE_SIDED, COUNT_T, true, true><<<NBLOCKS, NTHREADS, SHARED_MEM, S>>> ARGS;      \
  } else if (!X_TRANS & !D_TRANS) {                                                                \
    KERNEL<NUM_T, ONE_SIDED, COUNT_T, false, false><<<NBLOCKS, NTHREADS, SHARED_MEM, S>>> ARGS;    \
  } else if (!X_TRANS & D_TRANS) {                                                                 \
    KERNEL<NUM_T, ONE_SIDED, COUNT_T, false, true><<<NBLOCKS, NTHREADS, SHARED_MEM, S>>> ARGS;     \
  } else if (X_TRANS & !D_TRANS) {                                                                 \
    KERNEL<NUM_T, ONE_SIDED, COUNT_T, true, false><<<NBLOCKS, NTHREADS, SHARED_MEM, S>>> ARGS;     \
  };

#define RPU_SWITCH_TRANS_TEMPLATE_FUNCTOR_OS(                                                      \
    NUM_T, ONE_SIDED, COUNT_T, X_TRANS, D_TRANS, S, NBLOCKS, NTHREADS, SHARED_MEM, KERNEL,         \
    FUNCTOR, GP_COUNT, ARGS)                                                                       \
  if (X_TRANS & D_TRANS) {                                                                         \
    KERNEL<NUM_T, ONE_SIDED, COUNT_T, true, true, FUNCTOR, GP_COUNT>                               \
        <<<NBLOCKS, NTHREADS, SHARED_MEM, S>>> ARGS;                                               \
  } else if (!X_TRANS & !D_TRANS) {                                                                \
    KERNEL<NUM_T, ONE_SIDED, COUNT_T, false, false, FUNCTOR, GP_COUNT>                             \
        <<<NBLOCKS, NTHREADS, SHARED_MEM, S>>> ARGS;                                               \
  } else if (!X_TRANS & D_TRANS) {                                                                 \
    KERNEL<NUM_T, ONE_SIDED, COUNT_T, false, true, FUNCTOR, GP_COUNT>                              \
        <<<NBLOCKS, NTHREADS, SHARED_MEM, S>>> ARGS;                                               \
  } else if (X_TRANS & !D_TRANS) {                                                                 \
    KERNEL<NUM_T, ONE_SIDED, COUNT_T, true, false, FUNCTOR, GP_COUNT>                              \
        <<<NBLOCKS, NTHREADS, SHARED_MEM, S>>> ARGS;                                               \
  };

#define RPU_SWITCH_TRANS_TEMPLATE(                                                                 \
    NUM_T, ONE_SIDED, COUNT_T, X_TRANS, D_TRANS, S, NBLOCKS, NTHREADS, SHARED_MEM, KERNEL, ARGS)   \
  if (ONE_SIDED == -1) {                                                                           \
    RPU_SWITCH_TRANS_TEMPLATE_OS(                                                                  \
        NUM_T, -1, COUNT_T, X_TRANS, D_TRANS, S, NBLOCKS, NTHREADS, SHARED_MEM, KERNEL, ARGS);     \
  } else if (ONE_SIDED == 1) {                                                                     \
    RPU_SWITCH_TRANS_TEMPLATE_OS(                                                                  \
        NUM_T, 1, COUNT_T, X_TRANS, D_TRANS, S, NBLOCKS, NTHREADS, SHARED_MEM, KERNEL, ARGS);      \
  } else {                                                                                         \
    RPU_SWITCH_TRANS_TEMPLATE_OS(                                                                  \
        NUM_T, 0, COUNT_T, X_TRANS, D_TRANS, S, NBLOCKS, NTHREADS, SHARED_MEM, KERNEL, ARGS);      \
  }

#define RPU_SWITCH_TRANS_TEMPLATE_FUNCTOR(                                                         \
    NUM_T, ONE_SIDED, COUNT_T, X_TRANS, D_TRANS, S, NBLOCKS, NTHREADS, SHARED_MEM, KERNEL,         \
    FUNCTOR, GP_COUNT, ARGS)                                                                       \
  if (ONE_SIDED == -1) {                                                                           \
    RPU_SWITCH_TRANS_TEMPLATE_FUNCTOR_OS(                                                          \
        NUM_T, -1, COUNT_T, X_TRANS, D_TRANS, S, NBLOCKS, NTHREADS, SHARED_MEM, KERNEL, FUNCTOR,   \
        GP_COUNT, ARGS);                                                                           \
  } else if (ONE_SIDED == 1) {                                                                     \
    RPU_SWITCH_TRANS_TEMPLATE_FUNCTOR_OS(                                                          \
        NUM_T, 1, COUNT_T, X_TRANS, D_TRANS, S, NBLOCKS, NTHREADS, SHARED_MEM, KERNEL, FUNCTOR,    \
        GP_COUNT, ARGS);                                                                           \
  } else {                                                                                         \
    RPU_SWITCH_TRANS_TEMPLATE_FUNCTOR_OS(                                                          \
        NUM_T, 0, COUNT_T, X_TRANS, D_TRANS, S, NBLOCKS, NTHREADS, SHARED_MEM, KERNEL, FUNCTOR,    \
        GP_COUNT, ARGS);                                                                           \
  }

#define RPU_SWITCH_TRANS_TEMPLATE_FUNCTOR_IMPLICIT(                                                \
    NUM_T, X_TRANS, D_TRANS, S, NBLOCKS, NTHREADS, SHARED_MEM, KERNEL, FUNCTOR, GP_COUNT, ARGS)    \
  if (X_TRANS & D_TRANS) {                                                                         \
    KERNEL<NUM_T, true, true, FUNCTOR, GP_COUNT><<<NBLOCKS, NTHREADS, SHARED_MEM, S>>> ARGS;       \
  } else if (!X_TRANS & !D_TRANS) {                                                                \
    KERNEL<NUM_T, false, false, FUNCTOR, GP_COUNT><<<NBLOCKS, NTHREADS, SHARED_MEM, S>>> ARGS;     \
  } else if (!X_TRANS & D_TRANS) {                                                                 \
    KERNEL<NUM_T, false, true, FUNCTOR, GP_COUNT><<<NBLOCKS, NTHREADS, SHARED_MEM, S>>> ARGS;      \
  } else if (X_TRANS & !D_TRANS) {                                                                 \
    KERNEL<NUM_T, true, false, FUNCTOR, GP_COUNT><<<NBLOCKS, NTHREADS, SHARED_MEM, S>>> ARGS;      \
  };

#define COMMA ,

/********************************************************************************
 * PWUKernelParameterSingleBase
 *********************************************************************************/

DEFINE_PWU_KERNEL_BASE(
    SingleBase,
    /*ctor*/
    if ((m_batch > 1) || (use_bo64 > 0)) {
      this->valid = false;
      return;
    }

    int n32 = (this->size / 32 + 1) * 32;
    this->nthreads = MIN(RPU_THREADS_PER_BLOCK_UPDATE, n32);
    this->nblocks =
        MIN(this->max_block_count, construction_context->getNBlocks(this->size, this->nthreads));
    this->nstates = this->nthreads * this->nblocks;);

/********************************************************************************
 * PWUKernelParameterSingleFunctor
 *********************************************************************************/

#define START_SINGLE_FUNCTOR(COUNT_T, ARGS1)                                                       \
  if (one_sided == -1) {                                                                           \
    kernelUpdateWFunctor<T, -1, COUNT_T, FunctorT, gp_count>                                       \
        <<<this->nblocks, this->nthreads, this->shared_mem, s>>> ARGS1;                            \
  } else if (one_sided == 1) {                                                                     \
    kernelUpdateWFunctor<T, 1, COUNT_T, FunctorT, gp_count>                                        \
        <<<this->nblocks, this->nthreads, this->shared_mem, s>>> ARGS1;                            \
  } else {                                                                                         \
    kernelUpdateWFunctor<T, 0, COUNT_T, FunctorT, gp_count>                                        \
        <<<this->nblocks, this->nthreads, this->shared_mem, s>>> ARGS1;                            \
  }

template <typename T, typename FunctorT, int gp_count>
DEFINE_PWU_KERNEL_PARAMETER(
    SingleFunctor,
    SingleBase,
    /*run*/
    if (this->implicit_pulses) {
      START_SINGLE_FUNCTOR(
          T, (dev_weights, blm->getXData(), this->x_size, blm->getDData(), this->d_size,
              rpucuda_device->get4ParamsData(), rpucuda_device->get2ParamsData(),
              rpucuda_device->get1ParamsData(), rpucuda_device->getGlobalParamsData(), 1,
              rpucuda_device->getWeightGranularityNoise(), dev_states));
    } else {
      START_SINGLE_FUNCTOR(
          uint32_t,
          (dev_weights, x_counts_chunk ? x_counts_chunk : blm->getXCountsData(), this->x_size,
           d_counts_chunk ? d_counts_chunk : blm->getDCountsData(), this->d_size,
           rpucuda_device->get4ParamsData(), rpucuda_device->get2ParamsData(),
           rpucuda_device->get1ParamsData(), rpucuda_device->getGlobalParamsData(), this->nK32,
           rpucuda_device->getWeightGranularityNoise(), dev_states));
    });
#undef START_SINGLE_FUNCTOR

/********************************************************************************
 * PWUKernelParameterBatchBase
 *********************************************************************************/
DEFINE_PWU_KERNEL_BASE(
    BatchBase,
    /*ctor*/
    if (m_batch > 1000) {
      this->valid = false;
      return;
    };
    this->nthreads = MIN(RPU_THREADS_PER_BLOCK_UPDATE, this->size);
    this->nthreads = (this->nthreads + 31) / 32 * 32;
    this->nblocks =
        MIN(this->max_block_count, construction_context->getNBlocks(this->size, this->nthreads));
    this->nstates = this->nthreads * this->nblocks;);

/********************************************************************************
 * PWUKernelParameterBatchBaseInf // no limit on size
 *********************************************************************************/
DEFINE_PWU_KERNEL_BASE(BatchBaseInf,
                       /*ctor*/
                       this->nthreads = MIN(RPU_THREADS_PER_BLOCK_UPDATE, this->size);
                       this->nthreads = (this->nthreads + 31) / 32 * 32;
                       this->nblocks =
                           MIN(this->max_block_count,
                               construction_context->getNBlocks(this->size, this->nthreads));
                       this->nstates = this->nthreads * this->nblocks;);

/********************************************************************************
 * PWUKernelParameterBatchFunctor
 *********************************************************************************/

template <typename T, typename FunctorT, int gp_count>
DEFINE_PWU_KERNEL_PARAMETER(
    BatchFunctor,
    BatchBase,
    /*run*/
    if (this->implicit_pulses) {
      RPU_SWITCH_TRANS_TEMPLATE_FUNCTOR(
          T, one_sided, T, this->out_trans, this->out_trans, s, this->nblocks, this->nthreads,
          this->shared_mem, kernelUpdateWBatchFunctor, FunctorT, gp_count,
          (dev_weights, blm->getXData(), this->x_size, blm->getDData(), this->d_size,
           rpucuda_device->get4ParamsData(), rpucuda_device->get2ParamsData(),
           rpucuda_device->get1ParamsData(), rpucuda_device->getGlobalParamsData(), 1, m_batch,
           rpucuda_device->getWeightGranularityNoise(), dev_states));
    } else if (this->use_bo64) {
      RPU_SWITCH_TRANS_TEMPLATE_FUNCTOR(
          T, one_sided, uint64_t, this->out_trans, this->out_trans, s, this->nblocks,
          this->nthreads, this->shared_mem, kernelUpdateWBatchFunctor, FunctorT, gp_count,
          (dev_weights, blm->getXCountsBo64Data(), this->x_size, blm->getDCountsBo64Data(),
           this->d_size, rpucuda_device->get4ParamsData(), rpucuda_device->get2ParamsData(),
           rpucuda_device->get1ParamsData(), rpucuda_device->getGlobalParamsData(), this->nK32,
           blm->getBo64Batch(m_batch), rpucuda_device->getWeightGranularityNoise(), dev_states,
           blm->getKnData(up.update_bl_management, m_batch)));
    } else { // standard
      RPU_SWITCH_TRANS_TEMPLATE_FUNCTOR(
          T, one_sided, uint32_t, this->out_trans, this->out_trans, s, this->nblocks,
          this->nthreads, this->shared_mem, kernelUpdateWBatchFunctor, FunctorT, gp_count,
          (dev_weights, x_counts_chunk ? x_counts_chunk : blm->getXCountsData(), this->x_size,
           d_counts_chunk ? d_counts_chunk : blm->getDCountsData(), this->d_size,
           rpucuda_device->get4ParamsData(), rpucuda_device->get2ParamsData(),
           rpucuda_device->get1ParamsData(), rpucuda_device->getGlobalParamsData(), this->nK32,
           m_batch, rpucuda_device->getWeightGranularityNoise(), dev_states));
    });

/********************************************************************************
 * PWUKernelParameterBatchSum / BoundCheck
 *********************************************************************************/

#define RPU_PWU_START_BATCH_KERNEL(KNAME)                                                          \
  if (this->implicit_pulses) {                                                                     \
    RPU_SWITCH_TRANS_TEMPLATE(                                                                     \
        T, one_sided, T, this->out_trans, this->out_trans, s, this->nblocks, this->nthreads,       \
        this->shared_mem, KNAME,                                                                   \
        (dev_weights, blm->getXData(), this->x_size, blm->getDData(), this->d_size,                \
         rpucuda_device->get4ParamsData(), 1, m_batch,                                             \
         rpucuda_device->getWeightGranularityNoise(), dev_states));                                \
  } else if (this->use_bo64) {                                                                     \
    RPU_SWITCH_TRANS_TEMPLATE(                                                                     \
        T, one_sided, uint64_t, this->out_trans, this->out_trans, s, this->nblocks,                \
        this->nthreads, this->shared_mem, KNAME,                                                   \
        (dev_weights, blm->getXCountsBo64Data(), this->x_size, blm->getDCountsBo64Data(),          \
         this->d_size, rpucuda_device->get4ParamsData(), this->nK32, blm->getBo64Batch(m_batch),   \
         rpucuda_device->getWeightGranularityNoise(), dev_states,                                  \
         blm->getKnData(up.update_bl_management, m_batch)));                                       \
  } else {                                                                                         \
    RPU_SWITCH_TRANS_TEMPLATE(                                                                     \
        T, one_sided, uint32_t, this->out_trans, this->out_trans, s, this->nblocks,                \
        this->nthreads, this->shared_mem, KNAME,                                                   \
        (dev_weights, x_counts_chunk ? x_counts_chunk : blm->getXCountsData(), this->x_size,       \
         d_counts_chunk ? d_counts_chunk : blm->getDCountsData(), this->d_size,                    \
         rpucuda_device->get4ParamsData(), this->nK32, m_batch,                                    \
         rpucuda_device->getWeightGranularityNoise(), dev_states));                                \
  }

template <typename T>
DEFINE_PWU_KERNEL_PARAMETER(BatchSum,
                            BatchBase,
                            /*run*/
                            RPU_PWU_START_BATCH_KERNEL(kernelUpdateWBatchSum););

template <typename T>
DEFINE_PWU_KERNEL_PARAMETER(BatchSumBoundCheck,
                            BatchBase,
                            /*run*/
                            RPU_PWU_START_BATCH_KERNEL(kernelUpdateWBatchSumBoundCheck););

#undef RPU_PWU_START_BATCH_KERNEL

/********************************************************************************
 * PWUKernelParameterBatchSharedBase
 *********************************************************************************/
DEFINE_PWU_KERNEL_BASE(
    BatchSharedBase,
    /*ctor*/
    if (m_batch == 1) {
      this->valid = false;
      return;
    }

    int nK32_shared = this->implicit_pulses ? 1 : nK32;
    int reserved_shared_mem = 128 * sizeof(T); // for global pars
    int sz = (d_size + 31) / 32 * 32 * (x_size + 15) / 16 * 16;
    this->nthreads = MIN(RPU_THREADS_PER_BLOCK_UPDATE, sz);
    this->nthreads = (this->nthreads + 31) / 32 * 32;
    this->nblocks =
        MIN(this->max_block_count, construction_context->getNBlocks(sz, this->nthreads));
    this->shared_mem_per_batch = (this->nthreads / 32 + 32) * this->sizeof_count * nK32_shared;
    int max_shared_mem = (construction_context->getSharedMemPerBlock() - reserved_shared_mem) /
                         RPU_UPDATE_BLOCKS_PER_SM;
    this->max_batch_load_stride = max_shared_mem / this->shared_mem_per_batch;
    this->shared_mem = this->shared_mem_per_batch * this->max_batch_load_stride;
    this->nstates = this->nthreads * this->nblocks;);

#define RPU_PWU_START_BATCH_SHARED_INIT                                                            \
  int nK32_in = blm->getNK32Current();                                                             \
  if (nK32_in != this->nK32) {                                                                     \
    RPU_FATAL("nK32 changed. This is not supported");                                              \
  };                                                                                               \
                                                                                                   \
  int batch_load_stride = MIN(this->max_batch_load_stride, m_batch);                               \
  int shared_mem = this->shared_mem_per_batch * batch_load_stride;

/********************************************************************************
 * PWUKernelParameterBatchSharedFunctor
 *********************************************************************************/

template <typename T, typename FunctorT, int gp_count>
DEFINE_PWU_KERNEL_PARAMETER(
    BatchSharedFunctor,
    BatchSharedBase,
    /*run*/
    RPU_PWU_START_BATCH_SHARED_INIT;
    if (this->implicit_pulses) {
      RPU_SWITCH_TRANS_TEMPLATE_FUNCTOR(
          T, one_sided, T, this->out_trans, this->out_trans, s, this->nblocks, this->nthreads,
          shared_mem, kernelUpdateWBatchSharedFunctor, FunctorT, gp_count,
          (dev_weights, blm->getXData(), this->x_size, blm->getDData(), this->d_size,
           rpucuda_device->get4ParamsData(), rpucuda_device->get2ParamsData(),
           rpucuda_device->get1ParamsData(), rpucuda_device->getGlobalParamsData(), 1, m_batch,
           batch_load_stride, rpucuda_device->getWeightGranularityNoise(), dev_states));
    } else if (this->use_bo64) {
      RPU_SWITCH_TRANS_TEMPLATE_FUNCTOR(
          T, one_sided, uint64_t, this->out_trans, this->out_trans, s, this->nblocks,
          this->nthreads, shared_mem, kernelUpdateWBatchSharedFunctor, FunctorT, gp_count,
          (dev_weights, blm->getXCountsBo64Data(), this->x_size, blm->getDCountsBo64Data(),
           this->d_size, rpucuda_device->get4ParamsData(), rpucuda_device->get2ParamsData(),
           rpucuda_device->get1ParamsData(), rpucuda_device->getGlobalParamsData(), this->nK32,
           blm->getBo64Batch(m_batch), batch_load_stride,
           rpucuda_device->getWeightGranularityNoise(), dev_states,
           blm->getKnData(up.update_bl_management, m_batch)));
    } else {
      RPU_SWITCH_TRANS_TEMPLATE_FUNCTOR(
          T, one_sided, uint32_t, this->out_trans, this->out_trans, s, this->nblocks,
          this->nthreads, shared_mem, kernelUpdateWBatchSharedFunctor, FunctorT, gp_count,
          (dev_weights, x_counts_chunk ? x_counts_chunk : blm->getXCountsData(), this->x_size,
           d_counts_chunk ? d_counts_chunk : blm->getDCountsData(), this->d_size,
           rpucuda_device->get4ParamsData(), rpucuda_device->get2ParamsData(),
           rpucuda_device->get1ParamsData(), rpucuda_device->getGlobalParamsData(), this->nK32,
           m_batch, batch_load_stride, rpucuda_device->getWeightGranularityNoise(), dev_states));
    });

/********************************************************************************
 * PWUKernelParameterBatchSharedSum / BoundCheck
 *********************************************************************************/
#define RPU_PWU_START_BATCH_SHARED_KERNEL(KNAME)                                                   \
  RPU_PWU_START_BATCH_SHARED_INIT;                                                                 \
  if (this->implicit_pulses) {                                                                     \
    RPU_SWITCH_TRANS_TEMPLATE(                                                                     \
        T, one_sided, T, this->out_trans, this->out_trans, s, this->nblocks, this->nthreads,       \
        shared_mem, KNAME,                                                                         \
        (dev_weights, blm->getXData(), this->x_size, blm->getDData(), this->d_size,                \
         rpucuda_device->get4ParamsData(), 1, m_batch, batch_load_stride,                          \
         rpucuda_device->getWeightGranularityNoise(), dev_states));                                \
  } else if (this->use_bo64) {                                                                     \
    RPU_SWITCH_TRANS_TEMPLATE(                                                                     \
        T, one_sided, uint64_t, this->out_trans, this->out_trans, s, this->nblocks,                \
        this->nthreads, shared_mem, KNAME,                                                         \
        (dev_weights, blm->getXCountsBo64Data(), this->x_size, blm->getDCountsBo64Data(),          \
         this->d_size, rpucuda_device->get4ParamsData(), this->nK32, blm->getBo64Batch(m_batch),   \
         batch_load_stride, rpucuda_device->getWeightGranularityNoise(), dev_states,               \
         blm->getKnData(up.update_bl_management, m_batch)));                                       \
  } else {                                                                                         \
    RPU_SWITCH_TRANS_TEMPLATE(                                                                     \
        T, one_sided, uint32_t, this->out_trans, this->out_trans, s, this->nblocks,                \
        this->nthreads, shared_mem, KNAME,                                                         \
        (dev_weights, x_counts_chunk ? x_counts_chunk : blm->getXCountsData(), this->x_size,       \
         d_counts_chunk ? d_counts_chunk : blm->getDCountsData(), this->d_size,                    \
         rpucuda_device->get4ParamsData(), this->nK32, m_batch, batch_load_stride,                 \
         rpucuda_device->getWeightGranularityNoise(), dev_states));                                \
  }

template <typename T>
DEFINE_PWU_KERNEL_PARAMETER(BatchSharedSum,
                            BatchSharedBase,
                            /*run*/
                            RPU_PWU_START_BATCH_SHARED_KERNEL(kernelUpdateWBatchSharedSum););

template <typename T>
DEFINE_PWU_KERNEL_PARAMETER(
    BatchSharedSumBoundCheck,
    BatchSharedBase,
    /*run*/
    RPU_PWU_START_BATCH_SHARED_KERNEL(kernelUpdateWBatchSharedSumBoundCheck););

/********************************************************************************
 * PWUKernelParameterBatchSharedWeightOutputBase
 *********************************************************************************/
DEFINE_PWU_KERNEL_BASE(
    BatchSharedWeightOutputBase,
    /*ctor*/

    int nK32_shared = this->implicit_pulses ? 1 : nK32;

    int sz = ((d_size + 31) / 32 * 32) * ((x_size + 15) / 16 * 16);
    this->nthreads = MIN(RPU_THREADS_PER_BLOCK_UPDATE, sz);
    this->nthreads = (this->nthreads + 31) / 32 * 32;
    this->use_cwo = true;
    this->nblocks =
        MIN(this->max_block_count, construction_context->getNBlocks(sz, this->nthreads));
    this->shared_mem_per_batch = (this->nthreads / 32 + 32) * this->sizeof_count * nK32_shared;
    if (this->use_bo64 && up.update_bl_management) {
      this->shared_mem_per_batch += sizeof(uint32_t);
    } int max_shared_mem = construction_context->getSharedMemPerBlock() / RPU_UPDATE_BLOCKS_PER_SM;
    this->max_batch_load_stride = max_shared_mem / this->shared_mem_per_batch;
    this->shared_mem = this->shared_mem_per_batch * this->max_batch_load_stride;
    this->nstates = this->nthreads * this->nblocks;);

#define RPU_PWU_START_BATCH_SHARED_INIT_WO                                                         \
  int nK32_in = blm->getNK32Current();                                                             \
  if (nK32_in != this->nK32) {                                                                     \
    RPU_FATAL("nK32 changed. This is not supported");                                              \
  };                                                                                               \
                                                                                                   \
  int batch_load_stride = MIN(this->max_batch_load_stride, (m_batch + 1) / 2 * 2);                 \
  int shared_mem = this->shared_mem_per_batch * batch_load_stride;

/********************************************************************************
 * PWUKernelParameterBatchSharedWeightOutputFunctor
 *********************************************************************************/

template <typename T, typename FunctorT, int gp_count>
DEFINE_PWU_KERNEL_PARAMETER(
    BatchSharedWeightOutputFunctor,
    BatchSharedWeightOutputBase,
    /*run*/
    RPU_PWU_START_BATCH_SHARED_INIT;
    if (this->use_bo64 && up.update_bl_management && (m_batch / 2 * 2 != m_batch)) {
      shared_mem += sizeof(uint32_t);
    }

    if (one_sided) { RPU_FATAL("One sided is not supported by weight-output kernels."); }

    if (x_counts_chunk || d_counts_chunk) {
      RPU_FATAL("Chunking and weight output is not supported together.");
    } T dw_min_std =
        static_cast<const PulsedRPUDeviceMetaParameter<T> &>(rpucuda_device->getPar()).dw_min_std;

    if (cwo) {
      const auto &cwo_par = cwo->getPar();

      if (this->implicit_pulses) {
        RPU_SWITCH_TRANS_TEMPLATE_FUNCTOR_OS(
            T, 0, T, this->out_trans, this->out_trans, s, this->nblocks, this->nthreads, shared_mem,
            kernelUpdateWBatchSharedWeightOutputFunctor, FunctorT, gp_count,
            (dev_weights, blm->getXData(), this->x_size, blm->getDData(), this->d_size,
             rpucuda_device->get4ParamsData(), rpucuda_device->get2ParamsData(),
             rpucuda_device->get1ParamsData(), rpucuda_device->getGlobalParamsData(), 1, m_batch,
             batch_load_stride, dw_min_std, cwo->getWeightOutputData(),
             cwo->getWeightOutputInChopperData(), cwo->getWeightOutputOutChopperData(),
             cwo->getNumWeightOutputs(), cwo->getNWOCounter(), cwo_par.every, cwo_par.use_columns,
             cwo->getValStart(), cwo->getBatchStart(), cwo->getFlexibleInSize(),
             cwo_par.in_chop_prob, cwo_par.out_chop_prob, cwo->getXChopperInData(),
             cwo->getDChopperInData(), cwo->getXChopperOutData(), cwo->getDChopperOutData(),
             cwo->getXSwitchingProbData(), cwo->getDSwitchingProbData(), dev_states));
      } else if (this->use_bo64) {
        RPU_SWITCH_TRANS_TEMPLATE_FUNCTOR_OS(
            T, 0, uint64_t, this->out_trans, this->out_trans, s, this->nblocks, this->nthreads,
            shared_mem, kernelUpdateWBatchSharedWeightOutputFunctor, FunctorT, gp_count,
            (dev_weights, blm->getXCountsBo64Data(), this->x_size, blm->getDCountsBo64Data(),
             this->d_size, rpucuda_device->get4ParamsData(), rpucuda_device->get2ParamsData(),
             rpucuda_device->get1ParamsData(), rpucuda_device->getGlobalParamsData(),
             blm->getCurrentBL(), blm->getBo64Batch(m_batch), batch_load_stride, dw_min_std,
             cwo->getWeightOutputData(), cwo->getWeightOutputInChopperData(),
             cwo->getWeightOutputOutChopperData(), cwo->getNumWeightOutputs(), cwo->getNWOCounter(),
             cwo_par.every, cwo_par.use_columns, cwo->getValStart(), cwo->getBatchStart(),
             cwo->getFlexibleInSize(), cwo_par.in_chop_prob, cwo_par.out_chop_prob,
             cwo->getXChopperInData(), cwo->getDChopperInData(), cwo->getXChopperOutData(),
             cwo->getDChopperOutData(), cwo->getXSwitchingProbData(), cwo->getDSwitchingProbData(),
             dev_states, cwo->getWeightOutputSignalsData(), m_batch,
             blm->getKnData(up.update_bl_management, m_batch)));
      } else {
        RPU_SWITCH_TRANS_TEMPLATE_FUNCTOR_OS(
            T, 0, uint32_t, this->out_trans, this->out_trans, s, this->nblocks, this->nthreads,
            shared_mem, kernelUpdateWBatchSharedWeightOutputFunctor, FunctorT, gp_count,
            (dev_weights, blm->getXCountsData(), this->x_size, blm->getDCountsData(), this->d_size,
             rpucuda_device->get4ParamsData(), rpucuda_device->get2ParamsData(),
             rpucuda_device->get1ParamsData(), rpucuda_device->getGlobalParamsData(),
             blm->getCurrentBL(), m_batch, batch_load_stride, dw_min_std,
             cwo->getWeightOutputData(), cwo->getWeightOutputInChopperData(),
             cwo->getWeightOutputOutChopperData(), cwo->getNumWeightOutputs(), cwo->getNWOCounter(),
             cwo_par.every, cwo_par.use_columns, cwo->getValStart(), cwo->getBatchStart(),
             cwo->getFlexibleInSize(), cwo_par.in_chop_prob, cwo_par.out_chop_prob,
             cwo->getXChopperInData(), cwo->getDChopperInData(), cwo->getXChopperOutData(),
             cwo->getDChopperOutData(), cwo->getXSwitchingProbData(), cwo->getDSwitchingProbData(),
             dev_states));
      }
      // no CWO given
    } else {
      if (this->implicit_pulses) {
        RPU_SWITCH_TRANS_TEMPLATE_FUNCTOR_OS(
            T, 0, T, this->out_trans, this->out_trans, s, this->nblocks, this->nthreads, shared_mem,
            kernelUpdateWBatchSharedWeightOutputFunctor, FunctorT, gp_count,
            (dev_weights, blm->getXData(), this->x_size, blm->getDData(), this->d_size,
             rpucuda_device->get4ParamsData(), rpucuda_device->get2ParamsData(),
             rpucuda_device->get1ParamsData(), rpucuda_device->getGlobalParamsData(), 1, m_batch,
             batch_load_stride, dw_min_std, nullptr, nullptr, nullptr, 0, 0, 0, false, 0, 0, false,
             (T)0, (T)0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, dev_states));
      } else if (this->use_bo64) {
        RPU_SWITCH_TRANS_TEMPLATE_FUNCTOR_OS(
            T, 0, uint64_t, this->out_trans, this->out_trans, s, this->nblocks, this->nthreads,
            shared_mem, kernelUpdateWBatchSharedWeightOutputFunctor, FunctorT, gp_count,
            (dev_weights, blm->getXCountsBo64Data(), this->x_size, blm->getDCountsBo64Data(),
             this->d_size, rpucuda_device->get4ParamsData(), rpucuda_device->get2ParamsData(),
             rpucuda_device->get1ParamsData(), rpucuda_device->getGlobalParamsData(),
             blm->getCurrentBL(), blm->getBo64Batch(m_batch), batch_load_stride, dw_min_std,
             nullptr, nullptr, nullptr, 0, 0, 0, false, 0, 0, false, (T)0, (T)0, nullptr, nullptr,
             nullptr, nullptr, nullptr, nullptr, dev_states, nullptr, m_batch,
             blm->getKnData(up.update_bl_management, m_batch)));
      } else {
        RPU_SWITCH_TRANS_TEMPLATE_FUNCTOR_OS(
            T, 0, uint32_t, this->out_trans, this->out_trans, s, this->nblocks, this->nthreads,
            shared_mem, kernelUpdateWBatchSharedWeightOutputFunctor, FunctorT, gp_count,
            (dev_weights, blm->getXCountsData(), this->x_size, blm->getDCountsData(), this->d_size,
             rpucuda_device->get4ParamsData(), rpucuda_device->get2ParamsData(),
             rpucuda_device->get1ParamsData(), rpucuda_device->getGlobalParamsData(),
             blm->getCurrentBL(), m_batch, batch_load_stride, dw_min_std, nullptr, nullptr, nullptr,
             0, 0, 0, false, 0, 0, false, (T)0, (T)0, nullptr, nullptr, nullptr, nullptr, nullptr,
             nullptr, dev_states));
      }
    });

/********************************************************************************
 * PWUKernelCounter
 *********************************************************************************/

#define RPU_PWU_COUNTER_KERNEL                                                                     \
  if (this->implicit_pulses) {                                                                     \
    RPU_SWITCH_TRANS_TEMPLATE(                                                                     \
        T, one_sided, T, this->out_trans, this->out_trans, s, this->nblocks, this->nthreads,       \
        this->shared_mem, kernelPulseCounter,                                                      \
        (rpucuda_device->getPosPulseCountData(), rpucuda_device->getNegPulseCountData(),           \
         blm->getXData(), this->x_size, blm->getDData(), this->d_size, 1, m_batch));               \
  } else if (this->use_bo64) {                                                                     \
    RPU_SWITCH_TRANS_TEMPLATE(                                                                     \
        T, one_sided, uint64_t, this->out_trans, this->out_trans, s, this->nblocks,                \
        this->nthreads, this->shared_mem, kernelPulseCounter,                                      \
        (rpucuda_device->getPosPulseCountData(), rpucuda_device->getNegPulseCountData(),           \
         blm->getXCountsBo64Data(), this->x_size, blm->getDCountsBo64Data(), this->d_size,         \
         this->nK32, blm->getBo64Batch(m_batch),                                                   \
         blm->getKnData(up.update_bl_management, m_batch)));                                       \
  } else {                                                                                         \
    RPU_SWITCH_TRANS_TEMPLATE(                                                                     \
        T, one_sided, uint32_t, this->out_trans, this->out_trans, s, this->nblocks,                \
        this->nthreads, this->shared_mem, kernelPulseCounter,                                      \
        (rpucuda_device->getPosPulseCountData(), rpucuda_device->getNegPulseCountData(),           \
         x_counts_chunk ? x_counts_chunk : blm->getXCountsData(), this->x_size,                    \
         d_counts_chunk ? d_counts_chunk : blm->getDCountsData(), this->d_size, this->nK32,        \
         m_batch));                                                                                \
  }

template <typename T>
DEFINE_PWU_KERNEL_PARAMETER(PulseCounter,
                            BatchBaseInf,
                            /*run*/
                            RPU_PWU_COUNTER_KERNEL;);

#undef RPU_PWU_COUNTER_KERNEL

#undef RPU_PWU_START_BATCH_SHARED_KERNEL
#undef RPU_PWU_START_BATCH_SHARED_INIT

#undef RPU_SWITCH_TRANS_TEMPLATE
#undef RPU_SWITCH_TRANS_TEMPLATE_OS
#undef RPU_SWITCH_TRANS_TEMPLATE_FUNCTOR
#undef RPU_SWITCH_TRANS_TEMPLATE_FUNCTOR_OS
#undef RPU_SWITCH_TRANS_TEMPLATE_FUNCTOR_IMPLICIT

#undef DEFINE_PWU_KERNEL_PARAMETER
#undef DEFINE_PWU_KERNEL_BASE
} // namespace RPU
