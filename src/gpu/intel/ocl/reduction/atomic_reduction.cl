/*******************************************************************************
* Copyright 2021-2025 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "gpu/intel/ocl/dispatch.h"
#include "gpu/intel/ocl/ocl_io.h"
#include "gpu/intel/ocl/ocl_math_utils.h"
#include "gpu/intel/ocl/ocl_utils.h"
#include "gpu/intel/ocl/reduction/ocl_reduction.h"
#include "gpu/intel/ocl/types_interop.h"

// Define accumulation functions
#define DEF_atomic_accumulate(dt) \
    dt atomic_accumulate(int alg, __global ATOMIC(dt) * atomic_p, dt data) { \
        switch (alg) { \
            case (REDUCTION_MAX): return atomic_max_global(atomic_p, data); \
            case (REDUCTION_MIN): return atomic_min_global(atomic_p, data); \
            case (REDUCTION_MEAN): \
            case (REDUCTION_SUM): return atomic_add_global(atomic_p, data); \
        } \
        printf("Atomic accumulate on unexpected algorithm\n"); \
        return 0; \
    }

#if ATOMIC_REDUCTION_SIZE > 1
#define MAYBE_ATOMIC(x) ATOMIC(x)
DEF_atomic_accumulate(float);
#else
#define MAYBE_ATOMIC(x) x
#endif

// Define how to read data
#define BLOCK_READ_DATA_T(data_ptr) \
    AS_VECT_DATA_T(VECT_BLOCK_READ((const __global BLOCK_DATA_T *)data_ptr))

#ifdef OCL_DEBUG
#define DUMP(str, ...) \
    do { \
        const size_t gid[3] \
                = {get_global_id(0), get_global_id(1), get_global_id(2)}; \
        const size_t lid[3] \
                = {get_local_id(0), get_local_id(1), get_local_id(2)}; \
        const size_t wgid[3] \
                = {get_group_id(0), get_group_id(1), get_group_id(2)}; \
        const size_t lin_g = get_global_linear_id(); \
        const size_t lin_l = get_local_linear_id(); \
        const uint sglid = get_sub_group_local_id(); \
        DEBUG_PRINT( \
                "gid=(%zu,%zu,%zu) lid=(%zu,%zu,%zu) " \
                "linear=(%zug/%zul/%usg): " str, \
                gid[0], gid[1], gid[2], lid[0], lid[1], lid[2], lin_g, lin_l, \
                sglid, ##__VA_ARGS__) \
    } while (0)
#else
#define DUMP(...)
#endif

__constant int REDUCTION_WI_COUNT = ATOMIC_REDUCTION_SIZE * LOCAL_SIZE;

KERNEL_ATTR
__kernel void atomic_reduce(__global SRC_DATA_T *src,
        __global MAYBE_ATOMIC(DST_DATA_T) * dst, int inner_size, off_t div,
        float power, float eps, off_t num_reductions,
        dispatch_gws_rt_params_t gws_params) {
    ASSUME(inner_size > 0);
    ASSUME(num_reductions > 0);
    const int local_idx = get_sub_group_id();
    const int sglid = get_sub_group_local_id();
    const int subgroup_size = get_max_sub_group_size();

    off_t atomic_idx = GWS_GET_OFF(ATOMIC, gws_params);

    off_t SRC_OFF = GWS_GET_OFF(SRC, gws_params) - sglid;
    off_t DST_OFF = GWS_GET_OFF(DST, gws_params);
    src += SRC_OFF;
    dst += DST_OFF;

    DUMP("Starting at %d src / %d dst... %p\n", SRC_OFF, DST_OFF, src);

    const int beg = local_idx + atomic_idx * LOCAL_SIZE;
    ASSUME(beg < REDUCTION_WI_COUNT);
    const int tail_count = num_reductions % REDUCTION_WI_COUNT;
    src += beg * inner_size;

    __constant int n_inner_iters = I_TILE_SIZE / GWS_SGS_DEFAULT;

    ACC_DT acc[n_inner_iters];
    unroll_for(int i = 0; i < I_TILE_SIZE; i++)
            init_acc(REDUCTION_ALG, &acc[i]);
    for (int red_off = beg; red_off < num_reductions;
            red_off += REDUCTION_WI_COUNT) {
        int iters_left = n_inner_iters;
        DUMP("Reduction iteration %d\n", red_off);
        // Load data using the largest possible loads
        while (iters_left >= 8) {
            DUMP("Unroll 8 iter: %d / %d left\n", iters_left, n_inner_iters);
            ACC_DT data[8];
            DUMP("SRC load at %p\n", src);
            block_load(&data[0], src, 8);
            unroll_for(int i = 0; i < 8; i++) {
                const int idx = i + (n_inner_iters - iters_left);
                acc[idx] = reduce(REDUCTION_ALG, acc[idx],
                        CONCAT2(into_, ACC_DT)(data[i]), power);
            }
            iters_left -= 8;
            src += subgroup_size * 8;
        }
        if (iters_left >= 4) {
            DUMP("Unroll 4 iter: %d / %d left\n", iters_left, n_inner_iters);
            ACC_DT data[4];
            DUMP("SRC load at %p\n", src);
            block_load(&data[0], src, 4);
            unroll_for(int i = 0; i < 4; i++) {
                const int idx = i + (n_inner_iters - iters_left);
                acc[idx] = reduce(REDUCTION_ALG, acc[idx],
                        CONCAT2(into_, ACC_DT)(data[i]), power);
            }
            iters_left -= 4;
            src += subgroup_size * 4;
        }
        if (iters_left >= 2) {
            DUMP("Unroll 2 iter: %d / %d left\n", iters_left, n_inner_iters);
            ACC_DT data[2];
            DUMP("SRC load at %p\n", src);
            block_load(&data[0], src, 2);
            unroll_for(int i = 0; i < 2; i++) {
                const int idx = i + (n_inner_iters - iters_left);
                acc[idx] = reduce(REDUCTION_ALG, acc[idx],
                        CONCAT2(into_, ACC_DT)(data[i]), power);
            }
            iters_left -= 2;
            src += subgroup_size * 2;
        }
        if (iters_left == 1) {
            DUMP("Final iter: %d / %d left\n", iters_left, n_inner_iters);
            ACC_DT data;
            DUMP("SRC load at %p\n", src);
            block_load(&data, src);
            const int idx = n_inner_iters - iters_left;
            acc[idx] = reduce(REDUCTION_ALG, acc[idx],
                    CONCAT2(into_, ACC_DT)(data), power);
            iters_left--;
            src += subgroup_size;
        }
        ASSUME(iters_left == 0);

        // Offset to the next reduction index
        src += inner_size * (REDUCTION_WI_COUNT - 1);
    }

    // Store results to SLM
    __local ACC_DT local_acc_buf[LOCAL_SIZE][I_TILE_SIZE];
    unroll_for(int i = 0; i < n_inner_iters; i++) {
        local_acc_buf[local_idx][sglid + i * subgroup_size] = acc[i];
    }

    // Wait for all subgroups to finish
    barrier(CLK_LOCAL_MEM_FENCE);

    // In the first subgroup of each work group:
    if (local_idx == 0) {
        // Perform the SLM reduction
        ACC_DT local_acc[n_inner_iters];
        unroll_for(int i = 0; i < n_inner_iters; i++)
                init_acc(SECONDARY_REDUCTION_ALG, &local_acc[i]);

        unroll_for(int slm_off = 0; slm_off < LOCAL_SIZE; slm_off++) {
            unroll_for(int vect_off = 0; vect_off < n_inner_iters; vect_off++) {
                const int idx = vect_off * subgroup_size + sglid;
                const ACC_DT slm_data = local_acc_buf[slm_off][idx];
                local_acc[vect_off] = reduce(SECONDARY_REDUCTION_ALG,
                        local_acc[vect_off], slm_data, power);
            }
        }

        // Finalize data, then (atomically) accumulate into to dst
        // XXX: There's a bug in the compiler that makes the following code break when
        // VECT_DT_N = 1. Instead, here's a workaround:
        unroll_for(int i = 0; i < n_inner_iters; i++) {
            float f = finalize(
                    REDUCTION_ALG, into_float(local_acc[i]), div, power, eps);
#if ATOMIC_REDUCTION_SIZE > 1
            const DST_DATA_T old_val
                    = atomic_accumulate(SECONDARY_REDUCTION_ALG,
                            &dst[v * subgroup_size], TO_DST(f));
#else
            dst[i * subgroup_size] = CONCAT2(into_, DST_DATA_T)(f);
#endif
        }
    }
}
