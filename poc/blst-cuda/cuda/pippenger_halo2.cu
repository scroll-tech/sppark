// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>

#if defined(FEATURE_BLS12_381)
# include <ff/bls12-381.hpp>
#elif defined(FEATURE_BLS12_377)
# include <ff/bls12-377.hpp>
#elif defined(FEATURE_BN254)
# include <ff/alt_bn128.hpp>
#else
# error "no FEATURE"
#endif

#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_t affine_t;
typedef fr_t scalar_t;

#include <msm/pippenger.cuh>

struct FFITraitObject{
    uint64_t *ptr;
};
typedef FFITraitObject Point;
typedef FFITraitObject Affine;
typedef FFITraitObject Scalar;

#ifndef __CUDA_ARCH__
extern "C"
RustError mult_pippenger_halo2(
    const Point *out,
    const Affine *point,
    uint64_t npoints,
    const Scalar *scalar,
    size_t ffi_affine_sz,
    uint64_t gpu_idx)
{
    cudaSetDevice(gpu_idx);
    point_t* ret = (point_t*)out->ptr;
    const affine_t* points = (affine_t*)(point->ptr);
    const scalar_t* scalars = (scalar_t*)(scalar->ptr);
    return mult_pippenger<bucket_t>(ret, points, npoints, scalars, true);
}
#endif
