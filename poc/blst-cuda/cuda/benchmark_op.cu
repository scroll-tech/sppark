#include <cuda.h>
#include <cstdio>

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
#include <msm/pippenger.cuh> //needs previous typedef


struct FFITraitObject{
    uint64_t *ptr;
};
typedef FFITraitObject Point;
typedef FFITraitObject Affine;
typedef FFITraitObject Scalar;


#ifndef __CUDA_ARCH__
extern "C"
void benchmark_scalar_mul_sppark(const Scalar *scalar, uint64_t loop_times, uint64_t repeat_times) {

    scalar_t *d_scalar;
    cudaMalloc(&d_scalar, sizeof(scalar_t));
    cudaMemcpy((uint64_t*)d_scalar, scalar->ptr, sizeof(scalar_t), cudaMemcpyHostToDevice);
    for(uint64_t i = 0; i < repeat_times; i++){
        op_scalar_mul_sppark<<<1, 1>>>(d_scalar, loop_times);
    }
    cudaFree(d_scalar);
}
#endif


#ifndef __CUDA_ARCH__
extern "C"
void benchmark_point_mixed_add_sppark(const Point *point, uint64_t loop_times, uint64_t repeat_times) {

    const size_t ELT_LIMBS = 4 * sizeof(uint64_t); // FF : 256bit = 32bytes
    const size_t AFF_POINT_LIMBS = 2 * ELT_LIMBS;  // X Y
    const size_t JAC_POINT_LIMBS = 4 * ELT_LIMBS;  // X Y Z

    affine_t *d_point;
    bucket_t *d_res;
    cudaMalloc(&d_point, JAC_POINT_LIMBS);
    cudaMalloc(&d_res, sizeof(bucket_t));
    cudaMemcpy((uint64_t*)d_point, point->ptr, AFF_POINT_LIMBS, cudaMemcpyHostToDevice);
    for(uint64_t i = 0; i < repeat_times; i++){
        op_point_mixed_add_sppark<<<1, 1>>>(d_point, loop_times, d_res);
    }
    cudaFree(d_point);
    cudaFree(d_res);
}
#endif
