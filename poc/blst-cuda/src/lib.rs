// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "bls12_377")]
use ark_bls12_377::{Fr, G1Affine, G1Projective};
#[cfg(feature = "bls12_381")]
use ark_bls12_381::{Fr, G1Affine, G1Projective};
#[cfg(feature = "bn254")]
use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ec::AffineCurve;
use ark_ff::PrimeField;
use ark_std::Zero;
use blst::*;

sppark::cuda_error!();

pub mod util;

#[derive(Debug, Clone)]
#[repr(C)]
pub struct FFITraitObject {
    ptr: usize,
}

#[cfg_attr(feature = "quiet", allow(improper_ctypes))]
extern "C" {
    fn mult_pippenger(
        out: *mut blst_p1,
        points: *const blst_p1_affine,
        npoints: usize,
        scalars: *const blst_scalar,
    ) -> cuda::Error;

    fn mult_pippenger_inf(
        out: *mut G1Projective,
        points_with_infinity: *const G1Affine,
        npoints: usize,
        scalars: *const Fr,
        ffi_affine_sz: usize,
    ) -> cuda::Error;

    fn mult_pippenger_halo2(
        out: *const FFITraitObject,
        points: *const FFITraitObject,
        npoints: usize,
        scalars: *const FFITraitObject,
        ffi_affine_sz: usize,
        gpu_idx: usize,
    ) -> cuda::Error;

    fn benchmark_scalar_mul_sppark(
        scalar: *const FFITraitObject,
        loop_times: usize,
        repeat_times: usize,
    );

    fn benchmark_point_mixed_add_sppark(
        point: *const FFITraitObject,
        loop_times: usize,
        repeat_times: usize,
    );
}

pub fn multi_scalar_mult(
    points: &[blst_p1_affine],
    scalars: &[blst_scalar],
) -> blst_p1 {
    let npoints = points.len();
    if npoints != scalars.len() {
        panic!("length mismatch")
    }

    let mut ret = blst_p1::default();
    let err =
        unsafe { mult_pippenger(&mut ret, &points[0], npoints, &scalars[0]) };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }
    ret
}

pub fn multi_scalar_mult_arkworks<G: AffineCurve>(
    points: &[G],
    scalars: &[<G::ScalarField as PrimeField>::BigInt],
) -> G::Projective {
    let npoints = points.len();
    if npoints != scalars.len() {
        panic!("length mismatch")
    }

    let mut ret = G::Projective::zero();
    let err = unsafe {
        mult_pippenger_inf(
            &mut ret as *mut _ as *mut G1Projective,
            points as *const _ as *const G1Affine,
            npoints,
            scalars as *const _ as *const Fr,
            std::mem::size_of::<G1Affine>(),
        )
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }

    ret
}


// for halo2_gpu
use std::mem;
use group::Group;
pub use pairing::arithmetic::*;

pub fn multi_scalar_mult_halo2<C: CurveAffine>(
    points: &[C],
    scalars: &[C::Scalar],
    gpu_idx: usize
) -> C::Curve {
    let npoints = points.len();
    if npoints != scalars.len() {
        panic!("length mismatch")
    }

    let mut out = C::Curve::identity();
    let err =
        unsafe {
            let mut out_ori = vec![];
            let mut points_ori = vec![];
            let mut scalars_ori = vec![];
            out_ori.push(mem::transmute(&out));
            points_ori.push(mem::transmute(&points[0]));
            scalars_ori.push(mem::transmute(&scalars[0]));
            mult_pippenger_halo2(
                out_ori.as_ptr(), 
                points_ori.as_ptr(), 
                npoints, 
                scalars_ori.as_ptr(),
                mem::size_of::<C>(),
                gpu_idx)
        };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }
    out
}

use pairing::bn256::Fr as Halo2Fr;
pub fn sppark_benchmark_scalar_mul(
    scalars: &[Halo2Fr],
    loop_times: usize, 
    repeat_times: usize
) {
    unsafe {
        let mut scalar_ori = vec![];
        scalar_ori.push(mem::transmute(&scalars[1]));
        benchmark_scalar_mul_sppark(scalar_ori.as_ptr(), loop_times, repeat_times);
    }
}

pub fn sppark_benchmark_point_mixed_add<C: CurveAffine>(
    points: &[C],
    loop_times: usize, 
    repeat_times: usize
) {
    unsafe {
        let mut point_ori = vec![];
        point_ori.push(mem::transmute(&points[1])); //bypass the first point
        benchmark_point_mixed_add_sppark(point_ori.as_ptr(), loop_times, repeat_times);
    }
}
