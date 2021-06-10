/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "opencv2/core/hal/intrin.hpp"

namespace cv {
namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void fastConv( const int8_t* weights, size_t wstep, const int* bias,
               const int8_t* rowbuf, int* output, const int* outShape,
               int blockSize, int vecsize, int vecsize_aligned, int outZp,
               const int* multiplier, const float* relu,
               bool initOutput, bool finalOutput );
void fastDepthwiseConv( const int8_t* wptr,
                        int kernel_h, int kernel_w,
                        int stride_h, int stride_w,
                        int dilation_h, int dilation_w,
                        int pad_t, int pad_l,
                        const int* biasptr, const int* multptr,
                        const int8_t* inptr_,
                        int height, int width,
                        int* outptr_,
                        int out_d, int outH, int outW,
                        int inpZp, int outZp );
void fastGEMM1T( const int8_t* vec, const int8_t* weights,
                 size_t wstep, const int* bias, const int* multiplier,
                 int* dst, int nvecs, int vecsize, int outZp );

#if !defined(CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY) && CV_AVX2
#define OPENCV_FMADD_EPI8(_Tpvec, func) \
    inline _Tpvec _##func##_fmaddepi8_epi32(const _Tpvec& a, const _Tpvec& b, const _Tpvec& c) \
    { \
        _Tpvec even_a = _##func##_srai_epi16(_##func##_bslli_epi128(a, 1), 8); \
        _Tpvec odd_a  = _##func##_srai_epi16(a, 8);                            \
                                                                               \
        _Tpvec even_b = _##func##_srai_epi16(_##func##_bslli_epi128(b, 1), 8); \
        _Tpvec odd_b  = _##func##_srai_epi16(b, 8);                            \
                                                                               \
        _Tpvec prod0  = _##func##_madd_epi16(even_a, even_b);                  \
        _Tpvec prod1  = _##func##_madd_epi16(odd_a, odd_b);                    \
        return _##func##_add_epi32(_##func##_add_epi32(prod0, prod1), c);      \
    }
OPENCV_FMADD_EPI8(__m256i, mm256)
//OPENCV_FMADD_EPI8(__m512i, mm512)

#define OPENCV_QUANT_OUTPUT_STAGE(_Tpvec, func) \
    inline _Tpvec OutputStage(const _Tpvec& accum, const _Tpvec& mult, const int& outZp) \
    { \
        _Tpvec mul = _##func##_mullo_epi32(accum, mult);                                 \
        _Tpvec nudge = _##func##_set1_epi32(1 << 21);                                    \
        _Tpvec rshr = _##func##_srai_epi32(_##func##_add_epi32(mul, nudge), 22);         \
        _Tpvec output = _##func##_add_epi32(_##func##_set1_epi32(outZp), rshr);          \
                                                                                         \
        _Tpvec qmin = _##func##_set1_epi32(-128), qmax = _##func##_set1_epi32(127);      \
        return _##func##_min_epi32(_##func##_max_epi32(output, qmin), qmax);             \
    }
OPENCV_QUANT_OUTPUT_STAGE(__m128i, mm)
OPENCV_QUANT_OUTPUT_STAGE(__m256i, mm256)

static inline int OutputStage(const int& accum, const int& multiplier, const int& outZp)
{
    int mul = accum * multiplier;
    int output = outZp + ((mul + (1 << 21)) >> 22);
    return std::min(std::max(output, -128), 127);
}

enum { FASCONV_BASE_VECSZ = 4 };

void fastConv( const int8_t* weights, size_t wstep, const int* bias,
               const int8_t* rowbuf, int* output, const int* outShape,
               int blockSize, int vecsize, int vecsize_aligned, int outZp,
               const int* multiplier, const float* relu,
               bool initOutput, bool finalOutput )
{
    int outCn = outShape[1];
    size_t outPlaneSize = outShape[2]*outShape[3];
    int CV_DECL_ALIGNED(16) maskbuf[FASCONV_BASE_VECSZ] = {0};
    int rsz = blockSize % FASCONV_BASE_VECSZ;
    for( int i = 0; i < rsz; i++ )
        maskbuf[FASCONV_BASE_VECSZ - i - 1] = -1;
    __m128 mask = _mm_loadu_ps((const float*)maskbuf);

    // now compute dot product of the weights
    // and im2row-transformed part of the tensor
    for( int i = 0; i < outCn; i += 3 )
    {
        const int8_t* wptr0 = weights + i*wstep;
        const int8_t* wptr1 = wptr0 + wstep;
        const int8_t* wptr2 = wptr1 + wstep;
        int* outptr0 = output + i*outPlaneSize;
        int* outptr1 = outptr0 + outPlaneSize;
        int* outptr2 = outptr1 + outPlaneSize;
        int bias0 = bias[i], bias1 = bias[i+1], bias2 = bias[i+2];
        int mult0 = multiplier[i], mult1 = multiplier[i+1], mult2 = multiplier[i+2];

        if( i+2 >= outCn )
        {
            wptr2 = wptr1;
            outptr2 = outptr1;
            bias2 = bias1;
            mult2 = mult1;

            if( i+1 >= outCn )
            {
                wptr2 = wptr1 = wptr0;
                outptr2 = outptr1 = outptr0;
                bias2 = bias1 = bias0;
                mult2 = mult1 = mult0;
            }
        }
        int j = 0;
        for( ; j < blockSize; j += FASCONV_BASE_VECSZ )
        {
            bool tail = false;
            if (j + FASCONV_BASE_VECSZ > blockSize)
            {
                if (j == 0)
                    break;
                j = blockSize - FASCONV_BASE_VECSZ;
                tail = true;
            }
            int k = 0;
            const int8_t* rptr = rowbuf + j*vecsize_aligned;

            __m256i vs00 = _mm256_setzero_si256(), vs01 = _mm256_setzero_si256(),
                    vs02 = _mm256_setzero_si256(), vs03 = _mm256_setzero_si256(),
                    vs10 = _mm256_setzero_si256(), vs11 = _mm256_setzero_si256(),
                    vs12 = _mm256_setzero_si256(), vs13 = _mm256_setzero_si256(),
                    vs20 = _mm256_setzero_si256(), vs21 = _mm256_setzero_si256(),
                    vs22 = _mm256_setzero_si256(), vs23 = _mm256_setzero_si256();

            /* TODO : Fix AVX-512 path. Segmentation fault in Conv2D Tests.
#if CV_AVX512_SKX // AVX512VL is necessary to avoid register spilling
            if (vecsize >= 64)
            {
                __m512i vs00_5 = _mm512_setzero_si512(), vs01_5 = _mm512_setzero_si512(),
                        vs02_5 = _mm512_setzero_si512(), vs03_5 = _mm512_setzero_si512(),
                        vs10_5 = _mm512_setzero_si512(), vs11_5 = _mm512_setzero_si512(),
                        vs12_5 = _mm512_setzero_si512(), vs13_5 = _mm512_setzero_si512(),
                        vs20_5 = _mm512_setzero_si512(), vs21_5 = _mm512_setzero_si512(),
                        vs22_5 = _mm512_setzero_si512(), vs23_5 = _mm512_setzero_si512();

                for (; k <= vecsize - 64; k += 64, rptr += 64)
                {
                    __m512i w0 = _mm512_load_si512(wptr0 + k);
                    __m512i w1 = _mm512_load_si512(wptr1 + k);
                    __m512i w2 = _mm512_load_si512(wptr2 + k);
                    __m512i r0 = _mm512_load_si512(rptr);

                    vs00_5 = _mm512_fmaddepi8_epi32(w0, r0, vs00_5);
                    vs10_5 = _mm512_fmaddepi8_epi32(w1, r0, vs10_5);
                    vs20_5 = _mm512_fmaddepi8_epi32(w2, r0, vs20_5);

                    r0 = _mm512_load_si512(rptr + vecsize_aligned);
                    vs01_5 = _mm512_fmaddepi8_epi32(w0, r0, vs01_5);
                    vs11_5 = _mm512_fmaddepi8_epi32(w1, r0, vs11_5);
                    vs21_5 = _mm512_fmaddepi8_epi32(w2, r0, vs21_5);

                    r0 = _mm512_load_si512(rptr + vecsize_aligned*2);
                    vs02_5 = _mm512_fmaddepi8_epi32(w0, r0, vs02_5);
                    vs12_5 = _mm512_fmaddepi8_epi32(w1, r0, vs12_5);
                    vs22_5 = _mm512_fmaddepi8_epi32(w2, r0, vs22_5);

                    r0 = _mm512_load_si512(rptr + vecsize_aligned*3);
                    vs03_5 = _mm512_fmaddepi8_epi32(w0, r0, vs03_5);
                    vs13_5 = _mm512_fmaddepi8_epi32(w1, r0, vs13_5);
                    vs23_5 = _mm512_fmaddepi8_epi32(w2, r0, vs23_5);
                }

                // now fold the 512 bit accumulator vectors into 256 bit vectors so that the AVX2 code can finish
                // the tail of the vector

                vs00 = _mm256_add_epi32( _mm512_extracti32x8_epi32(vs00_5, 0), _mm512_extracti32x8_epi32(vs00_5, 1));
                vs10 = _mm256_add_epi32( _mm512_extracti32x8_epi32(vs10_5, 0), _mm512_extracti32x8_epi32(vs10_5, 1));
                vs20 = _mm256_add_epi32( _mm512_extracti32x8_epi32(vs20_5, 0), _mm512_extracti32x8_epi32(vs20_5, 1));

                vs01 = _mm256_add_epi32( _mm512_extracti32x8_epi32(vs01_5, 0), _mm512_extracti32x8_epi32(vs01_5, 1));
                vs11 = _mm256_add_epi32( _mm512_extracti32x8_epi32(vs11_5, 0), _mm512_extracti32x8_epi32(vs11_5, 1));
                vs21 = _mm256_add_epi32( _mm512_extracti32x8_epi32(vs21_5, 0), _mm512_extracti32x8_epi32(vs21_5, 1));

                vs02 = _mm256_add_epi32( _mm512_extracti32x8_epi32(vs02_5, 0), _mm512_extracti32x8_epi32(vs02_5, 1));
                vs12 = _mm256_add_epi32( _mm512_extracti32x8_epi32(vs12_5, 0), _mm512_extracti32x8_epi32(vs12_5, 1));
                vs22 = _mm256_add_epi32( _mm512_extracti32x8_epi32(vs22_5, 0), _mm512_extracti32x8_epi32(vs22_5, 1));

                vs03 = _mm256_add_epi32( _mm512_extracti32x8_epi32(vs03_5, 0), _mm512_extracti32x8_epi32(vs03_5, 1));
                vs13 = _mm256_add_epi32( _mm512_extracti32x8_epi32(vs13_5, 0), _mm512_extracti32x8_epi32(vs13_5, 1));
                vs23 = _mm256_add_epi32( _mm512_extracti32x8_epi32(vs23_5, 0), _mm512_extracti32x8_epi32(vs23_5, 1));
            }
#endif
            */
            for (; k < vecsize; k += 32, rptr += 32 )
            {
                __m256i w0 = _mm256_load_si256((const __m256i*)(wptr0 + k));
                __m256i w1 = _mm256_load_si256((const __m256i*)(wptr1 + k));
                __m256i w2 = _mm256_load_si256((const __m256i*)(wptr2 + k));
                __m256i r0 = _mm256_load_si256((const __m256i*)rptr);

                vs00 = _mm256_fmaddepi8_epi32(w0, r0, vs00);
                vs10 = _mm256_fmaddepi8_epi32(w1, r0, vs10);
                vs20 = _mm256_fmaddepi8_epi32(w2, r0, vs20);

                r0 = _mm256_load_si256((const __m256i*)(rptr + vecsize_aligned));
                vs01 = _mm256_fmaddepi8_epi32(w0, r0, vs01);
                vs11 = _mm256_fmaddepi8_epi32(w1, r0, vs11);
                vs21 = _mm256_fmaddepi8_epi32(w2, r0, vs21);

                r0 = _mm256_load_si256((const __m256i*)(rptr + vecsize_aligned*2));
                vs02 = _mm256_fmaddepi8_epi32(w0, r0, vs02);
                vs12 = _mm256_fmaddepi8_epi32(w1, r0, vs12);
                vs22 = _mm256_fmaddepi8_epi32(w2, r0, vs22);

                r0 = _mm256_load_si256((const __m256i*)(rptr + vecsize_aligned*3));
                vs03 = _mm256_fmaddepi8_epi32(w0, r0, vs03);
                vs13 = _mm256_fmaddepi8_epi32(w1, r0, vs13);
                vs23 = _mm256_fmaddepi8_epi32(w2, r0, vs23);
            }

            __m256i t0 = _mm256_hadd_epi32(_mm256_hadd_epi32(vs00, vs01), _mm256_hadd_epi32(vs02, vs03));
            __m256i t1 = _mm256_hadd_epi32(_mm256_hadd_epi32(vs10, vs11), _mm256_hadd_epi32(vs12, vs13));
            __m256i t2 = _mm256_hadd_epi32(_mm256_hadd_epi32(vs20, vs21), _mm256_hadd_epi32(vs22, vs23));

            t0 = _mm256_add_epi32(t0, _mm256_permute2x128_si256(t0, t0, 1));
            t1 = _mm256_add_epi32(t1, _mm256_permute2x128_si256(t1, t1, 1));
            t2 = _mm256_add_epi32(t2, _mm256_permute2x128_si256(t2, t2, 1));

            __m128i s0, s1, s2;

            if( initOutput )
            {
                s0 = _mm_set1_epi32(bias0);
                s1 = _mm_set1_epi32(bias1);
                s2 = _mm_set1_epi32(bias2);
            }
            else
            {
                s0 = _mm_loadu_si128((__m128i*)(outptr0 + j));
                s1 = _mm_loadu_si128((__m128i*)(outptr1 + j));
                s2 = _mm_loadu_si128((__m128i*)(outptr2 + j));
            }

            s0 = _mm_add_epi32(s0, _mm256_castsi256_si128(t0));
            s1 = _mm_add_epi32(s1, _mm256_castsi256_si128(t1));
            s2 = _mm_add_epi32(s2, _mm256_castsi256_si128(t2));

            if( finalOutput )
            {
                s0 =  OutputStage(s0, _mm_set1_epi32(mult0), outZp);
                s1 =  OutputStage(s1, _mm_set1_epi32(mult1), outZp);
                s2 =  OutputStage(s2, _mm_set1_epi32(mult2), outZp);
            }
            if( tail )
            {
                s0 =  _mm_castps_si128(_mm_blendv_ps(_mm_loadu_ps((const float*)outptr0 + j),  _mm_castsi128_ps(s0), mask));
                s1 =  _mm_castps_si128(_mm_blendv_ps(_mm_loadu_ps((const float*)outptr1 + j),  _mm_castsi128_ps(s1), mask));
                s2 =  _mm_castps_si128(_mm_blendv_ps(_mm_loadu_ps((const float*)outptr2 + j),  _mm_castsi128_ps(s2), mask));
            }
            _mm_storeu_si128((__m128i*)(outptr0 + j), s0);
            _mm_storeu_si128((__m128i*)(outptr1 + j), s1);
            _mm_storeu_si128((__m128i*)(outptr2 + j), s2);
        }

        for( ; j <= blockSize - 2; j += 2 )
        {
            const int8_t* rptr0 = rowbuf + j*vecsize_aligned;
            const int8_t* rptr1 = rowbuf + (j+1)*vecsize_aligned;
            int s00, s01, s10, s11, s20, s21;

            if( initOutput )
            {
                s00 = s01 = bias0;
                s10 = s11 = bias1;
                s20 = s21 = bias2;
            }
            else
            {
                s00 = outptr0[j]; s01 = outptr0[j+1];
                s10 = outptr1[j]; s11 = outptr1[j+1];
                s20 = outptr2[j]; s21 = outptr2[j+1];
            }

            for( int k = 0; k < vecsize; k++ )
            {
                int8_t w0 = wptr0[k], w1 = wptr1[k], w2 = wptr2[k];
                int8_t r = rptr0[k];
                s00 += (int)w0*r; s10 += (int)w1*r; s20 += (int)w2*r;
                r = rptr1[k];
                s01 += (int)w0*r; s11 += (int)w1*r; s21 += (int)w2*r;
            }

            if( finalOutput )
            {
                s00 = OutputStage(s00, mult0, outZp);
                s01 = OutputStage(s01, mult0, outZp);
                s10 = OutputStage(s10, mult1, outZp);
                s11 = OutputStage(s11, mult1, outZp);
                s20 = OutputStage(s20, mult2, outZp);
                s21 = OutputStage(s21, mult2, outZp);
            }
            outptr0[j] = s00;
            outptr0[j+1] = s01;
            outptr1[j] = s10;
            outptr1[j+1] = s11;
            outptr2[j] = s20;
            outptr2[j+1] = s21;
        }

        for( ; j < blockSize; j++ )
        {
            const int8_t* rptr0 = rowbuf + j*vecsize_aligned;
            int s00, s10, s20;

            if( initOutput )
            {
                s00 = bias0;
                s10 = bias1;
                s20 = bias2;
            }
            else
            {
                s00 = outptr0[j];
                s10 = outptr1[j];
                s20 = outptr2[j];
            }

            for( int k = 0; k < vecsize; k++ )
            {
                int8_t w0 = wptr0[k], w1 = wptr1[k], w2 = wptr2[k];
                int8_t r = rptr0[k];
                s00 += (int)w0*r; s10 += (int)w1*r; s20 += (int)w2*r;
            }

            if( finalOutput )
            {
                s00 = OutputStage(s00, mult0, outZp);
                s10 = OutputStage(s10, mult1, outZp);
                s20 = OutputStage(s20, mult2, outZp);
            }
            outptr0[j] = s00;
            outptr1[j] = s10;
            outptr2[j] = s20;
        }
    }
    _mm256_zeroupper();
}

static inline void _mm256_expand_mul_add(const __m256i& a, const __m256i& b,
                                         __m256i& out0, __m256i& out1, __m256i& out2, __m256i& out3)
{
    __m256i a0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a));
    __m256i a1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 1));

    __m256i b0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b));
    __m256i b1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b, 1));

    __m256i a0b0 = _mm256_mullo_epi16(a0, b0);
    __m256i a1b1 = _mm256_mullo_epi16(a1, b1);

    out0 = _mm256_add_epi32(out0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(a0b0)));
    out1 = _mm256_add_epi32(out1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(a0b0, 1)));
    out2 = _mm256_add_epi32(out2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(a1b1)));
    out3 = _mm256_add_epi32(out3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(a1b1, 1)));
}

static inline void _mm256_load_deinterleave(const int8_t* ptr, __m256i& a, __m256i& b)
{
    __m256i t0 = _mm256_loadu_si256((const __m256i*)ptr);
    __m256i t1 = _mm256_loadu_si256((const __m256i*)(ptr + 32));

    const __m256i sh = _mm256_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
                                        0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
    __m256i p0 = _mm256_shuffle_epi8(t0, sh);
    __m256i p1 = _mm256_shuffle_epi8(t1, sh);
    __m256i lo = _mm256_permute2x128_si256(p0, p1, 0 + 2*16);
    __m256i hi = _mm256_permute2x128_si256(p0, p1, 1 + 3*16);
    a = _mm256_unpacklo_epi64(lo, hi);
    b = _mm256_unpackhi_epi64(lo, hi);
}

void fastDepthwiseConv( const int8_t* wptr,
                     int kernel_h, int kernel_w,
                     int stride_h, int stride_w,
                     int dilation_h, int dilation_w,
                     int pad_t, int pad_l,
                     const int* biasptr, const int* multptr,
                     const int8_t* inptr_,
                     int height, int width,
                     int* outptr_,
                     int out_d, int outH, int outW,
                     int inpZp, int outZp)
{
    const int8_t w00_ = wptr[0], w01_ = wptr[1], w02_ = wptr[2],
                 w10 = wptr[3], w11 = wptr[4], w12 = wptr[5],
                 w20_ = wptr[6], w21_ = wptr[7], w22_ = wptr[8];
    int outW1 = min(outW, (width - dilation_w*(kernel_w - 1) + pad_l)/stride_w);
    int bias = biasptr[out_d], mult = multptr[out_d];
    int biasCopy;

    for (int out_i = 0; out_i < outH; out_i++)
    {
        int in_i = out_i * stride_h - pad_t, out_j = 0;
        const int8_t* imgptr0 = inptr_ + in_i*width;
        const int8_t* imgptr1 = imgptr0 + dilation_h*width;
        const int8_t* imgptr2 = imgptr0 + (dilation_h*2)*width;
        int8_t w00 = w00_, w01 = w01_, w02 = w02_;
        int8_t w20 = w20_, w21 = w21_, w22 = w22_;
        int out, out1;
        biasCopy = bias;
        if (in_i < 0)
        {
            biasCopy += inpZp * (w00 + w01 + w02);
            w00 = w01 = w02 = 0;
            imgptr0 = imgptr1;
        }
        else if (in_i + dilation_h*(kernel_h-1) >= height)
        {
            biasCopy += inpZp * (w20 + w21 + w22);
            w20 = w21 = w22 = 0;
            imgptr2 = imgptr1;
        }
        int* outptr = outptr_ + out_i*outW;
        if (pad_l > 0)
        {
            out = (int)imgptr0[0]*w01 + (int)imgptr0[dilation_w]*w02 +
                  (int)imgptr1[0]*w11 + (int)imgptr1[dilation_w]*w12 +
                  (int)imgptr2[0]*w21 + (int)imgptr2[dilation_w]*w22 +
                  biasCopy + inpZp*(w00 + w10 + w20);
            out1 = outZp + ((out*mult + (1 << 21)) >> 22);
            outptr[0] = std::min(std::max(out1, -128), 127);
            out_j = 1;
        }

        if (stride_w == 1 || (stride_w == 2 && dilation_w == 1))
        {
            const int VECSZ = 32;
            __m256i vw00 = _mm256_set1_epi8(w00), vw01 = _mm256_set1_epi8(w01), vw02 = _mm256_set1_epi8(w02),
                    vw10 = _mm256_set1_epi8(w10), vw11 = _mm256_set1_epi8(w11), vw12 = _mm256_set1_epi8(w12),
                    vw20 = _mm256_set1_epi8(w20), vw21 = _mm256_set1_epi8(w21), vw22 = _mm256_set1_epi8(w22);
            __m256i vbias = _mm256_set1_epi32(biasCopy), vmult = _mm256_set1_epi32(mult);
            __m256i vout0, vout1, vout2, vout3;

            if( stride_w == 1 )
            {
                for( ; out_j < outW1; out_j += VECSZ )
                {
                    if (out_j + VECSZ > outW1)
                    {
                        if (out_j <= pad_l)
                            break;
                        out_j = outW1 - VECSZ;
                    }
                    int in_j = out_j * stride_w - pad_l;
                    __m256i v00 = _mm256_loadu_si256((const __m256i*)(imgptr0 + in_j)),
                            v01 = _mm256_loadu_si256((const __m256i*)(imgptr0 + in_j + dilation_w)),
                            v02 = _mm256_loadu_si256((const __m256i*)(imgptr0 + in_j + dilation_w*2)),
                            v10 = _mm256_loadu_si256((const __m256i*)(imgptr1 + in_j)),
                            v11 = _mm256_loadu_si256((const __m256i*)(imgptr1 + in_j + dilation_w)),
                            v12 = _mm256_loadu_si256((const __m256i*)(imgptr1 + in_j + dilation_w*2)),
                            v20 = _mm256_loadu_si256((const __m256i*)(imgptr2 + in_j)),
                            v21 = _mm256_loadu_si256((const __m256i*)(imgptr2 + in_j + dilation_w)),
                            v22 = _mm256_loadu_si256((const __m256i*)(imgptr2 + in_j + dilation_w*2));

                    vout0 = vout1 = vout2 = vout3 = vbias;
                    _mm256_expand_mul_add(v00, vw00, vout0, vout1, vout2, vout3);
                    _mm256_expand_mul_add(v01, vw01, vout0, vout1, vout2, vout3);
                    _mm256_expand_mul_add(v02, vw02, vout0, vout1, vout2, vout3);
                    _mm256_expand_mul_add(v10, vw10, vout0, vout1, vout2, vout3);
                    _mm256_expand_mul_add(v11, vw11, vout0, vout1, vout2, vout3);
                    _mm256_expand_mul_add(v12, vw12, vout0, vout1, vout2, vout3);
                    _mm256_expand_mul_add(v20, vw20, vout0, vout1, vout2, vout3);
                    _mm256_expand_mul_add(v21, vw21, vout0, vout1, vout2, vout3);
                    _mm256_expand_mul_add(v22, vw22, vout0, vout1, vout2, vout3);

                    _mm256_storeu_si256((__m256i*)(outptr + out_j), OutputStage(vout0, vmult, outZp));
                    _mm256_storeu_si256((__m256i*)(outptr + out_j + 8), OutputStage(vout1, vmult, outZp));
                    _mm256_storeu_si256((__m256i*)(outptr + out_j + 16), OutputStage(vout2, vmult, outZp));
                    _mm256_storeu_si256((__m256i*)(outptr + out_j + 24), OutputStage(vout3, vmult, outZp));
                }
            }
            else
            {
                for( ; out_j < outW1; out_j += VECSZ )
                {
                    if (out_j + VECSZ > outW1)
                    {
                        if (out_j <= pad_l)
                            break;
                        out_j = outW1 - VECSZ;
                    }
                    int in_j = out_j * stride_w - pad_l;
                    __m256i v00, v01, v02, v10, v11, v12, v20, v21, v22, unused;
                    _mm256_load_deinterleave(imgptr0 + in_j, v00, v01);
                    _mm256_load_deinterleave(imgptr0 + in_j + 2, v02, unused);
                    _mm256_load_deinterleave(imgptr1 + in_j, v10, v11);
                    _mm256_load_deinterleave(imgptr1 + in_j + 2, v12, unused);
                    _mm256_load_deinterleave(imgptr2 + in_j, v20, v21);
                    _mm256_load_deinterleave(imgptr2 + in_j + 2, v22, unused);

                    vout0 = vout1 = vout2 = vout3 = vbias;
                    _mm256_expand_mul_add(v00, vw00, vout0, vout1, vout2, vout3);
                    _mm256_expand_mul_add(v01, vw01, vout0, vout1, vout2, vout3);
                    _mm256_expand_mul_add(v02, vw02, vout0, vout1, vout2, vout3);
                    _mm256_expand_mul_add(v10, vw10, vout0, vout1, vout2, vout3);
                    _mm256_expand_mul_add(v11, vw11, vout0, vout1, vout2, vout3);
                    _mm256_expand_mul_add(v12, vw12, vout0, vout1, vout2, vout3);
                    _mm256_expand_mul_add(v20, vw20, vout0, vout1, vout2, vout3);
                    _mm256_expand_mul_add(v21, vw21, vout0, vout1, vout2, vout3);
                    _mm256_expand_mul_add(v22, vw22, vout0, vout1, vout2, vout3);

                    _mm256_storeu_si256((__m256i*)(outptr + out_j), OutputStage(vout0, vmult, outZp));
                    _mm256_storeu_si256((__m256i*)(outptr + out_j + 8), OutputStage(vout1, vmult, outZp));
                    _mm256_storeu_si256((__m256i*)(outptr + out_j + 16), OutputStage(vout2, vmult, outZp));
                    _mm256_storeu_si256((__m256i*)(outptr + out_j + 24), OutputStage(vout3, vmult, outZp));
                }
            }
        }

        for (; out_j < outW1; out_j++)
        {
            int in_j = out_j * stride_w - pad_l;
            out = (int)imgptr0[in_j]*w00 + (int)imgptr0[in_j + dilation_w]*w01 + (int)imgptr0[in_j + dilation_w*2]*w02 +
                  (int)imgptr1[in_j]*w10 + (int)imgptr1[in_j + dilation_w]*w11 + (int)imgptr1[in_j + dilation_w*2]*w12 +
                  (int)imgptr2[in_j]*w20 + (int)imgptr2[in_j + dilation_w]*w21 + (int)imgptr2[in_j + dilation_w*2]*w22 + biasCopy;
            out1 = outZp + ((out*mult + (1 << 21)) >> 22);
            outptr[out_j] = std::min(std::max(out1, -128), 127);
        }

        for (; out_j < outW; out_j++ )
        {
            int in_j0 = out_j * stride_w - pad_l, in_j1 = in_j0 + dilation_w, in_j2 = in_j0 + dilation_w*2;
            int s0 = 1, s1 = 1, s2 = 1;
            if (in_j0 >= width)
            {
                in_j0 = 0;
                s0 = 0;
                biasCopy += inpZp*(w00 + w10 + w20);
            }
            if (in_j1 >= width)
            {
                in_j1 = 0;
                s1 = 0;
                biasCopy += inpZp*(w01 + w11 + w21);
            }
            if (in_j2 >= width)
            {
                in_j2 = 0;
                s2 = 0;
                biasCopy += inpZp*(w02 + w12 + w22);
            }
            out = (int)imgptr0[in_j0]*w00*s0 + (int)imgptr0[in_j1]*w01*s1 + (int)imgptr0[in_j2]*w02*s2 +
                  (int)imgptr1[in_j0]*w10*s0 + (int)imgptr1[in_j1]*w11*s1 + (int)imgptr1[in_j2]*w12*s2 +
                  (int)imgptr2[in_j0]*w20*s0 + (int)imgptr2[in_j1]*w21*s1 + (int)imgptr2[in_j2]*w22*s2 + biasCopy;
            out1 = outZp + ((out*mult + (1 << 21)) >> 22);
            outptr[out_j] = std::min(std::max(out1, -128), 127);
        }
    }
    _mm256_zeroupper();
}

// dst = vec * weights^t + bias
void fastGEMM1T( const int8_t* vec, const int8_t* weights,
                 size_t wstep, const int* bias, const int* multiplier,
                 int* dst, int nvecs, int vecsize, int outZp )
{
    int i = 0;

    for( ; i <= nvecs - 8; i += 8 )
    {
        const int8_t* wptr = weights + i*wstep;
        __m256i vs0 = _mm256_setzero_si256(), vs1 = _mm256_setzero_si256(),
                vs2 = _mm256_setzero_si256(), vs3 = _mm256_setzero_si256(),
                vs4 = _mm256_setzero_si256(), vs5 = _mm256_setzero_si256(),
                vs6 = _mm256_setzero_si256(), vs7 = _mm256_setzero_si256();

        for( int k = 0; k < vecsize; k += 32, wptr += 32 )
        {
            __m256i v = _mm256_load_si256((const __m256i*)(vec + k));

            vs0 = _mm256_fmaddepi8_epi32(_mm256_load_si256((const __m256i*)wptr), v, vs0);
            vs1 = _mm256_fmaddepi8_epi32(_mm256_load_si256((const __m256i*)(wptr + wstep)), v, vs1);
            vs2 = _mm256_fmaddepi8_epi32(_mm256_load_si256((const __m256i*)(wptr + wstep*2)), v, vs2);
            vs3 = _mm256_fmaddepi8_epi32(_mm256_load_si256((const __m256i*)(wptr + wstep*3)), v, vs3);
            vs4 = _mm256_fmaddepi8_epi32(_mm256_load_si256((const __m256i*)(wptr + wstep*4)), v, vs4);
            vs5 = _mm256_fmaddepi8_epi32(_mm256_load_si256((const __m256i*)(wptr + wstep*5)), v, vs5);
            vs6 = _mm256_fmaddepi8_epi32(_mm256_load_si256((const __m256i*)(wptr + wstep*6)), v, vs6);
            vs7 = _mm256_fmaddepi8_epi32(_mm256_load_si256((const __m256i*)(wptr + wstep*7)), v, vs7);
        }

        __m256i s0 = _mm256_hadd_epi32(_mm256_hadd_epi32(vs0, vs1), _mm256_hadd_epi32(vs2, vs3));
        __m256i s1 = _mm256_hadd_epi32(_mm256_hadd_epi32(vs4, vs5), _mm256_hadd_epi32(vs6, vs7));

        s0 = _mm256_add_epi32(s0, _mm256_permute2x128_si256(s0, s0, 1));
        s1 = _mm256_add_epi32(s1, _mm256_permute2x128_si256(s1, s1, 1));

        __m128i t0 = _mm_add_epi32(_mm256_castsi256_si128(s0), _mm_loadu_si128((__m128i*)(bias + i)));
        __m128i t1 = _mm_add_epi32(_mm256_castsi256_si128(s1), _mm_loadu_si128((__m128i*)(bias + i + 4)));

        _mm_storeu_si128((__m128i*)(dst + i), OutputStage(t0, _mm_loadu_si128((__m128i*)(multiplier + i)), outZp));
        _mm_storeu_si128((__m128i*)(dst + i + 4), OutputStage(t1, _mm_loadu_si128((__m128i*)(multiplier + i + 4)), outZp));
    }

    for( ; i < nvecs; i++ )
    {
        const int8_t* wptr = weights + i*wstep;
        __m256i vs0 = _mm256_setzero_si256();

        for( int k = 0; k < vecsize; k += 32, wptr += 32 )
        {
            __m256i v = _mm256_load_si256((const __m256i*)(vec + k));
            vs0 = _mm256_fmaddepi8_epi32(_mm256_load_si256((const __m256i*)wptr), v, vs0);
        }

        __m256i s0 = _mm256_hadd_epi32(_mm256_hadd_epi32(vs0, vs0), vs0);
        s0 = _mm256_add_epi32(s0, _mm256_permute2x128_si256(s0, s0, 1));
        int temp = _mm_extract_epi32(_mm256_castsi256_si128(s0), 0);
        dst[i] = OutputStage(temp + bias[i], multiplier[i], outZp);
    }

    _mm256_zeroupper();
}
#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END
}} // namespace
