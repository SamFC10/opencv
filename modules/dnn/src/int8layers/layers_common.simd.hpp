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
               const int* offset, const int* multiplier, const float* relu,
               bool initOutput, bool finalOutput );

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
OPENCV_FMADD_EPI8(__m512i, mm512)

static inline __m128i OutputStage(const __m128i& accum, const __m128i& multiplier, const int& outZp)
{
    __m128i mul = _mm_mullo_epi32(accum, multiplier);
    __m128i nudge = _mm_set1_epi32(1 << 21);
    __m128i rshr = _mm_srai_epi32(_mm_add_epi32(mul, nudge), 22);
    __m128i output = _mm_add_epi32(_mm_set1_epi32(outZp), rshr);

    __m128i qmin = _mm_set1_epi32(-128), qmax = _mm_set1_epi32(127);
    return _mm_min_epi32(_mm_max_epi32(output, qmin), qmax);
}

static inline int OutputStage(const int& accum, const int& multiplier, const int& outZp)
{
    int mul = accum * multiplier;
    int rshr = (mul + (1 << 21)) >> 22;
    int output = outZp + rshr;
    return std::min(std::max(output, -128), 127);
}

enum { FASCONV_BASE_VECSZ = 4 };

void fastConv( const int8_t* weights, size_t wstep, const int* bias,
               const int8_t* rowbuf, int* output, const int* outShape,
               int blockSize, int vecsize, int vecsize_aligned, int outZp,
               const int* offset, const int* multiplier, const float* relu,
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
        int offset0 = offset[i], offset1 = offset[i+1], offset2 = offset[i+2];

        if( i+2 >= outCn )
        {
            wptr2 = wptr1;
            outptr2 = outptr1;
            bias2 = bias1;
            mult2 = mult1;
            offset2 = offset1;

            if( i+1 >= outCn )
            {
                wptr2 = wptr1 = wptr0;
                outptr2 = outptr1 = outptr0;
                bias2 = bias1 = bias0;
                mult2 = mult1 = mult0;
                offset2 = offset1 = offset0;
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
                /*
                 * now fold the 512 bit accumulator vectors into 256 bit vectors so that the AVX2 code can finish
                 * the tail of the vector
                 */
                
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

            t0 = _mm256_add_epi32(t0, _mm256_permute2f128_si256(t0, t0, 1));
            t1 = _mm256_add_epi32(t1, _mm256_permute2f128_si256(t1, t1, 1));
            t2 = _mm256_add_epi32(t2, _mm256_permute2f128_si256(t2, t2, 1));

            __m128i s0, s1, s2;

            if( initOutput )
            {
                s0 = _mm_setzero_si128();
                s1 = _mm_setzero_si128();
                s2 = _mm_setzero_si128();
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
                s0 = _mm_add_epi32(s0, _mm_set1_epi32(offset0 + bias0));
                s1 = _mm_add_epi32(s1, _mm_set1_epi32(offset1 + bias1));
                s2 = _mm_add_epi32(s2, _mm_set1_epi32(offset2 + bias2));

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
                s00 = s01 = 0;
                s10 = s11 = 0;
                s20 = s21 = 0;
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
                s00 += (int16_t)w0*(int16_t)r; s10 += (int16_t)w1*(int16_t)r; s20 += (int16_t)w2*(int16_t)r;
                r = rptr1[k];
                s01 += (int16_t)w0*(int16_t)r; s11 += (int16_t)w1*(int16_t)r; s21 += (int16_t)w2*(int16_t)r;
            }

            if( finalOutput )
            {
                s00 += offset0 + bias0; s01 += offset0 + bias0;
                s10 += offset1 + bias1; s11 += offset1 + bias1;
                s20 += offset2 + bias2; s21 += offset2 + bias2;

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
                s00 = 0;
                s10 = 0;
                s20 = 0;
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
                s00 += (int16_t)w0*(int16_t)r; s10 += (int16_t)w1*(int16_t)r; s20 += (int16_t)w2*(int16_t)r;
            }

            if( finalOutput )
            {
                s00 += offset0 + bias0; 
                s10 += offset1 + bias1; 
                s20 += offset2 + bias2;

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
#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END
}} // namespace