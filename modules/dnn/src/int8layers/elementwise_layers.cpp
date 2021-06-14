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

#include "../precomp.hpp"
#include "layers_common.hpp"

#include <opencv2/dnn/shape_utils.hpp>
#include <iostream>

namespace cv
{
namespace dnn
{

class ActivationLayerInt8Impl CV_FINAL : public ActivationLayerInt8
{
public:
    ActivationLayerInt8Impl(const LayerParams &params)
    {
        setParamsFrom(params);
        activationMin = (int8_t)params.get<int>("activation_min", -128);
        activationMax = (int8_t)params.get<int>("activation_max", 127);
        activationLUT = (!blobs.empty()) ? blobs[0] : Mat();
        CV_Assert(activationMin <= activationMax);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        Layer::getMemoryShapes(inputs, requiredOutputs, outputs, internals);
        return true;
    }

    class Activation : public cv::ParallelLoopBody
    {
    public:
        const Mat* src;
        const Mat* lut;
        Mat* dst;
        int8_t actMin, actMax;
        int nstripes;

        Activation() : src(0), lut(0), dst(0), actMin(-128), actMax(127), nstripes(0){}

        static void run(const Mat& src, const Mat& lut, Mat& dst,
                        int8_t actMin, int8_t actMax, int nstripes)
        {
            Activation p;

            p.src = &src;
            p.lut = &lut;
            p.dst = &dst;
            p.actMin = actMin;
            p.actMax = actMax;
            p.nstripes = nstripes;

            parallel_for_(Range(0, nstripes), p, nstripes);
        }

        void operator()(const Range &r) const CV_OVERRIDE
        {
            const int8_t* table = lut->empty() ? 0 : lut->ptr<int8_t>();
            int nsamples = 1, outCn = 1;
            size_t planeSize = 1;

            if (src->dims > 1)
            {
                nsamples = src->size[0];
                outCn = src->size[1];
            }
            else
                outCn = src->size[0];

            for (int i = 2; i < src->dims; ++i)
                planeSize *= src->size[i];

            size_t stripeSize = (planeSize + nstripes - 1)/nstripes;
            size_t stripeStart = r.start*stripeSize;
            size_t stripeEnd = std::min(r.end*stripeSize, planeSize);
            int len = (int)(stripeEnd - stripeStart);

#if CV_SIMD128
            v_int8x16 qmin = v_setall_s8(actMin);
            v_int8x16 qmax = v_setall_s8(actMax);
#endif

            for( int i = 0; i < nsamples; i++ )
            {
                const int8_t* srcptr = src->ptr<int8_t>(i) + stripeStart;
                int8_t* dstptr = dst->ptr<int8_t>(i) + stripeStart;
                for( int cn = 0; cn < outCn; cn++, srcptr += planeSize, dstptr += planeSize )
                {
                    int i = 0;
#if CV_SIMD128
                    for( ; i <= len - 16; i += 16 )
                    {
                        if (table)
                        {
                            v_int8x16 out(table[srcptr[i] + 128], table[srcptr[i+1] + 128], table[srcptr[i+2] + 128], table[srcptr[i+3] + 128],
                                          table[srcptr[i+4] + 128], table[srcptr[i+5] + 128], table[srcptr[i+6] + 128], table[srcptr[i+7] + 128],
                                          table[srcptr[i+8] + 128], table[srcptr[i+9] + 128], table[srcptr[i+10] + 128], table[srcptr[i+11] + 128],
                                          table[srcptr[i+12] + 128], table[srcptr[i+13] + 128], table[srcptr[i+14] + 128], table[srcptr[i+15] + 128]);
                            v_store(dstptr + i, out);
                        }
                        else
                        {
                            v_store(dstptr + i, v_min(v_max(v_load(srcptr + i), qmin), qmax));
                        }
                    }
#endif
                    for( ; i < len; i++ )
                    {
                        if (table)
                            dstptr[i] = table[srcptr[i] + 128];
                        else
                            dstptr[i] = std::min(std::max(srcptr[i], actMin), actMax);
                    }
                }
            }
        }
    };

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const int nstripes = getNumThreads();
        for (size_t i = 0; i < inputs.size(); i++)
        {
            const Mat &src = inputs[i];
            Mat &dst = outputs[i];
            CV_Assert(src.size == dst.size && src.type() == dst.type() &&
                      src.isContinuous() && dst.isContinuous() && src.type() == CV_8S);

            Activation::run(src, activationLUT, dst, activationMin, activationMax, nstripes);
        }
    }

    void forwardSlice(const int8_t* src, const int8_t* lut, int8_t* dst, int len, size_t planeSize, int cn0, int cn1) const CV_OVERRIDE
    {
        for( int cn = cn0; cn < cn1; cn++, src += planeSize, dst += planeSize )
        {
            int i = 0;
#if CV_SIMD128
            for( ; i <= len - 16; i += 16 )
            {
                v_int8x16 out(lut[src[i] + 128], lut[src[i+1] + 128], lut[src[i+2] + 128], lut[src[i+3] + 128],
                              lut[src[i+4] + 128], lut[src[i+5] + 128], lut[src[i+6] + 128], lut[src[i+7] + 128],
                              lut[src[i+8] + 128], lut[src[i+9] + 128], lut[src[i+10] + 128], lut[src[i+11] + 128],
                              lut[src[i+12] + 128], lut[src[i+13] + 128], lut[src[i+14] + 128], lut[src[i+15] + 128]);
                v_store(dst + i, out);
            }
#endif
            for( ; i < len; i++ )
                dst[i] = lut[src[i] + 128];
        }
    }

    void forwardSlice(const int* src, const int* lut, int* dst, int len, size_t planeSize, int cn0, int cn1) const CV_OVERRIDE
    {
        for( int cn = cn0; cn < cn1; cn++, src += planeSize, dst += planeSize )
        {
            int i = 0;
#if CV_SIMD128
            for( ; i <= len - 16; i += 16 )
            {
                v_int32x4 out0(lut[src[i] + 128], lut[src[i+1] + 128], lut[src[i+2] + 128], lut[src[i+3] + 128]);
                v_int32x4 out1(lut[src[i+4] + 128], lut[src[i+5] + 128], lut[src[i+6] + 128], lut[src[i+7] + 128]);
                v_int32x4 out2(lut[src[i+8] + 128], lut[src[i+9] + 128], lut[src[i+10] + 128], lut[src[i+11] + 128]);
                v_int32x4 out3(lut[src[i+12] + 128], lut[src[i+13] + 128], lut[src[i+14] + 128], lut[src[i+15] + 128]);

                v_store(dst + i, out0);
                v_store(dst + i + 4, out1);
                v_store(dst + i + 8, out2);
                v_store(dst + i + 12, out3);
            }
#endif
            for( ; i < len; i++ )
                dst[i] = lut[src[i] + 128];
        }

    }

    Mat activationLUT;
    int8_t activationMin, activationMax;
};

Ptr<ActivationLayerInt8> ActivationLayerInt8::create(const LayerParams& params)
{
    return Ptr<ActivationLayerInt8>(new ActivationLayerInt8Impl(params));
}

}
}
