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

namespace cv
{
namespace dnn
{

class FullyConnectedLayerInt8Impl CV_FINAL : public InnerProductLayerInt8
{
public:
    enum { VEC_ALIGN = 32 };
    FullyConnectedLayerInt8Impl(const LayerParams& params)
    {
        setParamsFrom(params);
        output_zp = params.get<int>("zeropoints");
        axis = params.get<int>("axis", 1);
        if (blobs.size() == 3)
        {
            // blobs[0] - Weights
            // blobs[1] - Bias fused with offset
            // blobs[2] - Multipliers for output stage
            int numOutput = params.get<int>("num_output");
            int innerSize = (int)blobs[0].total() / numOutput;

            CV_Assert(blobs[0].dims >= 2 && (size_t)(innerSize * numOutput) == blobs[0].total());
            CV_Assert((size_t)numOutput == blobs[1].total());

            weightsMat = blobs[0] = blobs[0].reshape(1, numOutput);
            int vecsize = weightsMat.cols;
            if (vecsize % VEC_ALIGN != 0)
            {
                int vecsize_aligned = (int)alignSize(vecsize, VEC_ALIGN);
                Mat weightsBuf(weightsMat.rows, vecsize_aligned, weightsMat.type());
                Mat wpadding = weightsBuf.colRange(vecsize, vecsize_aligned);
                wpadding.setTo(Scalar::all(0));
                weightsMat = weightsBuf.colRange(0, vecsize);
                blobs[0].copyTo(weightsMat);
            }
            biasMat = blobs[1] = blobs[1].reshape(1, 1);
        }
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &) const CV_OVERRIDE
    {
        int numOutput, cAxis;
        CV_CheckEQ(inputs.size(), (size_t)1, "");
        CV_CheckEQ(blobs[0].dims, 2, "");
        numOutput = blobs[0].size[0];
        CV_Assert((size_t)numOutput == blobs[1].total());
        cAxis = normalize_axis(axis, inputs[0]);

        MatShape outShape(cAxis + 1);
        for (int i = 0; i < cAxis; ++i)
            outShape[i] = inputs[0][i];
        outShape.back() = numOutput;

        outputs.resize(1, outShape);
        return false;
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual bool setActivation(const Ptr<ActivationLayer>& layer) CV_OVERRIDE
    {
        if (activ.empty() || layer.empty())
        {
            activ = layer;
            return !activ.empty();
        }
        else
            return false;
    }

    class FullyConnected : public ParallelLoopBody
    {
    public:
        FullyConnected() : srcMat(0), weights(0), biasMat(0), additionalParams(0), activ(0), dstMat(0), nstripes(0), outZp(0),
                           useAVX(false), useAVX2(false), useAVX512(false) {}

        static void run(const Mat& srcMat, const Mat& weights, const Mat& biasMat, const Mat& additionalParams,
                        Mat& dstMat, const ActivationLayer* activ, int nstripes, int outZp)
        {
            CV_Assert( srcMat.dims == 2 && srcMat.cols == weights.cols &&
                       dstMat.rows == srcMat.rows && dstMat.cols == weights.rows &&
                       srcMat.type() == weights.type() && srcMat.type() == CV_8S &&
                       dstMat.type() == CV_32S && biasMat.type() == CV_32S &&
                       biasMat.isContinuous() && (int)biasMat.total() == dstMat.cols );

            FullyConnected p;

            p.srcMat = &srcMat;
            p.weights = &weights;
            p.biasMat = &biasMat;
            p.additionalParams = &additionalParams;
            p.dstMat = &dstMat;
            p.nstripes = nstripes;
            p.outZp = outZp;
            p.activ = activ;
            p.useAVX = checkHardwareSupport(CPU_AVX);
            p.useAVX2 = checkHardwareSupport(CPU_AVX2);
            p.useAVX512 = CV_CPU_HAS_SUPPORT_AVX512_SKX;

            parallel_for_(Range(0, nstripes), p, nstripes);
        }

        void operator()(const Range& r) const CV_OVERRIDE
        {
            int valign = FullyConnectedLayerInt8Impl::VEC_ALIGN;
            int nsamples = srcMat->rows;
            int nw0 = weights->rows;
            int k, vecsize = srcMat->cols;
            int vecsize_aligned = (int)alignSize(vecsize, VEC_ALIGN);
            size_t total = (size_t)nsamples*nw0;
            size_t stripeSize = (total + nstripes - 1)/nstripes;
            size_t stripeStart = r.start*stripeSize;
            size_t stripeEnd = r.end == nstripes ? total : std::min(r.end*stripeSize, total);
            size_t wstep = weights->step1();
            AutoBuffer<int8_t> srcbuf(vecsize_aligned + valign);
            int8_t* sptr = alignPtr(srcbuf.data(), (int)(valign*sizeof(int8_t)));

            for( k = vecsize; k < vecsize_aligned; k++ )
                sptr[k] = 0;

            for( size_t ofs = stripeStart; ofs < stripeEnd; )
            {
                int sampleIdx = (int)(ofs / nw0);
                int delta = (int)(ofs - (size_t)sampleIdx*nw0);
                const int8_t* sptr_ = srcMat->ptr<int8_t>(sampleIdx);
                const int8_t* wptr = weights->ptr<int8_t>(delta);
                int* dptr = dstMat->ptr<int>(sampleIdx) + delta;
                const int* biasptr = biasMat->ptr<int>() + delta;
                const int* multptr = additionalParams->ptr<int>() + delta;
                int nw = std::min(nw0 - delta, (int)(stripeEnd - ofs));

                memcpy(sptr, sptr_, vecsize*sizeof(sptr[0]));
            #if CV_TRY_AVX512_SKX
                if( useAVX512 )
                    opt_AVX512_SKX::fastGEMM1T( sptr, wptr, wstep, biasptr, multptr, dptr, nw, vecsize, outZp );
                else
            #endif
            #if CV_TRY_AVX2
                if( useAVX2 )
                    opt_AVX2::fastGEMM1T( sptr, wptr, wstep, biasptr, multptr, dptr, nw, vecsize, outZp );
                else
            #endif
                {
                    int i = 0;
            #if CV_SIMD128 || useAVX
                    for( ; i  <= nw - 4; i += 4, wptr += 4*wstep )
                    {
                        v_int32x4 vs0 = v_setall_s32(0);
                        v_int32x4 vs1 = v_setall_s32(0);
                        v_int32x4 vs2 = v_setall_s32(0);
                        v_int32x4 vs3 = v_setall_s32(0);
                        v_int32x4 s = v_load(biasptr + i);
                        v_int32x4 mult = v_load(multptr + i);

                        for( k = 0; k < vecsize; k += 16 )
                        {
                            v_int8x16 v = v_load_aligned(sptr + k);
                            vs0 = v_dotprod_expand_fast(v, v_load_aligned(wptr + k), vs0);
                            vs1 = v_dotprod_expand_fast(v, v_load_aligned(wptr + wstep + k), vs1);
                            vs2 = v_dotprod_expand_fast(v, v_load_aligned(wptr + wstep*2 + k), vs2);
                            vs3 = v_dotprod_expand_fast(v, v_load_aligned(wptr + wstep*3 + k), vs3);
                        }

                        s += v_reduce_sum4(vs0, vs1, vs2, vs3);
                        v_store(dptr + i, v_outputStage(s, mult, outZp));
                    }
            #endif

                    for( ; i < nw; i++, wptr += wstep )
                    {
                        int s0 = biasptr[i];
                        int mult0 = multptr[i];

                        for( k = 0; k < vecsize; k++ )
                        {
                            int8_t v = sptr[k];
                            s0 += (int)v*wptr[k];
                        }
                        int out0 = outZp + ((s0*mult0 + (1 << 21)) >> 22);
                        dptr[i] = std::min(std::max(out0, -128), 127);
                    }
                }

                /*if(activ)
                    activ->forwardSlice(dptr, dptr, 1, 1, delta, delta + nw);*/

                ofs += nw;
            }
        }

        const Mat *srcMat, *weights, *biasMat, *additionalParams;
        const ActivationLayer* activ;
        Mat* dstMat;
        int nstripes, outZp;
        bool useAVX;
        bool useAVX2;
        bool useAVX512;
    };

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> input, output;
        inputs_arr.getMatVector(input);
        outputs_arr.getMatVector(output);

        int axisCan = normalize_axis(axis, input[0].dims);
        int outerSize = input[0].total(0, axisCan);
        Mat srcMat = input[0].reshape(1, outerSize);

        Mat dstMat = output[0].reshape(1, outerSize);
        Mat dstMatInt32= Mat(shape(dstMat), CV_32S);

        const int nstripes = getNumThreads();
        FullyConnected::run(srcMat, weightsMat, biasMat, blobs[2], dstMatInt32, activ.get(), nstripes, output_zp);
        dstMatInt32.convertTo(dstMat, CV_8S);
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_UNUSED(inputs); // suppress unused variable warning
        long flops = 0;

        int innerSize = blobs[0].size[1];
        for(int i = 0; i < outputs.size(); i++)
        {
            flops += CV_BIG_INT(3)*innerSize*total(outputs[i]);
        }

        return flops;

    }

    Mat weightsMat, biasMat;
    Ptr<ActivationLayer> activ;
};

Ptr<InnerProductLayerInt8> InnerProductLayerInt8::create(const LayerParams& params)
{
    return Ptr<InnerProductLayerInt8>(new FullyConnectedLayerInt8Impl(params));
}

}
}
