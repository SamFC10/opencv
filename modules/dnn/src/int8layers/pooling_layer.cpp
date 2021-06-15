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
#include "opencv2/core/hal/intrin.hpp"

#include <float.h>
#include <algorithm>
#include <numeric>
using std::max;
using std::min;

namespace cv
{
namespace dnn
{

class PoolingLayerInt8Impl CV_FINAL : public PoolingLayerInt8
{
public:
    PoolingLayerInt8Impl(const LayerParams& params)
    {
        computeMaxIdx = false;
        globalPooling = false;
        isGlobalPooling = std::vector<bool>(3, false);

        hasDynamicShapes = params.get<bool>("has_dynamic_shapes", false);
        shapesInitialized = !hasDynamicShapes;

        if (params.has("pool") || params.has("kernel_size") ||
            params.has("kernel_w") || params.has("kernel_h"))
        {
            String pool = toLowerCase(params.get<String>("pool", "max"));
            if (pool == "max")
                type = MAX;
            else if (pool == "ave")
                type = AVE;
            else
                CV_Error(Error::StsBadArg, "Unknown pooling type \"" + pool + "\"");

            getPoolingKernelParams(params, kernel_size, isGlobalPooling, pads_begin, pads_end, strides, padMode);
            globalPooling = isGlobalPooling[0] || isGlobalPooling[1] || isGlobalPooling[2];
        }
        else
            CV_Error(Error::StsBadArg, "Cannot determine pooling type");
        setParamsFrom(params);
        ceilMode = params.get<bool>("ceil_mode", true);
        spatialScale = params.get<float>("spatial_scale", 1);
        avePoolPaddedArea = params.get<bool>("ave_pool_padded_area", true);
        output_zp = params.get<int>("zeropoints", 0);
    }

    void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert(!inputs.empty());
        CV_Assert(outputs.size() == 1);

        std::vector<int> inp;
        std::vector<int> out;
        for (int i = 2; i < inputs[0].dims; i++) {
            inp.push_back(inputs[0].size[i]);
            out.push_back(outputs[0].size[i]);
        }
        if (globalPooling) {
            std::vector<size_t> finalKernel;
            for (int i = 0; i < inp.size(); i++) {
                int idx = isGlobalPooling.size() - inp.size() + i;
                finalKernel.push_back(isGlobalPooling[idx] ? inp[i] : kernel_size[idx]);
             }
             kernel_size = finalKernel;
         }

        getConvPoolPaddings(inp, kernel_size, strides, padMode, pads_begin, pads_end);

        if (inputs[0].dims == 3)
        {
            // Pool1D
            kernel_size.assign(1, kernel_size[0]);
            strides.assign(1, strides[0]);
            pads_begin.assign(1, pads_begin[0]);
            pads_end.assign(1, pads_end[0]);
        }
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        if (backendId == DNN_BACKEND_OPENCV)
        {
            if (kernel_size.size() == 3)
                return preferableTarget == DNN_TARGET_CPU;
            if (kernel_size.size() <= 2)
                return true;
            else
                return false;
        }
        return false;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        switch (type)
        {
            case MAX:
            {
                CV_Assert_N(inputs.size() == 1, outputs.size() == 1);
                maxPooling(inputs[0], outputs[0]);
                break;
            }
            case AVE:
                CV_Assert_N(inputs.size() == 1, outputs.size() == 1);
                avePooling(inputs[0], outputs[0]);
                break;
            default:
                CV_Error(Error::StsNotImplemented, "Not implemented");
                break;
        }
    }

    class PoolingInvoker : public ParallelLoopBody
    {
    public:
        const Mat* src, *rois;
        Mat *dst;
        int pad_l, pad_t, pad_r, pad_b;
        bool avePoolPaddedArea;
        int nstripes, outZp;
        std::vector<int> ofsbuf;
        int poolingType;
        float spatialScale;

        std::vector<size_t> pads_begin, pads_end;
        std::vector<size_t> kernel_size;
        std::vector<size_t> strides;

        PoolingInvoker() : src(0), rois(0), dst(0), pad_l(0), pad_t(0), pad_r(0), pad_b(0),
                           avePoolPaddedArea(false), nstripes(0), outZp(0),
                           poolingType(MAX), spatialScale(0){}

        static void run(const Mat& src, const Mat& rois, Mat& dst,
                        std::vector<size_t> kernel_size, std::vector<size_t> strides,
                        std::vector<size_t> pads_begin, std::vector<size_t> pads_end,
                        bool avePoolPaddedArea, int poolingType, float spatialScale,
                        int outZp, int nstripes)
        {
            CV_Assert_N(
                      src.isContinuous(), dst.isContinuous(),
                      src.type() == CV_8S, src.type() == dst.type(),
                      src.dims == 3 || src.dims == 4 || src.dims == 5, dst.dims == 3 || dst.dims == 4 || dst.dims == 5,
                      src.size[0] == dst.size[0], src.size[1] == dst.size[1], rois.empty());

            PoolingInvoker p;

            bool isPool1D = src.dims == 3;
            bool isPool3D = src.dims == 5;

            p.src = &src;
            p.rois = &rois;
            p.dst = &dst;

            p.kernel_size = kernel_size;
            p.strides = strides;
            p.pads_begin = pads_begin;
            p.pads_end = pads_end;

            p.pad_l = pads_begin.back();
            p.pad_t = isPool1D ? 0 : pads_begin[pads_begin.size() - 2];
            p.pad_r = pads_end.back();
            p.pad_b = isPool1D ? 0 : pads_end[pads_end.size() - 2];

            p.avePoolPaddedArea = avePoolPaddedArea;
            p.nstripes = nstripes;
            p.outZp = outZp;
            p.poolingType = poolingType;
            p.spatialScale = spatialScale;

            int height = isPool1D ? 1 : src.size[src.dims - 2];
            int width = src.size[src.dims - 1];

            int kernel_d = isPool3D ? kernel_size[0] : 1;
            int kernel_h = isPool1D ? 1 : kernel_size[kernel_size.size() - 2];
            int kernel_w = kernel_size.back();

            p.ofsbuf.resize(kernel_d * kernel_h * kernel_w);
            for (int i = 0; i < kernel_d; ++i) {
                for (int j = 0; j < kernel_h; ++j) {
                    for (int k = 0; k < kernel_w; ++k) {
                        p.ofsbuf[i * kernel_h * kernel_w + j * kernel_w + k] = width * height * i + width * j + k;
                    }
                }
            }

            parallel_for_(Range(0, nstripes), p, nstripes);
        }

        void operator()(const Range& r) const CV_OVERRIDE
        {
            int channels = dst->size[1];

            bool isPool3D = src->dims == 5;
            bool isPool2D = src->dims == 4;
            bool isPool1D = src->dims == 3;
            int depth = isPool3D? dst->size[2] : 1;
            int height = isPool1D? 1 : dst->size[dst->dims - 2];
            int width = dst->size[dst->dims - 1];

            int inp_depth = isPool3D? src->size[2] : 1;
            int inp_height = isPool1D? 1 : src->size[src->dims - 2];
            int inp_width = src->size[src->dims - 1];

            size_t total = dst->total();
            size_t stripeSize = (total + nstripes - 1)/nstripes;
            size_t stripeStart = r.start*stripeSize;
            size_t stripeEnd = std::min(r.end*stripeSize, total);

            int kernel_d = isPool3D? kernel_size[0] : 1;
            int kernel_h = isPool1D? 1 : kernel_size[kernel_size.size() - 2];
            int kernel_w = kernel_size.back();

            int stride_d = isPool3D? strides[0] : 0;
            int stride_h = isPool1D? 1 :strides[strides.size() - 2];
            int stride_w = strides.back();

#if CV_SIMD128
            const int* ofsptr = (const int*)&ofsbuf[0];
            if (poolingType == MAX && !ofsptr)
                CV_Error(Error::StsBadArg, "ofsbuf should be initialized in this mode");
#endif

            for( size_t ofs0 = stripeStart; ofs0 < stripeEnd; )
            {
                size_t ofs = ofs0;
                int x0 = (int)(ofs % width);
                ofs /= width;
                int y0 = (int)(ofs % height);
                ofs /= height;

                int d0 = (int)(ofs % depth);
                ofs /= depth;

                int c = (int)(ofs % channels);
                int n = (int)(ofs / channels);
                int ystart, yend;
                int dstart = 0, dend = 1;

                const int8_t *srcData = 0;
                int pad_d_begin = (pads_begin.size() == 3) ? pads_begin[0] : 0;
                dstart = d0 * stride_d - pad_d_begin;
                dend = min(dstart + kernel_d, (int)(inp_depth + pads_end[0]));

                ystart = y0 * stride_h - pad_t;
                yend = min(ystart + kernel_h, inp_height + pad_b);
                srcData = src->ptr<int8_t>(n, c);

                int ddelta = dend - dstart;
                dstart = max(dstart, 0);
                dend = min(dend, inp_depth);
                int ydelta = yend - ystart;
                ystart = max(ystart, 0);
                yend = min(yend, inp_height);
                int8_t *dstData = &dst->ptr<int8_t>(n, c, d0)[y0 * width];

                int delta = std::min((int)(stripeEnd - ofs0), width - x0);
                ofs0 += delta;
                int x1 = x0 + delta;

                if( poolingType == MAX )
                    for( ; x0 < x1; x0++ )
                    {
                        int xstart = x0 * stride_w - pad_l;
                        int xend = min(xstart + kernel_w, inp_width);
                        xstart = max(xstart, 0);
                        if (xstart >= xend || ystart >= yend)
                        {
                            dstData[x0] = (int8_t)outZp;
                            continue;
                        }
#if CV_SIMD128
                        if( isPool2D && xstart > 0 && x0 + 15 < x1 && (x0 + 15) * stride_w - pad_l + kernel_w < inp_width )
                        {
                            v_int8x16 max_val0 = v_setall_s8(-128);
                            if( yend - ystart == kernel_h )
                            {
                                const int8_t* srcData1 = srcData + ystart*inp_width + xstart;
                                if( stride_w == 1 )
                                    for (int k = 0; k < kernel_w*kernel_h; k++)
                                    {
                                        int index = ofsptr[k];
                                        v_int8x16 v0 = v_load(srcData1 + index);
                                        max_val0 = v_max(max_val0, v0);
                                    }
                                else if( stride_w == 2 )
                                    for (int k = 0; k < kernel_w*kernel_h; k++)
                                    {
                                        int index = ofsptr[k];
                                        v_int8x16 v0, dummy;
                                        v_load_deinterleave(srcData1 + index, v0, dummy);
                                        max_val0 = v_max(max_val0, v0);
                                    }
                                else
                                    for (int k = 0; k < kernel_w*kernel_h; k++)
                                    {
                                        int index = ofsptr[k];
                                        v_int8x16 v0(srcData1[index], srcData1[index + stride_w],
                                                     srcData1[index + stride_w*2], srcData1[index + stride_w*3],
                                                     srcData1[index + stride_w*4], srcData1[index + stride_w*5],
                                                     srcData1[index + stride_w*6], srcData1[index + stride_w*7],
                                                     srcData1[index + stride_w*8], srcData1[index + stride_w*9],
                                                     srcData1[index + stride_w*10], srcData1[index + stride_w*11],
                                                     srcData1[index + stride_w*12], srcData1[index + stride_w*13],
                                                     srcData1[index + stride_w*14], srcData1[index + stride_w*15]);
                                        max_val0 = v_max(max_val0, v0);
                                    }
                            }
                            else
                            {
                                for (int y = ystart; y < yend; ++y)
                                {
                                    for (int x = xstart; x < xend; ++x)
                                    {
                                        const int index = y * inp_width + x;
                                        v_int8x16 v0(srcData[index], srcData[index + stride_w],
                                                     srcData[index + stride_w*2], srcData[index + stride_w*3],
                                                     srcData[index + stride_w*4], srcData[index + stride_w*5],
                                                     srcData[index + stride_w*6], srcData[index + stride_w*7],
                                                     srcData[index + stride_w*8], srcData[index + stride_w*9],
                                                     srcData[index + stride_w*10], srcData[index + stride_w*11],
                                                     srcData[index + stride_w*12], srcData[index + stride_w*13],
                                                     srcData[index + stride_w*14], srcData[index + stride_w*15]);
                                        max_val0 = v_max(max_val0, v0);
                                    }
                                }
                            }
                            v_store(dstData + x0, max_val0);
                            x0 += 15;
                        }
                        else
#else
                        CV_UNUSED(isPool2D);
#endif
                        if( isPool1D )
                        {
                            const int8_t* first = srcData + xstart;
                            const int8_t* last = srcData + xend;
                            const int8_t* max_elem = std::max_element(first, last);
                            if (max_elem != last)
                                dstData[x0] = *max_elem;
                        }
                        else
                        {
                            int8_t max_val = -128;
                            for (int d = dstart; d < dend; ++d) {
                                for (int y = ystart; y < yend; ++y) {
                                    for (int x = xstart; x < xend; ++x) {
                                        const int index = d * inp_width * inp_height + y * inp_width + x;
                                        int8_t val = srcData[index];
                                        max_val = std::max(max_val, val);
                                    }
                                }
                            }
                            dstData[x0] = max_val;
                        }
                    }
                else if (poolingType == AVE)
                {
                    for( ; x0 < x1; ++x0)
                    {
                        int xstart = x0 * stride_w - pad_l;
                        int xend = min(xstart + kernel_w, inp_width + pad_r);
                        int xdelta = xend - xstart;
                        xstart = max(xstart, 0);
                        xend = min(xend, inp_width);
                        float inv_kernel_area = avePoolPaddedArea ? xdelta * ydelta * ddelta :
                                                ((dend - dstart) * (yend - ystart) * (xend - xstart));
                        inv_kernel_area = 1.f / inv_kernel_area;
#if CV_SIMD128
                        if( isPool2D && xstart > 0 && x0 + 15 < x1 && (x0 + 15) * stride_w - pad_l + kernel_w < inp_width )
                        {
                            v_int32x4 sum_val0 = v_setzero_s32(), sum_val1 = v_setzero_s32(),
                                      sum_val2 = v_setzero_s32(), sum_val3 = v_setzero_s32();
                            v_float32x4 ikarea = v_setall_f32(inv_kernel_area);

                            for (int y = ystart; y < yend; ++y)
                            {
                                for (int x = xstart; x < xend; ++x)
                                {
                                    const int index = y * inp_width + x;
                                    v_int32x4 v0((int)srcData[index], (int)srcData[index + stride_w],
                                                 (int)srcData[index + stride_w*2], (int)srcData[index + stride_w*3]);
                                    v_int32x4 v1((int)srcData[index + stride_w*4], (int)srcData[index + stride_w*5],
                                                 (int)srcData[index + stride_w*6], (int)srcData[index + stride_w*7]);
                                    v_int32x4 v2((int)srcData[index + stride_w*8], (int)srcData[index + stride_w*9],
                                                 (int)srcData[index + stride_w*10], (int)srcData[index + stride_w*11]);
                                    v_int32x4 v3((int)srcData[index + stride_w*12], (int)srcData[index + stride_w*13],
                                                 (int)srcData[index + stride_w*14], (int)srcData[index + stride_w*15]);
                                    sum_val0 += v0;
                                    sum_val1 += v1;
                                    sum_val2 += v2;
                                    sum_val3 += v3;
                                }
                            }

                            sum_val0 = v_round(v_reinterpret_as_f32(sum_val0)*ikarea);
                            sum_val1 = v_round(v_reinterpret_as_f32(sum_val1)*ikarea);
                            sum_val2 = v_round(v_reinterpret_as_f32(sum_val2)*ikarea);
                            sum_val3 = v_round(v_reinterpret_as_f32(sum_val3)*ikarea);

                            v_store(dstData + x0, v_pack(v_pack(sum_val0, sum_val1), v_pack(sum_val2, sum_val3)));
                            x0 += 15;
                        }
                        else
#endif
                        if( isPool1D )
                        {
                            const int8_t* first = srcData + xstart;
                            const int8_t* last = srcData + xend;
                            int sum_val = std::accumulate(first, last, 0);
                            dstData[x0] = (int8_t)std::round(sum_val*inv_kernel_area);
                        }
                        else
                        {
                            int sum_val = 0;
                            for (int d = dstart; d < dend; ++d) {
                                for (int y = ystart; y < yend; ++y) {
                                    for (int x = xstart; x < xend; ++x) {
                                        const int index = d * inp_width * inp_height + y * inp_width + x;
                                        int8_t val = srcData[index];
                                        sum_val += (int)val;
                                    }
                                }
                            }
                            dstData[x0] = (int8_t)std::round(sum_val*inv_kernel_area);
                        }
                    }
                }
            }
        }
    };

    void maxPooling(Mat &src, Mat &dst)
    {
        const int nstripes = getNumThreads();
        Mat rois;
        PoolingInvoker::run(src, rois, dst, kernel_size, strides, pads_begin, pads_end, avePoolPaddedArea, type, spatialScale, output_zp, nstripes);
    }

    void avePooling(Mat &src, Mat &dst)
    {
        const int nstripes = getNumThreads();
        Mat rois;
        PoolingInvoker::run(src, rois, dst, kernel_size, strides, pads_begin, pads_end, avePoolPaddedArea, type, spatialScale, output_zp, nstripes);
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() != 0);

        bool isPool1D = inputs[0].size() == 3;
        std::vector<int> inpShape(inputs[0].begin() + 2, inputs[0].end());
        std::vector<int> outShape(inputs[0].begin(), inputs[0].begin() + 2);

        std::vector<size_t> local_kernel;
        if (globalPooling) {
            for (int i = 0; i < inpShape.size(); i++) {
                int idx = isGlobalPooling.size() - inpShape.size() + i;
                local_kernel.push_back(isGlobalPooling[idx] ? inpShape[i] : kernel_size[idx]);
            }
        } else {
            local_kernel = kernel_size;
        }

        if (hasDynamicShapes && !shapesInitialized)
        {
            //Just copy input shapes for width and height to prevent errors on loading stage
            for (int i = 0; i < inpShape.size(); i++)
                outShape.push_back(inpShape[i]);
        }
        else if (padMode.empty())
        {
            int addedDims = isPool1D? inpShape.size() : local_kernel.size();
            for (int i = 0; i < addedDims; i++) {
                float dst = (float) (inpShape[i] + pads_begin[i] + pads_end[i] - local_kernel[i]) / strides[i];
                outShape.push_back(1 + (ceilMode ? ceil(dst) : floor(dst)));
            }

            // If we have padding, ensure that the last pooling starts strictly
            // inside the image (instead of at the padding); otherwise clip the last.
            for (int i = 0; i < addedDims; i++) {
                if (pads_end[i] && (outShape[2 + i] - 1) * strides[i] >= inpShape[i] + pads_end[i]) {
                    --outShape[2 + i];
                    CV_Assert((outShape[2 + i] - 1) * strides[i] < inpShape[i] + pads_end[i]);
                }
            }
        }
        else {
            getConvPoolOutParams(inpShape, local_kernel, strides, padMode,
                                 std::vector<size_t>(local_kernel.size(), 1), outShape);
        }

        outputs.assign(1, outShape);
        return false;
    }

    bool updateMemoryShapes(const std::vector<MatShape> &inputs) CV_OVERRIDE
    {
        int dims = inputs[0].size();
        CV_Assert(inputs[0][dims - 1] > 0 && inputs[0][dims - 2] > 0);
        shapesInitialized = true;
        return true;
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_UNUSED(inputs); // suppress unused variable warning
        long flops = 0;
        bool isPool1D = inputs[0].size() == 3;
        size_t karea = std::accumulate(kernel_size.begin(), isPool1D? kernel_size.begin() + 1 : kernel_size.end(),
                                    1, std::multiplies<size_t>());
        for(int i = 0; i < outputs.size(); i++)
        {
            if (type == MAX)
            {
                if (i%2 == 0)
                    flops += total(outputs[i])*karea;
            }
            else
            {
                flops += total(outputs[i])*(karea + 1);
            }
        }
        return flops;
    }
private:
    enum Type
    {
        MAX,
        AVE,
        STOCHASTIC,
        SUM,
        ROI,   // RoI pooling, https://arxiv.org/pdf/1504.08083.pdf
        PSROI  // Position-sensitive RoI pooling, https://arxiv.org/pdf/1605.06409.pdf
    };
    bool hasDynamicShapes;
    bool shapesInitialized;
};

Ptr<PoolingLayerInt8> PoolingLayerInt8::create(const LayerParams& params)
{
    return Ptr<PoolingLayerInt8>(new PoolingLayerInt8Impl(params));
}

}
}
