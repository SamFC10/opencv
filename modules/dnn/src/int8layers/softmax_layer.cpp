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

#include <algorithm>
#include <stdlib.h>

namespace cv
{
namespace dnn
{

class SoftMaxLayerInt8Impl CV_FINAL : public SoftmaxLayerInt8
{
public:

    SoftMaxLayerInt8Impl(const LayerParams& params)
    {
        axisRaw = params.get<int>("axis", 1);
        logSoftMax = params.get<bool>("log_softmax", false);
        output_sc = params.get<float>("scales");
        output_zp = params.get<int>("zeropoints");
        setParamsFrom(params);
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        bool inplace = Layer::getMemoryShapes(inputs, requiredOutputs, outputs, internals);
        MatShape shape = inputs[0];
        int cAxis = normalize_axis(axisRaw, shape.size());
        shape[cAxis] = 1;
        internals.assign(1, shape);
        return inplace;
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual bool tryFuse(Ptr<Layer>& top) CV_OVERRIDE
    {
        Ptr<DequantizeLayer> dequantize_layer = top.dynamicCast<DequantizeLayer>();
        return !dequantize_layer.empty();
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs, internals;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        internals_arr.getMatVector(internals);

        const Mat &src = inputs[0];
        Mat &dst = outputs[0];

        int axis = normalize_axis(axisRaw, src.dims);
        size_t outerSize = src.total(0, axis), channels = src.size[axis],
               innerSize = src.total(axis + 1);

        CV_Assert(src.type() == CV_8S && (dst.type() == CV_8S || dst.type() == CV_32F));
        CV_Assert(src.isContinuous() && dst.isContinuous());

        size_t outerStep = src.total(axis);
        size_t cnStep = src.total(axis + 1);
        const int8_t *srcPtr = src.ptr<int8_t>();
        const float *expPtr = blobs[0].ptr<float>();

        if (dst.type() == CV_32F)
        {
            float *dstPtr = dst.ptr<float>();
            for (size_t outerDim = 0; outerDim < outerSize; outerDim++)
            {
                size_t srcOffset = outerDim * outerStep;
                std::vector<float> expSum(innerSize, 0.f);

                // sum exp along axis
                for (size_t cnDim = 0; cnDim < channels; cnDim++)
                {
                    const int offset = srcOffset + cnDim * cnStep;
                    for (size_t i = 0; i < innerSize; i++)
                        expSum[i] += expPtr[srcPtr[offset + i] + 128];
                }

                // divide by computed sum
                for (size_t cnDim = 0; cnDim < channels; cnDim++)
                {
                    const int offset = srcOffset + cnDim * cnStep;
                    for (size_t i = 0; i < innerSize; i++)
                        dstPtr[offset + i] = expPtr[srcPtr[offset + i] + 128]/expSum[i];
                }

                if (logSoftMax)
                {
                    for (size_t cnDim = 0; cnDim < channels; cnDim++)
                    {
                        const int offset = srcOffset + cnDim * cnStep;
                        for (size_t i = 0; i < innerSize; i++)
                            dstPtr[offset + i] = log(dstPtr[offset + i]);
                    }
                }
            }
        }
        else
        {
            const float inv_scale = 1.f/output_sc;
            int8_t *dstPtr = dst.ptr<int8_t>();
            for (size_t outerDim = 0; outerDim < outerSize; outerDim++)
            {
                size_t srcOffset = outerDim * outerStep;
                std::vector<float> expSum(innerSize, 0.f);

                // sum exp along axis
                for (size_t cnDim = 0; cnDim < channels; cnDim++)
                {
                    const int offset = srcOffset + cnDim * cnStep;
                    for (size_t i = 0; i < innerSize; i++)
                        expSum[i] += expPtr[srcPtr[offset + i] + 128];
                }

                // divide by computed sum and quantize to int8
                if (logSoftMax)
                {
                    for (size_t cnDim = 0; cnDim < channels; cnDim++)
                    {
                        const int offset = srcOffset + cnDim * cnStep;
                        for (size_t i = 0; i < innerSize; i++)
                            dstPtr[offset + i] = saturate_cast<int8_t>(output_zp + std::round(inv_scale*log(expPtr[srcPtr[offset + i] + 128]/expSum[i])));
                    }
                }
                else
                {
                    for (size_t cnDim = 0; cnDim < channels; cnDim++)
                    {
                        const int offset = srcOffset + cnDim * cnStep;
                        for (size_t i = 0; i < innerSize; i++)
                            dstPtr[offset + i] = saturate_cast<int8_t>(output_zp + std::round(inv_scale*(expPtr[srcPtr[offset + i] + 128]/expSum[i])));
                    }
                }
            }
        }
    }

    int64 getFLOPS(const std::vector<MatShape> &inputs,
                  const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_UNUSED(outputs); // suppress unused variable warning
        int64 flops = 0;

        for (int i = 0; i < inputs.size(); i++)
        {
            flops += 4*total(inputs[i]);
        }

        return flops;
    }

    int axisRaw;
};

Ptr<SoftmaxLayerInt8> SoftmaxLayerInt8::create(const LayerParams& params)
{
    return Ptr<SoftmaxLayerInt8>(new SoftMaxLayerInt8Impl(params));
}

}
}
