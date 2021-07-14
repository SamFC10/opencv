// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"

namespace cv
{
namespace dnn
{

class QuantizeLayerImpl CV_FINAL : public QuantizeLayer
{
public:
    QuantizeLayerImpl(const LayerParams& params)
    {
        scale = params.get<float>("scales", 1.0f);
        zeropoint = params.get<int>("zeropoints", 0);
        setParamsFrom(params);
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
        CV_Assert(inputs.size() == 1);
        Layer::getMemoryShapes(inputs, requiredOutputs, outputs, internals);
        return false;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs, internals;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        internals_arr.getMatVector(internals);

        inputs[0].convertTo(outputs[0], CV_8S, 1.f/scale, zeropoint);
    }
};

class DequantizeLayerImpl CV_FINAL : public DequantizeLayer
{
public:
    DequantizeLayerImpl(const LayerParams& params)
    {
        scale = params.get<float>("scales", 1.0f);
        zeropoint = params.get<int>("zeropoints", 0);
        setParamsFrom(params);
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
        CV_Assert(inputs.size() == 1);
        Layer::getMemoryShapes(inputs, requiredOutputs, outputs, internals);
        return false;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs, internals;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        internals_arr.getMatVector(internals);

        inputs[0].convertTo(outputs[0], CV_32F, scale, -(scale*zeropoint));
    }
};

Ptr<QuantizeLayer> QuantizeLayer::create(const LayerParams& params)
{
    return Ptr<QuantizeLayer>(new QuantizeLayerImpl(params));
}

Ptr<DequantizeLayer> DequantizeLayer::create(const LayerParams& params)
{
    return Ptr<DequantizeLayer>(new DequantizeLayerImpl(params));
}

}
}
