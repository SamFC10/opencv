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

#include "test_precomp.hpp"
#include "npy_blob.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/dnn/all_layers.hpp>
namespace opencv_test { namespace {

template<typename TString>
static std::string _tf(TString filename)
{
    return (getOpenCVExtraDir() + "dnn/") + filename;
}

class Test_Int8_layers : public DNNTestLayer
{
public:
    void testLayer(const String& basename, const String& importer, double l1, double lInf,
                   int numInps = 1, int numOuts = 1, bool useCaffeModel = false,
                   bool useCommonInputBlob = true, bool hasText = false)
    {
        CV_Assert_N(numInps >= 1, numInps <= 10, numOuts >= 1, numOuts <= 10);
        std::vector<Mat> inps(numInps), inps_int8(numInps);
        std::vector<Mat> refs(numOuts), outs_int8(numOuts), outs_dequantized(numOuts);
        std::vector<float> inputScale, outputScale;
        std::vector<int> inputZp, outputZp;
        String inpPath, outPath;
        Net net, qnet;

        if (importer == "Caffe")
        {
            String prototxt = _tf("layers/" + basename + ".prototxt");
            String caffemodel = _tf("layers/" + basename + ".caffemodel");
            net = readNetFromCaffe(prototxt, useCaffeModel ? caffemodel : String());

            inpPath = _tf("layers/" + (useCommonInputBlob ? "blob" : basename + ".input"));
            outPath =  _tf("layers/" + basename);
        }
        else if (importer == "TensorFlow")
        {
            String netPath = _tf("tensorflow/" + basename + "_net.pb");
            String netConfig = hasText ? _tf("tensorflow/" + basename + "_net.pbtxt") : "";
            net = readNetFromTensorflow(netPath, netConfig);

            inpPath = _tf("tensorflow/" + basename + "_in");
            outPath = _tf("tensorflow/" + basename + "_out");
        }
        else if (importer == "ONNX")
        {
            String onnxmodel = _tf("onnx/models/" + basename + ".onnx");
            net = readNetFromONNX(onnxmodel);

            inpPath = _tf("onnx/data/input_" + basename);
            outPath = _tf("onnx/data/output_" + basename);
        }
        ASSERT_FALSE(net.empty());
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);

        for (int i = 0; i < numInps; i++)
            inps[i] = blobFromNPY(inpPath + ((numInps > 1) ? cv::format("_%d.npy", i) : ".npy"));

        for (int i = 0; i < numOuts; i++)
            refs[i] = blobFromNPY(outPath + ((numOuts > 1) ? cv::format("_%d.npy", i) : ".npy"));

        qnet = net.quantize(inps);
        qnet.getInputDetails(inputScale, inputZp);
        qnet.getOutputDetails(outputScale, outputZp);
        for (int i = 0; i < numInps; i++)
        {
            // Quantize inputs to int8
            // int8_value = float_value/scale + zero-point
            inps[i].convertTo(inps_int8[i], CV_8S, 1.f/inputScale, inputZp);
            qnet.setInput(inps_int8[i]);
        }
        qnet.forward(outs_int8);
        for (int i = 0; i < numOuts; i++)
        {
            // Dequantize outputs and compare with reference outputs
            // float_value = scale*(int8_value - zero-point)
            outs_int8[i].convertTo(outs_dequantized[i], CV_32F, outputScale, -(outputScale * outputZp));
            normAssert(refs[i], outs_dequantized[i], "", l1, lInf);
        }
    }
};

TEST_P(Test_Int8_layers, Convolution1D)
{
    testLayer("conv1d", "ONNX", 0.00385, 0.01003);
    testLayer("conv1d_bias", "ONNX", 0.00325, 0.00948);
}

TEST_P(Test_Int8_layers, Convolution2D)
{
    testLayer("single_conv", "TensorFlow", 0.00413, 0.02201);
    testLayer("convolution", "ONNX", 0.00530, 0.01516);
    testLayer("two_convolution", "ONNX", 0.00333, 0.00978);
}

TEST_P(Test_Int8_layers, Convolution3D)
{
    testLayer("conv3d", "TensorFlow", 0.00777, 0.02274);
    testLayer("conv3d", "ONNX", 0.00360, 0.01193);
    testLayer("conv3d_bias", "ONNX", 0.00252, 0.00390);
}

// TODO : skip this test for all other backends except OpenCV/CPU.
INSTANTIATE_TEST_CASE_P(/**/, Test_Int8_layers, dnnBackendsAndTargets());
}} // namespace
