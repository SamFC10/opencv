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

        // Reuse as much data as possible from OpenCV Extra repo to test all variations of quantized layers.
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

        qnet = net.quantize(inps, CV_8S, CV_8S);
        qnet.getInputDetails(inputScale, inputZp);
        qnet.getOutputDetails(outputScale, outputZp);

        // Quantize inputs to int8
        // int8_value = float_value/scale + zero-point
        for (int i = 0; i < numInps; i++)
        {
            inps[i].convertTo(inps_int8[i], CV_8S, 1.f/inputScale[i], inputZp[i]);
            qnet.setInput(inps_int8[i]);
        }
        qnet.forward(outs_int8);

        // Dequantize outputs and compare with reference outputs
        // float_value = scale*(int8_value - zero-point)
        for (int i = 0; i < numOuts; i++)
        {
            outs_int8[i].convertTo(outs_dequantized[i], CV_32F, outputScale[i], -(outputScale[i] * outputZp[i]));
            normAssert(refs[i], outs_dequantized[i], "", l1, lInf);
        }
    }
};

TEST_P(Test_Int8_layers, Convolution1D)
{
    testLayer("conv1d", "ONNX", 0.00302, 0.00909);
    testLayer("conv1d_bias", "ONNX", 0.00306, 0.00948);
}

TEST_P(Test_Int8_layers, Convolution2D)
{
    testLayer("layer_convolution", "Caffe", 0.0174, 0.0758, 1, 1, true);
    testLayer("single_conv", "TensorFlow", 0.00413, 0.02201);
    testLayer("depthwise_conv2d", "TensorFlow", 0.0388, 0.169);
    testLayer("atrous_conv2d_valid", "TensorFlow", 0.0193, 0.0633);
    testLayer("atrous_conv2d_same", "TensorFlow", 0.0185, 0.1322);
    testLayer("keras_atrous_conv2d_same", "TensorFlow", 0.0056, 0.0244);
    testLayer("convolution", "ONNX", 0.0052, 0.01516);
    testLayer("two_convolution", "ONNX", 0.00295, 0.00840);
}

TEST_P(Test_Int8_layers, Convolution3D)
{
    testLayer("conv3d", "TensorFlow", 0.00742, 0.02434);
    testLayer("conv3d", "ONNX", 0.00353, 0.00941);
    testLayer("conv3d_bias", "ONNX", 0.00129, 0.00249);
}

TEST_P(Test_Int8_layers, Flatten)
{
    testLayer("flatten", "TensorFlow", 0.0036, 0.0069, 1, 1, false, true, true);
    testLayer("unfused_flatten", "TensorFlow", 0.0014, 0.0028);
    testLayer("unfused_flatten_unknown_batch", "TensorFlow", 0.0043, 0.0051);
}

TEST_P(Test_Int8_layers, Padding)
{
    testLayer("spatial_padding", "TensorFlow", 0.0078, 0.028);
    testLayer("mirror_pad", "TensorFlow", 0.0064, 0.013);
    testLayer("pad_and_concat", "TensorFlow", 0.0026, 0.0121);
    testLayer("padding", "ONNX", 0.0005, 0.0069);
    testLayer("ReflectionPad2d", "ONNX", 0.00062, 0.0018);
    testLayer("ZeroPad2d", "ONNX", 0.00037, 0.0018);
}

TEST_P(Test_Int8_layers, AvePooling)
{
    testLayer("layer_pooling_ave", "Caffe", 0.397, 1.005);
    testLayer("ave_pool_same", "TensorFlow", 0.00293, 0.008);
    testLayer("ave_pool3d", "TensorFlow", 0.00465, 0.0113);
    testLayer("average_pooling_1d", "ONNX", 0.0023, 0.0061);
    testLayer("average_pooling", "ONNX", 0.0887, 0.149);
    testLayer("average_pooling_dynamic_axes", "ONNX", 0.0046, 0.0096);
    testLayer("ave_pool3d", "ONNX", 0.00413, 0.0094);
}

TEST_P(Test_Int8_layers, MaxPooling)
{
    testLayer("pool_conv_1d", "ONNX", 0.0005, 0.0013);
    testLayer("pool_conv_3d", "ONNX", 0.00426, 0.0118);

    /* All the below tests have MaxPooling as last layer, so computeMaxIdx is set to true
       which is not supported by int8 maxpooling.
    testLayer("max_pool_even", "TensorFlow", 0.0062, 0.0184);
    testLayer("max_pool_odd_valid", "TensorFlow", 0.0064, 0.0133);
    testLayer("conv_pool_nchw", "TensorFlow", 0.009, 0.033);
    testLayer("max_pool3d", "TensorFlow", 0.0047, 0.012);
    testLayer("maxpooling_1d", "ONNX", 0.0042, 0.0062);
    testLayer("two_maxpooling_1d", "ONNX", 0.0047, 0.0097);
    testLayer("maxpooling", "ONNX", 0.0069, 0.0115);
    testLayer("two_maxpooling", "ONNX", 0.0049, 0.0097);
    testLayer("max_pool3d", "ONNX", 0.0069, 0.0112);*/
}

// TODO : skip this test for all other backends except OpenCV/CPU.
INSTANTIATE_TEST_CASE_P(/**/, Test_Int8_layers, dnnBackendsAndTargets());
}} // namespace
