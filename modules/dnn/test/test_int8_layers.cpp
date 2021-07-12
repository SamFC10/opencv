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
            String inp_name = numInps > 1 ? (importer == "Caffe" ? cv::format("input_%d", i) : cv::format("%d", i)) : "";
            qnet.setInput(inps_int8[i], inp_name);
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
    testLayer("conv3d", "TensorFlow", 0.00734, 0.02434);
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
    testLayer("padding_valid", "TensorFlow", 0.0026, 0.0064);
    testLayer("padding_same", "TensorFlow", 0.0081, 0.032);
    testLayer("spatial_padding", "TensorFlow", 0.0078, 0.028);
    testLayer("mirror_pad", "TensorFlow", 0.0064, 0.013);
    testLayer("pad_and_concat", "TensorFlow", 0.0021, 0.0098);
    testLayer("padding", "ONNX", 0.0005, 0.0069);
    testLayer("ReflectionPad2d", "ONNX", 0.00062, 0.0018);
    testLayer("ZeroPad2d", "ONNX", 0.00037, 0.0018);
}

TEST_P(Test_Int8_layers, AvePooling)
{
    testLayer("layer_pooling_ave", "Caffe", 0.0021, 0.0075);
    testLayer("ave_pool_same", "TensorFlow", 0.00153, 0.0041);
    testLayer("ave_pool3d", "TensorFlow", 0.00175, 0.0047);
    testLayer("average_pooling_1d", "ONNX", 0.002, 0.0048);
    testLayer("average_pooling", "ONNX", 0.0014, 0.0032);
    testLayer("average_pooling_dynamic_axes", "ONNX", 0.0014, 0.006);
    testLayer("ave_pool3d", "ONNX", 0.00063, 0.0016);
}

TEST_P(Test_Int8_layers, MaxPooling)
{
    testLayer("pool_conv_1d", "ONNX", 0.0006, 0.0015);
    testLayer("pool_conv_3d", "ONNX", 0.0033, 0.0124);

    /* All the below tests have MaxPooling as last layer, so computeMaxIdx is set to true
       which is not supported by int8 maxpooling
    testLayer("layer_pooling_max", "Caffe", 0.0021, 0.004);
    testLayer("max_pool_even", "TensorFlow", 0.0048, 0.0139);
    testLayer("max_pool_odd_valid", "TensorFlow", 0.0043, 0.012);
    testLayer("conv_pool_nchw", "TensorFlow", 0.007, 0.025);
    testLayer("max_pool3d", "TensorFlow", 0.0025, 0.0058);
    testLayer("maxpooling_1d", "ONNX", 0.0018, 0.0037);
    testLayer("two_maxpooling_1d", "ONNX", 0.0037, 0.0052);
    testLayer("maxpooling", "ONNX", 0.0034, 0.0065);
    testLayer("two_maxpooling", "ONNX", 0.0025, 0.0052);
    testLayer("max_pool3d", "ONNX", 0.0028, 0.0069);*/
}

TEST_P(Test_Int8_layers, Reduce)
{
    testLayer("reduce_mean", "TensorFlow", 0.0005, 0.0014);
    testLayer("reduce_mean", "ONNX", 0.00062, 0.0014);
    testLayer("reduce_mean_axis1", "ONNX", 0.00032, 0.0007);
    testLayer("reduce_mean_axis2", "ONNX", 0.00033, 0.001);
    testLayer("reduce_mean3d", "ONNX", 0.00048, 0.0016);

    testLayer("reduce_sum", "TensorFlow", 0.015, 0.031);
    testLayer("reduce_sum_channel", "TensorFlow", 0.008, 0.019);
    testLayer("sum_pool_by_axis", "TensorFlow", 0.012, 0.032);
    testLayer("reduce_sum", "ONNX", 0.0025, 0.0048);

    testLayer("reduce_max", "ONNX", 0, 0);
    testLayer("reduce_max_axis_0", "ONNX", 0.0042, 0.007);
    testLayer("reduce_max_axis_1", "ONNX", 0.0018, 0.0036);
}

TEST_P(Test_Int8_layers, ReLU)
{
    testLayer("layer_relu", "Caffe", 0.0005, 0.002);
    testLayer("ReLU", "ONNX", 0.0012, 0.0047);
}

TEST_P(Test_Int8_layers, LeakyReLU)
{
    testLayer("leaky_relu", "TensorFlow", 0.0002, 0.0004);
}

TEST_P(Test_Int8_layers, ReLU6)
{
    testLayer("keras_relu6", "TensorFlow", 0.0018, 0.0062);
    testLayer("keras_relu6", "TensorFlow", 0.0018, 0.0062, 1, 1, false, true, true);
    testLayer("clip_by_value", "TensorFlow", 0.0009, 0.002);
    testLayer("clip", "ONNX", 0.00006, 0.00037);
}

TEST_P(Test_Int8_layers, Sigmoid)
{
    testLayer("maxpooling_sigmoid", "ONNX", 0.0011, 0.0032);
    testLayer("maxpooling_sigmoid_dynamic_axes", "ONNX", 0.0011, 0.0032);
    testLayer("maxpooling_sigmoid_1d", "ONNX", 0.0011, 0.0037);
}

TEST_P(Test_Int8_layers, Mish)
{
    testLayer("mish", "ONNX", 0.0015, 0.0025);
}

TEST_P(Test_Int8_layers, Softmax)
{
    testLayer("layer_softmax", "Caffe", 0.0011, 0.0036);
    testLayer("keras_softmax", "TensorFlow", 0.00093, 0.0027);
    testLayer("slim_softmax", "TensorFlow", 0.0016, 0.0034);
    testLayer("slim_softmax_v2", "TensorFlow", 0.0029, 0.017);
    testLayer("softmax", "ONNX", 0.0016, 0.0028);
    testLayer("log_softmax", "ONNX", 0.014, 0.025);
    testLayer("softmax_unfused", "ONNX", 0.0009, 0.0021);
}

TEST_P(Test_Int8_layers, Concat)
{
    testLayer("layer_concat_shared_input", "Caffe", 0.0076, 0.029, 1, 1, true, false);
    testLayer("concat_axis_1", "TensorFlow", 0.0056, 0.017);
    testLayer("keras_pad_concat", "TensorFlow", 0.0032, 0.0089);
    testLayer("concat_3d", "TensorFlow", 0.005, 0.014);
    testLayer("concatenation", "ONNX", 0.0032, 0.009);
}

TEST_P(Test_Int8_layers, BatchNorm)
{
    testLayer("layer_batch_norm", "Caffe", 0.0061, 0.019, 1, 1, true);
    testLayer("fused_batch_norm", "TensorFlow", 0.0063, 0.02);
    testLayer("batch_norm_text", "TensorFlow", 0.0048, 0.013, 1, 1, false, true, true);
    testLayer("unfused_batch_norm", "TensorFlow", 0.0076, 0.019);
    testLayer("fused_batch_norm_no_gamma", "TensorFlow", 0.0067, 0.015);
    testLayer("unfused_batch_norm_no_gamma", "TensorFlow", 0.0123, 0.044);
    testLayer("switch_identity", "TensorFlow", 0.0035, 0.011);
    testLayer("batch_norm3d", "TensorFlow", 0.0077, 0.02);
    testLayer("batch_norm", "ONNX", 0.0012, 0.0049);
    testLayer("batch_norm_3d", "ONNX", 0.0039, 0.012);
    testLayer("frozenBatchNorm2d", "ONNX", 0.001, 0.0018);
    testLayer("batch_norm_subgraph", "ONNX", 0.0049, 0.0098);
}

TEST_P(Test_Int8_layers, Scale)
{
    testLayer("batch_norm", "TensorFlow", 0.0028, 0.0098);
    testLayer("scale", "ONNX", 0.0025, 0.0071);
    testLayer("expand_hw", "ONNX", 0.0012, 0.0012);
    testLayer("flatten_const", "ONNX", 0.0024, 0.0048);
}

TEST_P(Test_Int8_layers, InnerProduct)
{
    testLayer("layer_inner_product", "Caffe", 0.005, 0.02, 1, 1, true);
    testLayer("matmul", "TensorFlow", 0.0061, 0.019);
    testLayer("nhwc_transpose_reshape_matmul", "TensorFlow", 0.0009, 0.0091);
    testLayer("nhwc_reshape_matmul", "TensorFlow", 0.03, 0.071);
    testLayer("matmul_layout", "TensorFlow", 0.035, 0.06);
    testLayer("tf2_dense", "TensorFlow", 0, 0);
    testLayer("matmul_add", "ONNX", 0.041, 0.082);
    testLayer("linear", "ONNX", 0.0018, 0.0029);
    testLayer("constant", "ONNX", 0.00021, 0.0006);
    testLayer("lin_with_constant", "ONNX", 0.0011, 0.0016);
}

TEST_P(Test_Int8_layers, Reshape)
{
    testLayer("reshape_layer", "TensorFlow", 0.0032, 0.0082);
    testLayer("reshape_nchw", "TensorFlow", 0.0089, 0.029);
    testLayer("reshape_conv", "TensorFlow", 0.035, 0.054);
    testLayer("reshape_reduce", "TensorFlow", 0.0042, 0.0078);
    testLayer("reshape_as_shape", "TensorFlow", 0.0014, 0.0028);
    testLayer("reshape_no_reorder", "TensorFlow", 0.0014, 0.0028);
    testLayer("shift_reshape_no_reorder", "TensorFlow", 0.0063, 0.014);
    testLayer("dynamic_reshape", "ONNX", 0.0047, 0.0079);
    testLayer("dynamic_reshape_opset_11", "ONNX", 0.0048, 0.0081);
    testLayer("flatten_by_prod", "ONNX", 0.0048, 0.0081);
    testLayer("squeeze", "ONNX", 0.0048, 0.0081);
    testLayer("unsqueeze", "ONNX", 0.0033, 0.0053);
    testLayer("squeeze_and_conv_dynamic_axes", "ONNX", 0.0054, 0.0154);
    testLayer("unsqueeze_and_conv_dynamic_axes", "ONNX", 0.0037, 0.0151);
}

TEST_P(Test_Int8_layers, Permute)
{
    testLayer("tf2_permute_nhwc_ncwh", "TensorFlow", 0.0028, 0.006);
    testLayer("transpose", "ONNX", 0.0015, 0.0046);
}

TEST_P(Test_Int8_layers, Identity)
{
    testLayer("expand_batch", "ONNX", 0.0027, 0.0036);
    testLayer("expand_channels", "ONNX", 0.0013, 0.0019);
    testLayer("expand_neg_batch", "ONNX", 0.00071, 0.0019);
}

TEST_P(Test_Int8_layers, Slice)
{
    testLayer("split", "TensorFlow", 0.0033, 0.0056);
    testLayer("slice_4d", "TensorFlow", 0.003, 0.0073);
    testLayer("strided_slice", "TensorFlow", 0.008, 0.0142);
    testLayer("slice", "ONNX", 0.0046, 0.0077);
    testLayer("slice_dynamic_axes", "ONNX", 0.0039, 0.0084);
    testLayer("slice_opset_11_steps_2d", "ONNX", 0.0052, 0.0124);
    testLayer("slice_opset_11_steps_3d", "ONNX", 0.0068, 0.014);
    testLayer("slice_opset_11_steps_4d", "ONNX", 0.0041, 0.008);
    testLayer("slice_opset_11_steps_5d", "ONNX", 0.0085, 0.021);
}

TEST_P(Test_Int8_layers, Dropout)
{
    testLayer("layer_dropout", "Caffe", 0.0021, 0.004);
    testLayer("dropout", "ONNX", 0.0029, 0.004);
}

TEST_P(Test_Int8_layers, Eltwise)
{
    testLayer("layer_eltwise", "Caffe", 0.062, 0.15);
    testLayer("conv_2_inps", "Caffe", 0.0086, 0.0232, 2, 1, true, false);
    testLayer("eltwise_sub", "TensorFlow", 0.015, 0.047);
    testLayer("eltwise_add_vec", "TensorFlow", 0.037, 0.21); // tflite 0.0095, 0.0365
    testLayer("eltwise_mul_vec", "TensorFlow", 0.173, 1.14); // tflite 0.0028, 0.017
    testLayer("channel_broadcast", "TensorFlow", 0.0025, 0.0063);
    testLayer("split_equals", "TensorFlow", 0.02, 0.065);
    testLayer("mul", "ONNX", 0.0039, 0.014);
    testLayer("split_max", "ONNX", 0.004, 0.012);
}

// TODO : skip this test for all other backends except OpenCV/CPU.
INSTANTIATE_TEST_CASE_P(/**/, Test_Int8_layers, dnnBackendsAndTargets());
}} // namespace
