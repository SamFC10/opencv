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

#include <opencv2/core/utils/logger.hpp>

#include "opencv2/core/hal/hal.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include <iostream>
#include <numeric>

namespace cv
{
namespace dnn
{

class BaseConvolutionLayerInt8Impl : public ConvolutionLayerInt8
{
public:
    bool fusedWeights, fusedBias;
    BaseConvolutionLayerInt8Impl(const LayerParams &params)
    {
        setParamsFrom(params);
        getConvolutionKernelParams(params, kernel_size, pads_begin, pads_end, strides, dilations, padMode, adjust_pads);

        numOutput = params.get<int>("num_output");
        int ngroups = params.get<int>("group", 1);
        CV_Assert(numOutput % ngroups == 0);

        input_sc = params.get<float>("input_scale");
        input_zp = params.get<int>("input_zeropoint");
        output_sc = params.get<float>("output_scale");
        output_zp = params.get<int>("output_zeropoint");

        if (kernel_size.size() == 2) {
            kernel = Size(kernel_size[1], kernel_size[0]);
            stride = Size(strides[1], strides[0]);
            for (int i = 0; i < pads_begin.size(); i++) {
                if (pads_begin[i] != pads_end[i])
                    CV_Error(Error::StsNotImplemented, "Unsupported asymmetric padding in convolution layer");
            }
            pad = Size(pads_begin[1], pads_begin[0]);
            dilation = Size(dilations[1], dilations[0]);

            adjustPad.height = adjust_pads[0];
            adjustPad.width = adjust_pads[1];
        }

        for (int i = 0; i < adjust_pads.size(); i++) {
            CV_Assert(adjust_pads[i] < strides[i]);
        }

        fusedWeights = false;
        fusedBias = false;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        // blobs[0] - Additional quantization params for convolution
        // blobs[1] - Weights
        // blobs[2] - Biases
        CV_Assert(!inputs.empty() && blobs.size() == 3);
        MatSize weightShape = blobs[1].size;

        CV_Assert(inputs[0].dims == outputs[0].dims);
        if (weightShape.dims() == 3)
        {
            kernel_size.assign(1, kernel_size[0]);
            strides.assign(1, strides[0]);
        }
        CV_Assert(weightShape.dims() == kernel_size.size() + 2);
        for (int i = 0; i < kernel_size.size(); i++) {
            CV_Assert(weightShape[i + 2] == kernel_size[i]);
        }

        const Mat &input = inputs[0];
        CV_Assert(((input.dims == 3 && kernel_size.size() == 1) || input.dims == 4 || input.dims == 5) && (input.type() == CV_8S || input.type() == CV_32F));
        for (size_t i = 0; i < outputs.size(); i++)
        {
            CV_Assert(inputs[i].type() == input.type());
            CV_Assert(((input.dims == 3 && kernel_size.size() == 1) || inputs[i].dims == 4 || inputs[i].dims == 5) && inputs[i].size[1] == input.size[1]);
            for (int j = 0; j < inputs[i].dims; j++) {
                CV_Assert(inputs[i].size[j] == input.size[j]);
            }
        }

        std::vector<int> inpShape;
        std::vector<int> outShape;
        for (int i = 2; i < inputs[0].dims; i++) {
            inpShape.push_back(inputs[0].size[i]);
            outShape.push_back(outputs[0].size[i]);
        }
        getConvPoolPaddings(inpShape, kernel_size, strides, padMode, pads_begin, pads_end);
        if (pads_begin.size() == 2) {
            for (int i = 0; i < pads_begin.size(); i++) {
                if (pads_begin[i] != pads_end[i])
                    CV_Error(Error::StsNotImplemented, "Unsupported asymmetric padding in convolution layer");
            }
            pad = Size(pads_begin[1], pads_begin[0]);
        }
        fusedWeights = false;
        fusedBias = false;
    }

    virtual MatShape computeColRowShape(const MatShape &inpShape, const MatShape &outShape) const = 0;
    bool is1x1() const
    {
        return (kernel.height == 1 && kernel.width == 1) &&
               (stride.height == 1 && stride.width == 1) &&
               (dilation.height == 1 && dilation.width == 1);
    }

    virtual bool tryFuse(Ptr<Layer>& top) CV_OVERRIDE
    {
        Ptr<BatchNormLayer> batchnorm_layer = top.dynamicCast<BatchNormLayer>();
        if (batchnorm_layer)
            return true;

        return false;
    }
};

//TODO: simultaneously convolution and bias addition for cache optimization
class ConvolutionLayerInt8Impl CV_FINAL : public BaseConvolutionLayerInt8Impl
{
public:
    enum { VEC_ALIGN = 32, DFT_TYPE = CV_8S };
    Mat weightsMat;
    std::vector<int> biasvec;
    // TODO : Deal with fused activation with convolution. For now activations which are just clamps,
    //        like ReLU (without negative slope) and ReLU6 are fused in the Output stage. All the remaining activations
    //        will need seperate fixed point arithmetic which hasn't been done yet.
    std::vector<float> reluslope;
    Ptr<ActivationLayer> activ;

    ConvolutionLayerInt8Impl(const LayerParams &params) : BaseConvolutionLayerInt8Impl(params){}

    MatShape computeColRowShape(const MatShape &inpShape, const MatShape &outShape) const CV_OVERRIDE
    {
        CV_Assert(!blobs.empty());
        int dims = inpShape.size();
        int inpD = dims == 5 ? inpShape[2] : 1;
        int inpH = inpShape[dims - 2];
        int inpW = inpShape.back();
        int inpGroupCn = blobs[1].size[1];
        int ksize = inpGroupCn * std::accumulate(kernel_size.begin(), kernel_size.end(),
                                                 1, std::multiplies<size_t>());
        return shape(inpD * inpH * inpW, ksize);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        size_t ksize = kernel_size.size();
        // Only default backend and Conv1D/Conv2D/Conv3D are supported
        return backendId == DNN_BACKEND_OPENCV && ksize >= 1 && ksize <= 3;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(!blobs.empty());
        const int* weightShape = blobs[1].size.p;
        CV_Assert(blobs[2].total() == (size_t)weightShape[0]);

        internals.clear();

        CV_Assert(inputs.size() != 0);
        std::vector<int> inpShape(inputs[0].begin() + 2, inputs[0].end());

        int outCn = weightShape[0];
        std::vector<int> outShape;
        outShape.push_back(inputs[0][0]);
        outShape.push_back(outCn);

        int inpCn = inputs[0][1];
        if (padMode.empty())
        {
            for (int i = 0; i < inpShape.size(); i++)
                outShape.push_back((inpShape[i] + pads_begin[i] + pads_end[i] - dilations[i] * (kernel_size[i] - 1) - 1) / strides[i] + 1);
        }
        else
        {
            getConvPoolOutParams(inpShape, kernel_size, strides, padMode, dilations, outShape);
        }

        int ngroups = inpCn / weightShape[1];
        if (ngroups == 0 || ngroups * weightShape[1] != inpCn)
            CV_Error(Error::StsError, format("Number of input channels should "
                     "be multiple of %d but got %d", weightShape[1], inpCn));
        CV_Assert(ngroups > 0 && inpCn % ngroups == 0 && outCn % ngroups == 0);

        outputs.resize(1, outShape);

        return false;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        BaseConvolutionLayerInt8Impl::finalize(inputs_arr, outputs_arr);

        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);
        // prepare weightsMat where each row is aligned and has enough zero padding on the right to
        // use vectorized (i.e. with intrinsics) loops without tail processing
        Mat wm = blobs[1].reshape(1, numOutput);
        if( wm.step1() % VEC_ALIGN != 0 )
        {
            int newcols = (int)alignSize(wm.step1(), VEC_ALIGN);
            Mat wm_buffer = Mat(numOutput, newcols, wm.type());
            Mat wm_padding = wm_buffer.colRange(wm.cols, newcols);
            wm_padding.setTo(Scalar::all(0));
            Mat wm_aligned = wm_buffer.colRange(0, wm.cols);
            wm.copyTo(wm_aligned);
            wm = wm_aligned;
        }
        weightsMat = wm;

        Mat biasMat = blobs[2];
        biasvec.resize(numOutput+2);
        for(int i = 0; i < numOutput; i++ )
            biasvec[i] = biasMat.at<int>(i);
    }

    bool setActivation(const Ptr<ActivationLayer>& layer) CV_OVERRIDE
    {
        if ((!activ.empty() && !layer.empty()) || blobs.empty())
            return false;

        activ = layer;
        if (activ.empty())
            reluslope.clear();

        return !activ.empty();
    }

    virtual bool tryFuse(Ptr<Layer>& top) CV_OVERRIDE
    {
        return BaseConvolutionLayerInt8Impl::tryFuse(top);
    }

    class ParallelConv : public cv::ParallelLoopBody
    {
    public:
        enum { BLK_SIZE = 32, BLK_SIZE_CN = 64 };

        const Mat* input_;
        const Mat* weights_;
        Mat* output_;
        int outShape[4]; // used only for conv2d
        std::vector<size_t> kernel_size, pads_begin, pads_end, strides, dilations;
        int ngroups_, nstripes_;
        std::vector<int> ofstab_;
        const std::vector<int>* biasvec_;
        const std::vector<float>* reluslope_;
        const ActivationLayer* activ_;
        bool is1x1_;
        bool useAVX;
        bool useAVX2;
        bool useAVX512;
        int blk_size_cn;
        int inpZp, outZp;
        const int* multiplier;

        ParallelConv()
            : input_(0), weights_(0), output_(0), ngroups_(0), nstripes_(0),
              biasvec_(0), reluslope_(0), activ_(0), is1x1_(false), useAVX(false), useAVX2(false), useAVX512(false)
            , blk_size_cn(0)
        {}

        static void run( const Mat& input, Mat& output, const Mat& weights, const Mat& additionalParams,
                         const std::vector<int>& biasvec,
                         const std::vector<float>& reluslope,
                         const std::vector<size_t>& kernel_size, const std::vector<size_t>& strides,
                         const std::vector<size_t>& pads_begin, const std::vector<size_t>& pads_end,
                         const std::vector<size_t>& dilations,
                         const ActivationLayer* activ, int ngroups, int nstripes, int inp_Zp, int out_Zp)
        {
            size_t karea = std::accumulate(kernel_size.begin(), kernel_size.end(),
                                           1, std::multiplies<size_t>());
            bool isConv1D = input.dims == 3;
            bool isConv2D = input.dims == 4;
            bool isConv3D = input.dims == 5;
            CV_CheckEQ(static_cast<int>(kernel_size.size()), input.dims - 2, "");
            CV_Assert_N(input.dims == output.dims,
                       input.size[0] == output.size[0],
                       weights.rows == output.size[1],
                       weights.cols == (input.size[1]/ngroups)*karea,
                       input.type() == CV_8SC1,
                       output.type() == CV_32SC1,
                       input.type() == weights.type(),
                       input.isContinuous(),
                       output.isContinuous(),
                       biasvec.size() == (size_t)output.size[1]+2);
            CV_Check(weights.step1(), weights.step1() % VEC_ALIGN == 0, "");
            ParallelConv p;

            p.input_ = &input;
            p.weights_ = &weights;
            p.output_ = &output;
            int max_ind = isConv1D? 3: 4;
            for( int i = 0; i < max_ind; i++ ) p.outShape[i] = output.size[i];
            p.outShape[1] /= ngroups;

            p.kernel_size = kernel_size; p.strides = strides; p.dilations = dilations;
            p.pads_begin = pads_begin; p.pads_end = pads_end;

            p.ngroups_ = ngroups;
            p.nstripes_ = nstripes;

            int inpCnAll = input.size[1];
            int depth = (input.dims == 5) ? input.size[2] : 1;
            int width = input.size[input.dims - 1];
            int height = isConv1D? 1 : input.size[input.dims - 2];
            int inpCn = inpCnAll / ngroups;

            p.is1x1_ = (isConv2D && kernel_size[0] == 1 && kernel_size[1] == 1 &&
                       pads_begin[0] == 0  && pads_begin[1] == 0) ||
                       (isConv1D && pads_begin[0] == 0 && kernel_size[0] == 1);

            p.useAVX    = checkHardwareSupport(CPU_AVX)  && isConv2D;
            p.useAVX2   = checkHardwareSupport(CPU_AVX2) && isConv2D;
            p.useAVX512 = CV_CPU_HAS_SUPPORT_AVX512_SKX  && isConv2D;

            int kernel_d = isConv3D? kernel_size[0] : 1;
            int kernel_h = isConv1D? 1 : kernel_size[kernel_size.size() - 2];
            int kernel_w = kernel_size.back();

            int blk_size_cn0 = cvCeil(1600./(kernel_w*kernel_h));
            int ncn = 32;
            while (ncn*2 < blk_size_cn0 && ncn < inpCn)
                ncn *= 2;
            ncn = std::min(ncn, inpCn);
            p.blk_size_cn = ncn;

            int dil_d = isConv3D? dilations[0] : 1;
            int dil_h = isConv1D? 1 : dilations[dilations.size() - 2];
            int dil_w = dilations.back();

            p.inpZp = inp_Zp;
            p.outZp = out_Zp;
            p.multiplier = additionalParams.ptr<int>(0);

            p.ofstab_.resize(karea * ncn);
            int* ofstab = &p.ofstab_[0];

            if (isConv1D)
            {
                for( int k = 0; k < ncn; k++ )
                    for( int k_c = 0; k_c < kernel_w; k_c++ )
                        ofstab[k*kernel_w + k_c] = k*width + k_c*dil_w;
            }
            else if (isConv2D)
            {
                for( int k = 0; k < ncn; k++ )
                    for( int k_r = 0; k_r < kernel_h; k_r++ )
                        for( int k_c = 0; k_c < kernel_w; k_c++ )
                            ofstab[(k*kernel_h + k_r)*kernel_w + k_c] =
                                   (k*height + k_r*dil_h)*width + k_c*dil_w;
            }
            else
            {
                for( int k = 0; k < ncn; k++ )
                    for (int k_d = 0; k_d < kernel_d; k_d++)
                        for( int k_r = 0; k_r < kernel_h; k_r++ )
                            for( int k_c = 0; k_c < kernel_w; k_c++ )
                                ofstab[(k*kernel_d*kernel_h + k_d*kernel_h + k_r)*kernel_w + k_c] =
                                       (k*depth*height + k_d*dil_d*height + k_r*dil_h)*width + k_c*dil_w;
            }

            p.biasvec_ = &biasvec;
            p.reluslope_ = &reluslope;
            p.activ_ = p.reluslope_->empty() ? activ : 0;

            parallel_for_(Range(0, nstripes), p, nstripes);
        }

        virtual void operator ()(const Range &r0) const CV_OVERRIDE
        {
            const int valign = ConvolutionLayerInt8Impl::VEC_ALIGN;
            int ngroups = ngroups_, batchSize = input_->size[0]*ngroups;
            bool isConv1D = input_->dims == 3;
            bool isConv2D = input_->dims == 4;
            bool isConv3D = input_->dims == 5;

            int outW = output_->size[output_->dims - 1];
            int outH = isConv1D? 1 : output_->size[output_->dims - 2];
            int outCn = output_->size[1]/ngroups;

            int depth = isConv3D? input_->size[2] : 1;
            int height = isConv1D? 1 : input_->size[input_->dims - 2];
            int width = input_->size[input_->dims - 1];
            int inpCn = input_->size[1]/ngroups;

            const int nstripes = nstripes_;

            int kernel_d = isConv3D? kernel_size[0] : 1;
            int kernel_h = isConv1D? 1 : kernel_size[kernel_size.size() - 2];
            int kernel_w = kernel_size.back();
            int karea = kernel_w*kernel_h*kernel_d;

            int pad_d = isConv3D? pads_begin[0] : 0;
            int pad_t = isConv1D? 0 : pads_begin[pads_begin.size() - 2];
            int pad_l = pads_begin.back();

            int stride_d = isConv3D? strides[0] : 0;
            int stride_h = isConv1D? 0 : strides[strides.size() - 2];
            int stride_w = strides.back();

            int dilation_d = isConv3D? dilations[0] : 1;
            int dilation_h = isConv1D? 1 : dilations[dilations.size() - 2];
            int dilation_w = dilations.back();

            int i, j, k, d;
            int inpPlaneSize = (int)input_->total(2);
            int outPlaneSize = (int)output_->total(2);
            bool is1x1 = is1x1_;

            int stripesPerSample;
            int stripeSize;
            Range r = r0;
            bool depthWiseConvolution = !is1x1 && isConv2D && ngroups > 1 && inpCn == 1 &&
                outCn == 1 && kernel_d == 1 && dilation_d == 1 && stride_d == 0 && pad_d == 0 &&
                width >= 16 + dilation_w*(kernel_w - 1);
            // for now only 3x3 depth-wise convolutions are supported
            depthWiseConvolution = depthWiseConvolution && kernel_w == 3 && kernel_h == 3 &&
                // computing at most 1 pixel from each side can involve padding
                max(stride_w, dilation_w) >= pad_l && max(stride_h, dilation_h) >= pad_t &&
                pad_l <= 1 && pad_t <= 1;

            if( !depthWiseConvolution && nstripes >= batchSize*2 )
            {
                stripesPerSample = nstripes/batchSize;
                stripeSize = (int)alignSize((outPlaneSize + stripesPerSample - 1)/stripesPerSample, 8);
                stripeSize = std::min(stripeSize, outPlaneSize);
            }
            else
            {
                stripesPerSample = 1;
                int samplesPerStripe = std::max((batchSize + nstripes - 1)/nstripes, 1);
                r.start *= samplesPerStripe;
                r.end *= samplesPerStripe;
                stripeSize = outPlaneSize;
            }

            const int8_t* data_inp0_ = input_->ptr<int8_t>();
            const int* ofstab = &ofstab_[0];
            const int8_t* wptr_orig_ = weights_->ptr<int8_t>();
            size_t wstep = weights_->step1();
            const int* biasptr_ = &biasvec_->at(0);
            const float* reluptr_ = reluslope_->empty() ? 0 : &reluslope_->at(0);
            int* data_out0_ = output_->ptr<int>();
            AutoBuffer<int8_t> rowbuf0_;
            int8_t* rowbuf0 = 0;
            bool use_rowbuf = !depthWiseConvolution;
            int blk_size = depthWiseConvolution ? outPlaneSize : min((int)BLK_SIZE, stripeSize);

            // im2row buffer is not used for depth-wise convolution
            if(use_rowbuf)
            {
                size_t rowbufsz = alignSize(karea*blk_size_cn, valign)*min((int)BLK_SIZE, blk_size);
                //printf("karea=%d, blk_size_cn=%d, rowbufsz=%d, stripeSize=%d\n", karea, blk_size_cn, (int)rowbufsz, stripeSize);
                rowbuf0_.allocate(rowbufsz + valign);
                rowbuf0 = alignPtr(rowbuf0_.data(), (int)(valign*sizeof(int8_t)));
                // we clear the buffer once; ultimately, it lets us to avoid
                // tail processing after running the unrolled/vectorized loop.
                // the main idea is to make sure that the tail (a.k.a. padding) of each row
                // (i.e. the elements with indices between vsz=karea*ncn and vsz_a)
                // does not contain NaNs or Infs. Because the padding in the weights
                // matrix is explicitly initialized with 0's, we handle all other
                // cases nicely, i.e. we can skip expliciting re-initialization
                // of the padding - we just retain elements from the previous iteration
                // of the loop over channels (cn0).
                memset(rowbuf0, (int8_t)inpZp, rowbufsz*sizeof(rowbuf0[0]) );
            }

            for( int stripe = r.start; stripe < r.end; stripe++ )
            {
                int subsampleIdx = stripe/stripesPerSample;
                if( subsampleIdx >= batchSize )
                    break;
                int stripeStart = (int)((stripe - subsampleIdx*stripesPerSample)*stripeSize);
                int stripeEnd = (int)std::min(stripeStart + stripeSize, outPlaneSize);
                const int8_t* data_inp0 = data_inp0_ + subsampleIdx*inpPlaneSize*inpCn;
                int* data_out0 = data_out0_ + subsampleIdx*outPlaneSize*outCn;
                int startOutCn = (subsampleIdx % ngroups)*outCn;
                const int8_t* wptr_orig = wptr_orig_ + wstep*startOutCn;
                const int* biasptr = biasptr_ + startOutCn;

                for( int cn0 = 0; cn0 < inpCn; cn0 += blk_size_cn )
                {
                    int cn1 = std::min(cn0 + blk_size_cn, inpCn);
                    int ncn = cn1 - cn0, vsz = karea*ncn;
                    int vsz_a = (int)alignSize(vsz, valign);
                    const int8_t* wptr = wptr_orig + cn0*karea;
                    // we apply [Channels][P]ReLU (if any) during the final pass only.
                    const float* relu = cn1 == inpCn && reluptr_ ? reluptr_ + startOutCn : 0;

                    for( int ofs0 = stripeStart; ofs0 < stripeEnd; ofs0 += blk_size )
                    {
                        int ofs, ofs1 = std::min(ofs0 + blk_size, stripeEnd);
                        int bsz = ofs1 - ofs0;

                        int out_d = ofs0 / (outH * outW);
                        int out_i = (ofs0 - out_d * outH * outW) / outW;
                        int out_j = ofs0 % outW;

                        // TODO: Depth-Wise Convolution
                        /*if (depthWiseConvolution)
                        {
                            CV_Assert(out_i == 0 && out_j == 0);
                            int in_d = out_d * stride_d - pad_d;
                            const float* inptr_ = data_inp0 + (cn0*depth*height + in_d*height)*width;
                            float* outptr_ = data_out0 + ofs0;

                        #if CV_TRY_AVX2
                            if(useAVX2)
                                opt_AVX2::fastDepthwiseConv(wptr, kernel_h, kernel_w,
                                    stride_h, stride_w, dilation_h, dilation_w, pad_t, pad_l,
                                    biasptr, relu, inptr_, height, width, outptr_, out_d, outH, outW);
                            else
                        #endif
                        #if CV_TRY_AVX
                            if(useAVX)
                                opt_AVX::fastDepthwiseConv(wptr, kernel_h, kernel_w,
                                    stride_h, stride_w, dilation_h, dilation_w, pad_t, pad_l,
                                    biasptr, relu, inptr_, height, width, outptr_, out_d, outH, outW);
                            else
                        #endif
                            {
                                const float w00_ = wptr[0], w01_ = wptr[1], w02_ = wptr[2],
                                            w10 = wptr[3], w11 = wptr[4], w12 = wptr[5],
                                            w20_ = wptr[6], w21_ = wptr[7], w22_ = wptr[8];
                                int outW1 = min(outW, (width - dilation_w*(kernel_w - 1) + pad_l)/stride_w);
                                float relu_coeff = relu ? relu[out_d] : 1.f, bias = biasptr[out_d];

                                for (int out_i = 0; out_i < outH; out_i++)
                                {
                                    int in_i = out_i * stride_h - pad_t, out_j = 0;
                                    const float* imgptr0 = inptr_ + in_i*width;
                                    const float* imgptr1 = imgptr0 + dilation_h*width;
                                    const float* imgptr2 = imgptr0 + (dilation_h*2)*width;
                                    float out, w00 = w00_, w01 = w01_, w02 = w02_;
                                    float w20 = w20_, w21 = w21_, w22 = w22_;
                                    if (in_i < 0)
                                    {
                                        w00 = w01 = w02 = 0.f;
                                        imgptr0 = imgptr1;
                                    }
                                    else if (in_i + dilation_h*(kernel_h-1) >= height)
                                    {
                                        w20 = w21 = w22 = 0.f;
                                        imgptr2 = imgptr1;
                                    }
                                    float* outptr = outptr_ + out_i*outW;
                                    if (pad_l > 0)
                                    {
                                        out = imgptr0[0]*w01 + imgptr0[dilation_w]*w02 +
                                              imgptr1[0]*w11 + imgptr1[dilation_w]*w12 +
                                              imgptr2[0]*w21 + imgptr2[dilation_w]*w22 + bias;
                                        if (relu)
                                            out = out > 0.f ? out : out*relu_coeff;
                                        outptr[0] = out;
                                        out_j = 1;
                                    }

                                #if CV_SIMD
                                    // maybe with AVX or AVX512 strided depthwise convolution
                                    // can be accelerated with vector code, but with 4xfloat vectors
                                    // it's hardly the case
                                    if( stride_w == 1 )
                                    {
                                        const int VECSZ = v_float32::nlanes;
                                        const int out_delta = VECSZ/stride_w;
                                        v_float32 vw00 = vx_setall_f32(w00), vw01 = vx_setall_f32(w01), vw02 = vx_setall_f32(w02),
                                                  vw10 = vx_setall_f32(w10), vw11 = vx_setall_f32(w11), vw12 = vx_setall_f32(w12),
                                                  vw20 = vx_setall_f32(w20), vw21 = vx_setall_f32(w21), vw22 = vx_setall_f32(w22);
                                        v_float32 z = vx_setzero_f32(), vbias = vx_setall_f32(bias), vrc = vx_setall_f32(relu_coeff);
                                        for( ; out_j < outW1; out_j += out_delta )
                                        {
                                            if (out_j + out_delta > outW1)
                                            {
                                                if (out_j <= pad_l)
                                                    break;
                                                out_j = outW1 - out_delta;
                                            }
                                            int in_j = out_j * stride_w - pad_l;
                                            v_float32 v00 = vx_load(imgptr0 + in_j),
                                                      v01 = vx_load(imgptr0 + in_j + dilation_w),
                                                      v02 = vx_load(imgptr0 + in_j + dilation_w*2),
                                                      v10 = vx_load(imgptr1 + in_j),
                                                      v11 = vx_load(imgptr1 + in_j + dilation_w),
                                                      v12 = vx_load(imgptr1 + in_j + dilation_w*2),
                                                      v20 = vx_load(imgptr2 + in_j),
                                                      v21 = vx_load(imgptr2 + in_j + dilation_w),
                                                      v22 = vx_load(imgptr2 + in_j + dilation_w*2);

                                            v_float32 vout = v00*vw00 + v01*vw01 + v02*vw02 +
                                                             v10*vw10 + v11*vw11 + v12*vw12 +
                                                             v20*vw20 + v21*vw21 + v22*vw22 + vbias;
                                            if (relu)
                                                vout = v_select(vout > z, vout, vout*vrc);
                                            vx_store(outptr + out_j, vout);
                                        }
                                    }
                                #endif
                                    for (; out_j < outW1; out_j++)
                                    {
                                        int in_j = out_j * stride_w - pad_l;
                                        out = imgptr0[in_j]*w00 + imgptr0[in_j + dilation_w]*w01 + imgptr0[in_j + dilation_w*2]*w02 +
                                              imgptr1[in_j]*w10 + imgptr1[in_j + dilation_w]*w11 + imgptr1[in_j + dilation_w*2]*w12 +
                                              imgptr2[in_j]*w20 + imgptr2[in_j + dilation_w]*w21 + imgptr2[in_j + dilation_w*2]*w22 + bias;
                                        if (relu)
                                            out = out > 0.f ? out : out*relu_coeff;
                                        outptr[out_j] = out;
                                    }

                                    for (; out_j < outW; out_j++ )
                                    {
                                        int in_j0 = out_j * stride_w - pad_l, in_j1 = in_j0 + dilation_w, in_j2 = in_j0 + dilation_w*2;
                                        float s0 = 1.f, s1 = 1.f, s2 = 1.f;
                                        if (in_j0 >= width)
                                        {
                                            in_j0 = 0;
                                            s0 = 0.f;
                                        }
                                        if (in_j1 >= width)
                                        {
                                            in_j1 = 0;
                                            s1 = 0.f;
                                        }
                                        if (in_j2 >= width)
                                        {
                                            in_j2 = 0;
                                            s2 = 0.f;
                                        }
                                        out = imgptr0[in_j0]*w00*s0 + imgptr0[in_j1]*w01*s1 + imgptr0[in_j2]*w02*s2 +
                                              imgptr1[in_j0]*w10*s0 + imgptr1[in_j1]*w11*s1 + imgptr1[in_j2]*w12*s2 +
                                              imgptr2[in_j0]*w20*s0 + imgptr2[in_j1]*w21*s1 + imgptr2[in_j2]*w22*s2 + bias;
                                        if (relu)
                                            out = out > 0.f ? out : out*relu_coeff;
                                        outptr[out_j] = out;
                                    }
                                }
                            }
                            continue;
                        }*/
                        // do im2row for a part of input tensor
                        int8_t* rowbuf = rowbuf0;

                        if (isConv1D)
                        {
                            for( ofs = ofs0; ofs < ofs1; out_j = 0, ++out_i )
                            {
                                int delta = std::min(ofs1 - ofs, outW - out_j);
                                int out_j1 = out_j + delta;

                                int in_j = out_j * stride_w - pad_l;
                                const int8_t* imgptr = data_inp0 + cn0*width + in_j;
                                ofs += delta;

                                // do im2row for a part of input tensor
                                if( is1x1 )
                                {
                                    for( ; out_j < out_j1; out_j++, rowbuf += vsz_a, imgptr += stride_w )
                                    {
                                        for( k = 0; k < vsz; k++ )
                                            rowbuf[k] = imgptr[k*inpPlaneSize];
                                    }
                                }
                                else
                                {
                                    for( ; out_j < out_j1; out_j++, rowbuf += vsz_a, imgptr += stride_w, in_j += stride_w )
                                    {
                                        // this condition should be true for most of the tensor elements, i.e.
                                        // most of the time the kernel aperture is inside the tensor X-Y plane.
                                        if( out_j + 2 <= out_j1 && 0 <= in_j && in_j + stride_w*2 <= width - (kernel_w-1)*dilation_w )
                                        {
                                            for( k = 0; k < vsz; k++ )
                                            {
                                                int k1 = ofstab[k];
                                                int8_t v0 = imgptr[k1];
                                                int8_t v1 = imgptr[k1 + stride_w];
                                                rowbuf[k] = v0;
                                                rowbuf[k+vsz_a] = v1;
                                            }
                                            out_j++;
                                            rowbuf += vsz_a;
                                            imgptr += stride_w;
                                            in_j += stride_w;
                                        }
                                        else
                                        {
                                            int i0 = std::max(0, (-in_j + dilation_w-1)/dilation_w);
                                            int i1 = std::min(kernel_w, (width - in_j + dilation_w-1)/dilation_w);

                                            // here some non-continuous sub-row of the row will not be
                                            // filled from the tensor; we need to make sure that the uncovered
                                            // elements are explicitly set to 0's. the easiest way is to
                                            // set all the elements to 0's before the loop.
                                            memset(rowbuf, (int8_t)inpZp, vsz*sizeof(rowbuf[0]));
                                            for( k = 0; k < ncn; k++ )
                                            {
                                                for( i = i0; i < i1; i++ )
                                                {
                                                    int imgofs = k*width + i*dilation_w;
                                                    rowbuf[k*kernel_w + i] = imgptr[imgofs];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        else if (isConv2D)
                        {
                            if( is1x1 && stride_w == 1 && stride_h == 1 )
                            {
                                const int8_t* imgptr = data_inp0 + (cn0*height + out_i)*width + out_j;
                                for( int j = 0; j < bsz; j++, rowbuf += vsz_a )
                                {
                                    if( j + 4 <= bsz )
                                    {
                                        k = 0;
                                        // TODO: check performance with and without this part
                                        for( ; k < vsz; k++ )
                                        {
                                            const int8_t* inp = imgptr + j + k*inpPlaneSize;
                                            int8_t v0 = inp[0], v1 = inp[1], v2 = inp[2], v3 = inp[3];
                                            rowbuf[k] = v0;
                                            rowbuf[k + vsz_a] = v1;
                                            rowbuf[k + vsz_a*2] = v2;
                                            rowbuf[k + vsz_a*3] = v3;
                                        }
                                        j += 3;
                                        rowbuf += vsz_a*3;
                                    }
                                    else
                                    {
                                        for( k = 0; k < vsz; k++ )
                                        {
                                            rowbuf[k] = imgptr[j + k*inpPlaneSize];
                                        }
                                    }
                                }
                            }
                            else
                            for( ofs = ofs0; ofs < ofs1; out_j = 0, ++out_i )
                            {
                                int delta = std::min(ofs1 - ofs, outW - out_j);
                                int out_j1 = out_j + delta;

                                int in_i = out_i * stride_h - pad_t;
                                int in_j = out_j * stride_w - pad_l;
                                const int8_t* imgptr = data_inp0 + (cn0*height + in_i)*width + in_j;
                                ofs += delta;

                                // do im2row for a part of input tensor
                                if( is1x1 )
                                {
                                    for( ; out_j < out_j1; out_j++, rowbuf += vsz_a, imgptr += stride_w )
                                    {
                                        for( k = 0; k < vsz; k++ )
                                            rowbuf[k] = imgptr[k*inpPlaneSize];
                                    }
                                }
                                else
                                {
                                    bool ok_i = 0 <= in_i && in_i < height - (kernel_h-1)*dilation_h;
                                    int i0 = std::max(0, (-in_i + dilation_h-1)/dilation_h);
                                    int i1 = std::min(kernel_h, (height - in_i + dilation_h-1)/dilation_h);

                                    for( ; out_j < out_j1; out_j++, rowbuf += vsz_a, imgptr += stride_w, in_j += stride_w )
                                    {
                                        // this condition should be true for most of the tensor elements, i.e.
                                        // most of the time the kernel aperture is inside the tensor X-Y plane.
                                        if( ok_i && out_j + 2 <= out_j1 && 0 <= in_j && in_j + stride_w*2 <= width - (kernel_w-1)*dilation_w )
                                        {
                                            for( k = 0; k < vsz; k++ )
                                            {
                                                int k1 = ofstab[k];
                                                int8_t v0 = imgptr[k1];
                                                int8_t v1 = imgptr[k1 + stride_w];
                                                rowbuf[k] = v0;
                                                rowbuf[k+vsz_a] = v1;
                                            }
                                            out_j++;
                                            rowbuf += vsz_a;
                                            imgptr += stride_w;
                                            in_j += stride_w;
                                        }
                                        else
                                        {
                                            int j0 = std::max(0, (-in_j + dilation_w-1)/dilation_w);
                                            int j1 = std::min(kernel_w, (width - in_j + dilation_w-1)/dilation_w);

                                            // here some non-continuous sub-row of the row will not be
                                            // filled from the tensor; we need to make sure that the uncovered
                                            // elements are explicitly set to 0's. the easiest way is to
                                            // set all the elements to 0's before the loop.
                                            memset(rowbuf, (int8_t)inpZp, vsz*sizeof(rowbuf[0]));
                                            for( k = 0; k < ncn; k++ )
                                            {
                                                for( i = i0; i < i1; i++ )
                                                {
                                                    for( j = j0; j < j1; j++ )
                                                    {
                                                        int imgofs = k*(width*height) + i*(dilation_h*width) + j*dilation_w;
                                                        rowbuf[(k*kernel_h + i)*kernel_w + j] = imgptr[imgofs];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        else
                        {
                            for( ofs = ofs0; ofs < ofs1; out_d += (out_i + 1) / outH, out_i = (out_i + 1) % outH, out_j = 0 )
                            {
                                int delta = std::min(ofs1 - ofs, outW - out_j);
                                int out_j1 = out_j + delta;

                                int in_d = out_d * stride_d - pad_d;
                                int in_i = out_i * stride_h - pad_t;
                                int in_j = out_j * stride_w - pad_l;
                                const int8_t* imgptr = data_inp0 + (cn0*depth*height + in_d*height + in_i)*width + in_j;
                                ofs += delta;

                                int d0 = std::max(0, (-in_d + dilation_d - 1) / dilation_d);
                                int d1 = std::min(kernel_d, (depth - in_d + dilation_d - 1) / dilation_d);

                                int i0 = std::max(0, (-in_i + dilation_h-1)/dilation_h);
                                int i1 = std::min(kernel_h, (height - in_i + dilation_h-1)/dilation_h);

                                for( ; out_j < out_j1; out_j++, rowbuf += vsz_a, imgptr += stride_w, in_j += stride_w )
                                {
                                    int j0 = std::max(0, (-in_j + dilation_w-1)/dilation_w);
                                    int j1 = std::min(kernel_w, (width - in_j + dilation_w-1)/dilation_w);

                                    // here some non-continuous sub-row of the row will not be
                                    // filled from the tensor; we need to make sure that the uncovered
                                    // elements are explicitly set to 0's. the easiest way is to
                                    // set all the elements to 0's before the loop.
                                    memset(rowbuf, (int8_t)inpZp, vsz*sizeof(rowbuf[0]));
                                    for( k = 0; k < ncn; k++ )
                                    {
                                        for ( d = d0; d < d1; d++)
                                        {
                                            for( i = i0; i < i1; i++ )
                                            {
                                                for( j = j0; j < j1; j++ )
                                                {
                                                    int imgofs = k*(depth*width*height) + d*dilation_d*width*height + i*(dilation_h*width) + j*dilation_w;
                                                    rowbuf[(k*kernel_d*kernel_h + d*kernel_h + i)*kernel_w + j] = imgptr[imgofs];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        // now compute dot product of the weights
                        // and im2row-transformed part of the tensor
                    #if CV_TRY_AVX512_SKX
                        if(useAVX512)
                            opt_AVX512_SKX::fastConv(wptr, wstep, biasptr, rowbuf0, data_out0 + ofs0,
                                          outShape, bsz, vsz, vsz_a, outZp, multiplier, relu,
                                          cn0 == 0, cn1 == inpCn);
                        else
                    #endif
                    #if CV_TRY_AVX2
                        if(useAVX2)
                            opt_AVX2::fastConv(wptr, wstep, biasptr, rowbuf0, data_out0 + ofs0,
                                          outShape, bsz, vsz, vsz_a, outZp, multiplier, relu,
                                          cn0 == 0, cn1 == inpCn);
                        else
                    #endif
                        for( int i = 0; i < outCn; i += 2 )
                        {
                            const int8_t* wptr0 = wptr + i*wstep;
                            const int8_t* wptr1 = wptr0 + wstep;
                            int* outptr0 = data_out0 + ofs0 + i*outPlaneSize;
                            int* outptr1 = outptr0 + outPlaneSize;
                            int bias0 = biasptr[i], bias1 = biasptr[i+1];
                            int mult0 = multiplier[i], mult1 = multiplier[i+1];

                            if( i+1 >= outCn )
                            {
                                wptr1 = wptr0;
                                outptr1 = outptr0;
                                bias1 = bias0;
                                mult1 = mult0;
                            }
                            int j = 0;
                        #if CV_SIMD128 || useAVX
                            for( ; j <= bsz - 4; j += 4 )
                            {
                                const int8_t* rptr = rowbuf0 + j*vsz_a;
                                v_int32x4 s0, s1;

                                if( cn0 == 0 )
                                {
                                    s0 = v_setall_s32(bias0);
                                    s1 = v_setall_s32(bias1);
                                }
                                else
                                {
                                    s0 = v_load(outptr0 + j);
                                    s1 = v_load(outptr1 + j);
                                }

                                v_int32x4 vs00 = v_setzero_s32(), vs01 = v_setzero_s32(),
                                          vs02 = v_setzero_s32(), vs03 = v_setzero_s32(),
                                          vs10 = v_setzero_s32(), vs11 = v_setzero_s32(),
                                          vs12 = v_setzero_s32(), vs13 = v_setzero_s32();
                                for( k = 0; k < vsz; k += 16, rptr += 16 )
                                {
                                    v_int8x16 w0 = v_load_aligned(wptr0 + k);
                                    v_int8x16 w1 = v_load_aligned(wptr1 + k);
                                    v_int8x16 r0 = v_load_aligned(rptr);
                                    v_int8x16 r1 = v_load_aligned(rptr + vsz_a);
                                    v_int8x16 r2 = v_load_aligned(rptr + vsz_a*2);
                                    v_int8x16 r3 = v_load_aligned(rptr + vsz_a*3);

                                    vs00 = v_dotprod_expand_fast(w0, r0, vs00);
                                    vs01 = v_dotprod_expand_fast(w0, r1, vs01);
                                    vs02 = v_dotprod_expand_fast(w0, r2, vs02);
                                    vs03 = v_dotprod_expand_fast(w0, r3, vs03);

                                    vs10 = v_dotprod_expand_fast(w1, r0, vs10);
                                    vs11 = v_dotprod_expand_fast(w1, r1, vs11);
                                    vs12 = v_dotprod_expand_fast(w1, r2, vs12);
                                    vs13 = v_dotprod_expand_fast(w1, r3, vs13);
                                }
                                s0 += v_reduce_sum4(vs00, vs01, vs02, vs03);
                                s1 += v_reduce_sum4(vs10, vs11, vs12, vs13);
                                if( cn1 == inpCn )
                                {
                                    s0 = v_outputStage(s0, v_setall_s32(mult0), outZp);
                                    s1 = v_outputStage(s1, v_setall_s32(mult1), outZp);
                                }
                                v_store(outptr0 + j, s0);
                                v_store(outptr1 + j, s1);
                            }
                        #endif
                            for( ; j < bsz; j++ )
                            {
                                const int8_t* rptr = rowbuf0 + j*vsz_a;
                                int s00, s10;

                                if( cn0 == 0 )
                                {
                                    s00 = bias0;
                                    s10 = bias1;
                                }
                                else
                                {
                                    s00 = outptr0[j];
                                    s10 = outptr1[j];
                                }

                                for( k = 0; k < vsz; k++ )
                                {
                                    int8_t r0 = rptr[k];
                                    s00 += (int16_t)wptr0[k] * (int16_t)r0;
                                    s10 += (int16_t)wptr1[k] * (int16_t)r0;
                                }
                                if( cn1 == inpCn )
                                {
                                    int mul_round0 = (s00*mult0 + (1 << 21)) >> 22;
                                    int mul_round1 = (s10*mult1 + (1 << 21)) >> 22;

                                    int out0 = outZp + mul_round0;
                                    int out1 = outZp + mul_round1;

                                    s00 = std::min(std::max(out0, -128), 127);
                                    s10 = std::min(std::max(out1, -128), 127);
                                }

                                outptr0[j] = s00;
                                outptr1[j] = s10;
                            }
                        }
                    }
                }
                /*if( activ_ )
                    activ_->forwardSlice(data_out0 + stripeStart, data_out0 + stripeStart,
                                         (int)(stripeEnd - stripeStart),
                                         outPlaneSize, startOutCn, startOutCn + outCn);*/
            }
        }
    };

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

#if CV_SSE3
        uint32_t ftzMode = _MM_GET_FLUSH_ZERO_MODE();
        uint32_t dazMode = _MM_GET_DENORMALS_ZERO_MODE();
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        // Quantize FP32 inputs to int8
        if (inputs[0].type() == CV_32F)
            inputs[0].convertTo(inputs[0], CV_8S, 1.f/input_sc, input_zp);

        /*if (inputs[0].dims > 3) {
            printf("conv %s: input (%d x %d x %d x %d), kernel (%d x %d), pad (%d x %d), stride (%d x %d), dilation (%d x %d)\n",
                   name.c_str(), inputs[0].size[0], inputs[0].size[1], inputs[0].size[2], inputs[0].size[3],
                   kernel.width, kernel.height, pad.width, pad.height,
                   stride.width, stride.height, dilation.width, dilation.height);
        }
        else {
            printf("conv %s: input (%d x %d x %d), kernel (%d x %d), pad (%d x %d), stride (%d x %d), dilation (%d x %d)\n",
                   name.c_str(), inputs[0].size[0], inputs[0].size[1], inputs[0].size[2],
                   kernel.width, kernel.height, pad.width, pad.height,
                   stride.width, stride.height, dilation.width, dilation.height);
        }*/

        int outCn = blobs[1].size[0];
        int inpGroupCn = blobs[1].size[1];
        CV_Assert_N(inputs.size() == (size_t)1, inputs[0].size[1] % inpGroupCn == 0,
                    outputs.size() == 1, inputs[0].data != outputs[0].data);

        int ngroups = inputs[0].size[1] / inpGroupCn;
        CV_Assert(ngroups == 1); // TODO : Depthwise Convolution
        CV_Assert(outputs[0].size[1] % ngroups == 0);

        reluslope.clear();
        if( activ )
        {
            Ptr<ReLULayer> activ_relu = activ.dynamicCast<ReLULayer>();
            if( !activ_relu.empty() )
            {
                reluslope.assign(outCn+2, activ_relu->negativeSlope);
            }

            Ptr<ChannelsPReLULayer> activ_chprelu = activ.dynamicCast<ChannelsPReLULayer>();
            if( !activ_chprelu.empty() )
            {
                const Mat& m = activ_chprelu->blobs[0];
                CV_Assert(m.isContinuous() && m.type() == CV_32F && (int)m.total() == outCn);
                const float* mdata = m.ptr<float>();
                reluslope.resize(outCn+2);
                std::copy(mdata, mdata + outCn, reluslope.begin());
                reluslope[outCn] = reluslope[outCn+1] = reluslope[outCn-1];
            }
        }
        int nstripes = std::max(getNumThreads(), 1);
        Mat output_int32 = Mat(shape(outputs[0]), CV_32S);

        ParallelConv::run(inputs[0], output_int32, weightsMat, blobs[0], biasvec, reluslope, kernel_size, strides,
                          pads_begin, pads_end, dilations, activ.get(), ngroups, nstripes, input_zp, output_zp);

        // Dequantize outputs to FP32 or typecast outputs to int8.
        if (outputs[0].type() == CV_32F)
            output_int32.convertTo(outputs[0], CV_32F, output_sc, -(output_sc * output_zp));
        else
            output_int32.convertTo(outputs[0], CV_8S);

#if CV_SSE3
        _MM_SET_FLUSH_ZERO_MODE(ftzMode);
        _MM_SET_DENORMALS_ZERO_MODE(dazMode);
#endif
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == outputs.size());

        int64 flops = 0;
        int karea = std::accumulate(kernel_size.begin(), kernel_size.end(), 1, std::multiplies<size_t>());
        for (int i = 0; i < outputs.size(); i++)
        {
            flops += total(outputs[i])*(CV_BIG_INT(2)*karea*inputs[i][1] + 1);
        }
        return flops;
    }
};

Ptr<BaseConvolutionLayer> ConvolutionLayerInt8::create(const LayerParams &params)
{
    return Ptr<BaseConvolutionLayer>(new ConvolutionLayerInt8Impl(params));
}

}
}
