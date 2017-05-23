/*!
 * Copyright (c) 2017 by Contributors
 * \file batch_norm-inl.h
 * \brief
 * \author Bing Xu, Chris Olivier
 */
#ifndef MXNET_OPERATOR_BATCH_NORM_INL_H_
#define MXNET_OPERATOR_BATCH_NORM_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./mshadow_op.h"
#include "./operator_common.h"
#include "mxnet_op.h"

#ifdef __GNUG__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif

namespace mxnet {
namespace op {

namespace batchnorm {
enum BatchNormOpInputs {kData, kGamma, kBeta};  // kGamma: weights, kBeta: biases
enum BatchNormOpOutputs {kOut, kMean, kVar};  // req, out_data
enum BatchNormOpAuxiliary {kMovingMean, kMovingVar};  // aux_states
}  // namespace batchnorm

/*! \brief Parameters for BatchNoram operator */
struct BatchNormParam : public dmlc::Parameter<BatchNormParam> {
  float eps;
  float momentum;
  bool fix_gamma;
  bool fix_linear_trans;
  bool use_global_stats;
  bool output_mean_var;
  bool cudnn_off;
  DMLC_DECLARE_PARAMETER(BatchNormParam) {
    DMLC_DECLARE_FIELD(eps).set_default(1e-3f)
    .describe("Epsilon to prevent div 0. "
              "Must be bigger than CUDNN_BN_MIN_EPSILON "
              "defined in cudnn.h when using cudnn (usually 1e-5)");
    DMLC_DECLARE_FIELD(momentum).set_default(0.9f)
    .describe("Momentum for moving average");
    DMLC_DECLARE_FIELD(fix_gamma).set_default(true)
    .describe("Fix gamma while training");
    DMLC_DECLARE_FIELD(fix_linear_trans).set_default(false)
    .describe("Fix linear transformation while training");
    DMLC_DECLARE_FIELD(use_global_stats).set_default(false)
    .describe("Whether use global moving statistics instead of local batch-norm. "
              "This will force change batch-norm into a scale shift operator.");
    DMLC_DECLARE_FIELD(output_mean_var).set_default(false)
    .describe("Output All,normal mean and var");
    DMLC_DECLARE_FIELD(cudnn_off).set_default(false)
      .describe("Do not select CUDNN operator, if available");
  }
};

/*! \brief Batch normalization operator */
template <typename xpu, typename DType, typename AccReal>
class BatchNormOp : public Operator {
 public:
  explicit BatchNormOp(BatchNormParam param) {
    this->param_ = param;
  }

  static inline bool IsWriting(const OpReqType ort) {
    return ort == kWriteTo || ort == kWriteInplace;
  }

  /*!
   * \brief perform a forward operation of Operator, save the output to TBlob.
   * \param ctx runtime context available to this call
   * \param in_data array of input data, it is const
   * \param req the request types of saving operation, can only be kWriteTo or kWriteInplace.
   * \param out_data array of output data, pointer is used to indicate that this is holder
   *        the space of TBlob in out_data must be pre-allocated with InferShape
   * \param aux_states Auxiliary states of operator. Normally operator doesn't
   *        need, epecial case like Batch Norm requires.
   * \sa OpReqType, OpContext
   */
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;

    CHECK_EQ(in_data.size(), 3U);
    CHECK_EQ(aux_states.size(), 2U);
    if (ctx.is_train) {
      CHECK_EQ(out_data.size(), 3U);
      CHECK_EQ(req.size(), 3U);
    } else {
      CHECK_GE(out_data.size(), 1U);
      CHECK_GE(req.size(), 1U);
      CHECK_EQ(req[batchnorm::kOut], kWriteTo);
    }
    Stream<xpu> *s = ctx.get_stream<xpu>();
    DoForward(s, ctx, in_data, req, out_data, aux_states);
  }

  /*!
   * \brief Perform a Backward Operation, write gradient to the in_grad.
   *
   * \note
   * Convention:
   *   out_grad.size() == OperatorProperty.NumVisibleOutputs()
   *   out_data.size() == OperatorProperty.NumOutputs()
   * out_data can contain additional invisible returns that remembers the
   * state carried from the Forward pass. For example mask in the dropout.
   * The gradients are passed from visible returns in this function.
   *
   * \par
   * Not all the TBlobs in the arguments will be available
   * if you override the DeclareBackwardDependency of corresponding OperatorProperty class.
   * Only the dependencies you declared will be available at corresponding position,
   * the rest of the parameters are simply dummy where you will get a nullptr.
   * You will be safe if you use the default DeclareBackwardDependency.
   * But only declare what you need will give engine more chance for optimization.
   *
   * \param ctx runtime context available to this call
   * \param out_grad the gradient value we get from of the Operator.
   * \param in_data the array of input data.
   * \param out_data the array of output data.
   * \param req request types of the saving operation, can be all types.
   * \param in_grad the array of gradient we need to write to.
   * \param aux_states Auxiliary states of operator. Normally operator doesn't need
   * \sa OperatorProperty, OpReqType, OpContext
   */
  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    CHECK_EQ(out_grad.size(), param_.output_mean_var ? 3U : 1U);
    CHECK_EQ(in_data.size(), 3U);
    CHECK_EQ(out_data.size(), 3U);
    CHECK_EQ(in_grad.size(), 3U);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data, grad, grad_in;
    const real_t scale = static_cast<real_t>(out_grad[batchnorm::kOut].shape_[1]) /
                         static_cast<real_t>(out_grad[batchnorm::kOut].shape_.Size());
    if (in_data[batchnorm::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(out_grad[batchnorm::kOut].shape_[0],
                               out_grad[batchnorm::kOut].shape_[1], 1, 1);
      data = in_data[batchnorm::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
      grad = out_grad[batchnorm::kOut].get_with_shape<xpu, 4, real_t>(dshape, s);
      grad_in = in_grad[batchnorm::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
    } else {
      data = in_data[batchnorm::kData].get<xpu, 4, real_t>(s);
      grad = out_grad[batchnorm::kOut].get<xpu, 4, real_t>(s);
      grad_in = in_grad[batchnorm::kData].get<xpu, 4, real_t>(s);
    }

    Tensor<xpu, 1> mean = out_data[batchnorm::kMean].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> var = out_data[batchnorm::kVar].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> slope = in_data[batchnorm::kGamma].get<xpu, 1, real_t>(s);
    // Tensor<xpu, 1> bias = in_data[kBeta].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> gslope = in_grad[batchnorm::kGamma].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> gbias = in_grad[batchnorm::kBeta].get<xpu, 1, real_t>(s);
    // update moving avg
    Tensor<xpu, 1> moving_mean = aux_states[batchnorm::kMovingMean].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> moving_var = aux_states[batchnorm::kMovingVar].get<xpu, 1, real_t>(s);

    if (param_.fix_gamma) slope = 1.f;

    if (ctx.is_train && !param_.use_global_stats) {
      // get requested temp space
      Tensor<xpu, 2> workspace = ctx.requested[batchnorm::kTempSpace].get_space<xpu>(
          mshadow::Shape2(3, mean.shape_[0]), s);
      Tensor<xpu, 1> gmean = workspace[0];
      Tensor<xpu, 1> gvar = workspace[1];
      Tensor<xpu, 1> tmp = workspace[2];

      moving_mean = moving_mean * param_.momentum + mean * (1 - param_.momentum);
      moving_var = moving_var * param_.momentum + var * (1 - param_.momentum);
      // cal
      gvar = sumall_except_dim<1>((grad * broadcast<1>(slope, data.shape_)) *
                                  (data - broadcast<1>(mean, data.shape_)) *
                                  -0.5f *
                                  F<mshadow_op::power>(broadcast<1>(var + param_.eps, data.shape_),
                                                       -1.5f));
      gmean = sumall_except_dim<1>(grad * broadcast<1>(slope, data.shape_));
      gmean *= -1.0f / F<mshadow_op::square_root>(var + param_.eps);
      tmp = scale * sumall_except_dim<1>(-2.0f * (data - broadcast<1>(mean, data.shape_)));
      tmp *= gvar;
      gmean += tmp;
      // assign
      if (!param_.fix_gamma || !param_.fix_linear_trans) {
        Assign(gslope, req[batchnorm::kGamma],
               sumall_except_dim<1>(
                   grad * (data - broadcast<1>(mean, data.shape_)) /
                   F<mshadow_op::square_root>(broadcast<1>(var + param_.eps, data.shape_))));
      } else {
        Assign(gslope, req[batchnorm::kGamma], 0.0f);
      }
      Assign(grad_in, req[batchnorm::kData],
             (grad * broadcast<1>(slope, data.shape_)) *
             broadcast<1>(1.0f / F<mshadow_op::square_root>(var + param_.eps), data.shape_) +
             broadcast<1>(gvar, data.shape_) * scale * 2.0f * (data - broadcast<1>(mean,
                                                                                   data.shape_)) +
             broadcast<1>(gmean, data.shape_) * scale);
      Assign(gbias, req[batchnorm::kBeta], sumall_except_dim<1>(grad));
    } else {
      // use global statistics with freeze moving mean and var.
      if (!param_.fix_gamma || !param_.fix_linear_trans) {
        Assign(gslope, req[batchnorm::kGamma],
               sumall_except_dim<1>(
                   grad * (data - broadcast<1>(moving_mean, data.shape_)) /
                   F<mshadow_op::square_root>(broadcast<1>(moving_var + param_.eps, data.shape_))));
      } else {
        Assign(gslope, req[batchnorm::kGamma], 0.0f);
      }
      Assign(gbias, req[batchnorm::kBeta], sumall_except_dim<1>(grad));
      Assign(grad_in, req[batchnorm::kData], (grad * broadcast<1>(slope, data.shape_)) *
             broadcast<1>(
                 1.0f / F<mshadow_op::square_root>(moving_var + param_.eps), data.shape_));
    }
  }

 private:
  void DoForward(mshadow::Stream<cpu> *stream,
                 const OpContext &ctx,
                 const std::vector<TBlob> &in_data,
                 const std::vector<OpReqType> &req,
                 const std::vector<TBlob> &out_data,
                 const std::vector<TBlob> &aux_states);

  void DoBackward(mshadow::Stream<cpu> *stream,
                  const OpContext &ctx,
                  const std::vector<TBlob> &out_grad,
                  const std::vector<TBlob> &in_data,
                  const std::vector<TBlob> &out_data,
                  const std::vector<OpReqType> &req,
                  const std::vector<TBlob> &in_grad,
                  const std::vector<TBlob> &aux_states);

#if MXNET_USE_CUDA
  void DoForward(mshadow::Stream<gpu> *stream,
                 const OpContext &ctx,
                 const std::vector<TBlob> &in_data,
                 const std::vector<OpReqType> &req,
                 const std::vector<TBlob> &out_data,
                 const std::vector<TBlob> &aux_states);
  void DoBackward(mshadow::Stream<gpu> *stream,
                  const OpContext &ctx,
                  const std::vector<TBlob> &out_grad,
                  const std::vector<TBlob> &in_data,
                  const std::vector<TBlob> &out_data,
                  const std::vector<OpReqType> &req,
                  const std::vector<TBlob> &in_grad,
                  const std::vector<TBlob> &aux_states);
#endif  // MXNET_USE_CUDA

  /*! \brief Batch normalization operator parameters */
  BatchNormParam param_;
};  // class BatchNormOp

template<typename xpu>
Operator *CreateOp(const BatchNormParam& param, const int dtype, const TShape& shape);

#if DMLC_USE_CXX11
class BatchNormProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 3U) << "Input:[data, gamma, beta]";
    const TShape &dshape = in_shape->at(0);

    if (dshape.ndim() == 0) {
      return false;
    }

    in_shape->at(1) = TShape(Shape1(dshape[1]));
    in_shape->at(2) = TShape(Shape1(dshape[1]));

    out_shape->clear();
    out_shape->push_back(dshape);             // kOut
    out_shape->push_back(Shape1(dshape[1]));  // kMean
    out_shape->push_back(Shape1(dshape[1]));  // kVar

    aux_shape->clear();
    aux_shape->push_back(Shape1(dshape[1]));  // kMovingMean
    aux_shape->push_back(Shape1(dshape[1]));  // kMovingVar
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    using namespace mshadow;
    CHECK_GE(in_type->size(), 1U);
    const int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    // For float16 input type beta, gamma, mean, and average are stored in float32.
    // For other input types, these parameters have the same type as input
    // NOTE: This requirement is from cuDNN (v. 4 and 5)
    int dtype_param;
    MSHADOW_REAL_TYPE_SWITCH_EX(dtype, DTypeX, AccRealX, {
         dtype_param = mshadow::DataType<AccRealX>::kFlag; });
    for (index_t i = 1; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype_param;
      } else {
        CHECK_EQ((*in_type)[i], dtype_param) << "This layer requires uniform type. "
                                             << "Expected " << dtype_param << " v.s. given "
                                             << (*in_type)[i] << " at " << ListArguments()[i];
      }
    }
    for (index_t i = 0; i < aux_type->size(); ++i) {
      if ((*aux_type)[i] != -1) {
        CHECK_EQ((*aux_type)[i], dtype_param) << "This layer requires uniform type. "
                                              << "Expected " << dtype_param << " v.s. given "
                                              << (*aux_type)[i] << " at " << ListArguments()[i];
      }
    }
    const size_t n_aux = this->ListAuxiliaryStates().size();
    aux_type->clear();
    for (size_t i = 0; i < n_aux; ++i) {
      aux_type->push_back(dtype_param);
    }
    const size_t n_out = this->ListOutputs().size();
    out_type->clear();
    out_type->push_back(dtype);
    for (size_t i = 1; i < n_out; ++i) {
      out_type->push_back(dtype_param);
    }
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new BatchNormProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "BatchNorm";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[batchnorm::kOut],
            out_data[batchnorm::kMean],
            out_data[batchnorm::kVar],
            in_data[batchnorm::kData],
            in_data[batchnorm::kGamma]
           };
  }

  int NumVisibleOutputs() const override {
    if (param_.output_mean_var) {
      return 3;
    }
    return 1;
  }

  int NumOutputs() const override {
    return 3;
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "gamma", "beta"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "mean", "var"};
  }

  std::vector<std::string> ListAuxiliaryStates() const override {
    return {"moving_mean", "moving_var"};
  }

  Operator* CreateOperator(Context ctx) const override {
      LOG(FATAL) << "Not Implemented.";
      return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
      std::vector<int> *in_type) const override;

  inline const BatchNormParam& getParam() const {
    return param_;
  }

 private:
  BatchNormParam param_;
};  // class BatchNormProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#ifdef __GNUG__
#pragma GCC diagnostic pop
#endif

#endif  // MXNET_OPERATOR_BATCH_NORM_INL_H_

