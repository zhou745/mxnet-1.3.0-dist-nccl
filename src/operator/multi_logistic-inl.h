/*!
 * Copyright (c) 2015 by Contributors
 * \file multi_logistic-inl.h
 * \brief
 * \author Zehua Huang
*/
#ifndef MXNET_OPERATOR_MULTILOGISTIC_INL_H_
#define MXNET_OPERATOR_MULTILOGISTIC_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"


#include <iostream>
#include <fstream>
using namespace std;

namespace mxnet {
namespace op {

struct cmp{
  MSHADOW_XINLINE static real_t Map(real_t a, real_t b){
    return a > b ? 1.0f : -1.0f;
  }
};

struct abs{
  MSHADOW_XINLINE static real_t Map(real_t a){
    return a > 0 ? a : -a;
  }
};

enum MultiLogisticOpInputs {kData, kLabel};
enum MultiLogisticOpOutputs {kOut};

struct MultiLogisticParam : public dmlc::Parameter<MultiLogisticParam> {
  float grad_scale;
  float p;
  float weight;
  DMLC_DECLARE_PARAMETER(MultiLogisticParam) {
    DMLC_DECLARE_FIELD(p).set_default(2.0f)
        .describe("Scale the gradient by a float factor");
    DMLC_DECLARE_FIELD(grad_scale).set_default(1.0f)
    .describe("Scale the gradient by a float factor");
    DMLC_DECLARE_FIELD(weight).set_default(1.0f)
    .describe("postive weight");
  };
};

template<typename xpu>
class MultiLogisticOp : public Operator {
 public:
  explicit MultiLogisticOp(MultiLogisticParam param) : param_(param) {}

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {

  // do nothing
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2) << "Softmax Input: [data, label]";
    CHECK_EQ(out_data.size(), 1) << "Softmax Output: [output]";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> data = in_data[kData].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> out = out_data[kOut].FlatTo2D<xpu, real_t>(s);
    out = mshadow::expr::F<mshadow_op::sigmoid>(data);

  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {

    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_grad.size(), 1);
    CHECK_GE(in_grad.size(), 1);
    CHECK_GE(req.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> label = in_data[kLabel].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> out = out_data[kOut].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> grad = in_grad[kData].FlatTo2D<xpu, real_t>(s);

    grad = out - label;
//    grad = F<abs>(out - label);
//    grad = param_.p * F<mshadow_op::power>(grad, ScalarExp<real_t>(param_.p - 1));
//    grad = F<mshadow::op::mul>(grad, F<cmp>(out, label));

//    if (param_.grad_scale < 1.0) {
//      grad *= param_.grad_scale;
//    }
    grad = grad * label * param_.weight + grad * (1 - label);
//    for (int i = 0; i < grad.size(0); i++){
//      for (int j = 0; j < grad.size(1); j++){
//            if (grad[i][j][x][y] != 0)
//              LOG(INFO) << "grad " << i << " " << j << " " << x << " " << y << " " << grad[i][j][x][y];
//            if (label[i][j] != 0)
//                grad[i][j] *= param_.weight;
//              LOG(INFO) << "label " << i << " " << j << " " << x << " " << y << " " << grad[i][j][x][y];
//      }
//    }
  }

 private:
  MultiLogisticParam param_;
};  // class MultiLogisticOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(MultiLogisticParam param);

#if DMLC_USE_CXX11
class MultiLogisticProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "label"};
  }

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
    CHECK_EQ(in_shape->size(), 2) << "Input:[data, label]";
    const TShape &dshape = (*in_shape)[kData];
    if (dshape.ndim() == 0) return false;
    SHAPE_ASSIGN_CHECK(*in_shape, kLabel, dshape);
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new MultiLogisticProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "MultiLogistic";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {in_data[kLabel], out_data[kOut]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_data[kOut], in_grad[kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[kData], out_data[kOut]}};
  }

  Operator* CreateOperator(Context ctx) const;

 private:
  MultiLogisticParam param_;
};  // class MultiLogisticProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MULTILOGISTIC_INL_H_
