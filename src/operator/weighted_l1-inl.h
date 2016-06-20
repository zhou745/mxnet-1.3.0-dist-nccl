/*!
 * Copyright (c) 2016 by Contributors
 * \file weighted_l1-inl.h
 * \brief L1 loss with mask
 * \author Ye Yuan
*/

#ifndef MXNET_OPERATOR_WEIGHTED_L1_INL_H
#define MXNET_OPERATOR_WEIGHTED_L1_INL_H

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

/* Operation to get binary mask */
struct binary_mask {
	MSHADOW_XINLINE static real_t Map(real_t a) {
		return a > 0.0f ? 1.0f : 0.0f;
	}
};

enum WeightedL1OpInputs {kData, kLabel};
enum WeightedL1OpOutputs {kOut};

/* Parameters for layer */
struct WeightedL1Param: public dmlc::Parameter<WeightedL1Param> {
	float grad_scale;
	DMLC_DECLARE_PARAMETER(WeightedL1Param) {
    	DMLC_DECLARE_FIELD(grad_scale).set_default(1.0f)
    	.describe("Scale the regression gradient by a float factor");
  	};
};

/* Forward and Backward */
template<typename xpu>
class WeightedL1Op: public Operator {
public:
	explicit WeightedL1Op(WeightedL1Param param): param_(param) {}

	virtual void Forward(const OpContext &ctx,
						 					 const std::vector<TBlob> &in_data,
     					         const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {

		using namespace mshadow;
		using namespace mshadow::expr;
		CHECK_EQ(in_data.size(), 2) << "Weighted L1 Input: [data, label]";
		CHECK_EQ(out_data.size(), 1) << "Weighted L1 Output: [output]";
		Stream<xpu> *s = ctx.get_stream<xpu>();
		Tensor<xpu, 2> data = in_data[kData].FlatTo2D<xpu, real_t>(s);
		Tensor<xpu, 2> out = out_data[kOut].FlatTo2D<xpu, real_t>(s);
		/* Pass input data */
    out = mshadow::expr::F<mshadow_op::identity>(data);
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
	  
    grad = param_.grad_scale * mshadow::expr::F<mshadow_op::sign>(out - label) * 
            F<binary_mask>(label);
	}

private:
	WeightedL1Param param_;

}; // class WeightedL1Op


/* Decalre Factory function, used for dispatch specialization */
template<typename xpu>
Operator* CreateOp(WeightedL1Param param);

#if DMLC_USE_CXX11
class WeightedL1Prop: public OperatorProperty {
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
    CHECK_EQ(in_shape->size(), 2) << "Input: [data, label]";
    const TShape &dshape = (*in_shape)[kData];
    if (dshape.ndim() == 0) return false;
    SHAPE_ASSIGN_CHECK(*in_shape, kLabel, dshape);
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new WeightedL1Prop();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "WeightedL1";
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
  WeightedL1Param param_;
};         // class WeightedL1Prop
#endif     // DMLC_USE_CXX11


} // namespace op
} // namespace mxnet

#endif // MXNET_OPERATOR_WEIGHTED_L1_INL_H 
