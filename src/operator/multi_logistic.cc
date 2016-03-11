/*!
 * Copyright (c) 2015 by Contributors
 * \file multi_logistic.cc
 * \brief
 * \author Zehua Huang
*/
#include "./multi_logistic-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(MultiLogisticParam param) {
  return new MultiLogisticOp<cpu>(param);
}

Operator *MultiLogisticProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(MultiLogisticParam);

MXNET_REGISTER_OP_PROPERTY(MultiLogistic, MultiLogisticProp)
.describe("Perform a lp loss transformation on input.")
.add_argument("data", "Symbol", "Input data to multiple logistic loss.")
.add_arguments(MultiLogisticParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

