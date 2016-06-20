/*!
 * Copyright (c) 2016 by Contributors
 * \file weighted_l1.cc
 * \brief 
 * \author Ye Yuan
*/
#include "./weighted_l1-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(WeightedL1Param param) {
  return new WeightedL1Op<cpu>(param);
}

Operator *WeightedL1Prop::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(WeightedL1Param);

MXNET_REGISTER_OP_PROPERTY(WeightedL1, WeightedL1Prop)
.describe("Perform a L1 loss with mask on input.")
.add_argument("data", "Symbol", "L1 loss with mask.")
.add_arguments(WeightedL1Param::__FIELDS__());

}  // namespace op
}  // namespace mxnet
