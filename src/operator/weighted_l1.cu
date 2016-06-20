/*!
 * Copyright (c) 2016 by Contributors
 * \file weighted_l1.cu
 * \brief
 * \author Ye Yuan
*/

#include "./weighted_l1-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(WeightedL1Param param) {
  return new WeightedL1Op<gpu>(param);
}

}  // namespace op
}  // namespace mxnet
