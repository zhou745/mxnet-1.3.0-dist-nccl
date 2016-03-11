/*!
 * Copyright (c) 2015 by Contributors
 * \file multi_logistic.cu
 * \brief
 * \author Zehua Huang
*/

#include "./multi_logistic-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(MultiLogisticParam param) {
  return new MultiLogisticOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet

