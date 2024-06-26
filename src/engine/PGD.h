#ifndef PGD_H
#define PGD_H

#include "CustomDNN.h"

#include <torch/torch.h>

torch::Tensor findDelta( CustomDNNImpl &model,
                         const torch::Tensor &X,
                         int y,
                         float epsilon,
                         float alpha,
                         int num_iter,
                         torch::Device device );
bool displayAdversarialExample( CustomDNNImpl &model,
                                const torch::Tensor &input,
                                int target,
                                torch::Device device );


#endif