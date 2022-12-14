// META: title=test WebNN API operations
// META: global=window,dedicatedworker
// META: script=./resources/utils_worker.js
// META: timeout=long

'use strict';

// Part - test sync

// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-batchnorm

const buildBatchNorm = (operationName, builder, resources) => {
  // MLOperand batchNormalization(MLOperand input, MLOperand mean, MLOperand variance,
  //                              optional MLBatchNormalizationOptions options = {});
  const namedOutputOperand = {};
  const [inputOperand, meanOperand, varianceOperand] = createMultiInputOperands(builder, resources);
  const batchNormOptions = {...resources.options};
  if (batchNormOptions.scale) {
    batchNormOptions.scale = createConstantOperand(builder, batchNormOptions.scale);
  }
  if (batchNormOptions.bias) {
    batchNormOptions.bias = createConstantOperand(builder, batchNormOptions.bias);
  }
  if (batchNormOptions.activation) {
    batchNormOptions.activation = builder[batchNormOptions.activation]();
  }
  // invoke builder.batchNormalization()
  namedOutputOperand[resources.expected.name] =
      builder[operationName](inputOperand, meanOperand, varianceOperand, batchNormOptions);
  return namedOutputOperand;
};

testWebNNOperation('batchNormalization', './resources/test_data/batch_normalization.json', buildBatchNorm);


// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-clamp

testWebNNOperation('clamp', './resources/test_data/clamp.json', buildOperationWithSingleInput);

// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-concat

const buildConcat = (operationName, builder, resources) => {
  // MLOperand concat(sequence<MLOperand> inputs, long axis);
  const namedOutputOperand = {};
  const inputOperands = [];
  for (let input of resources.inputs) {
    inputOperands.push(builder.input(input.name, {type: input.type, dimensions: input.shape}));
  }
  // invoke builder.concat()
  namedOutputOperand[resources.expected.name] = builder[operationName](inputOperands, resources.axis);
  return namedOutputOperand;
};

testWebNNOperation('concat', './resources/test_data/concat.json', buildConcat);


// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-conv2d

const buildConv2d= (operationName, builder, resources) => {
  // MLOperand conv2d(MLOperand input, MLOperand filter, optional MLConv2dOptions options = {});
  const namedOutputOperand = {};
  const [inputOperand, filterOperand] = createMultiInputOperands(builder, resources);
  let conv2dOptions = {...resources.options};
  if (conv2dOptions.bias) {
    conv2dOptions.bias = createConstantOperand(builder, conv2dOptions.bias);
  }
  if (conv2dOptions.activation) {
    conv2dOptions.activation = builder[conv2dOptions.activation]();
  }
  namedOutputOperand[resources.expected.name] = builder[operationName](inputOperand, filterOperand, conv2dOptions);
  return namedOutputOperand;
};

testWebNNOperation('conv2d', './resources/test_data/conv2d.json', buildConv2d);

// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-binary

testWebNNOperation('add', './resources/test_data/add.json', buildOperationWithTwoInputs);
testWebNNOperation('sub', './resources/test_data/sub.json', buildOperationWithTwoInputs);
testWebNNOperation('mul', './resources/test_data/mul.json', buildOperationWithTwoInputs);
testWebNNOperation('div', './resources/test_data/div.json', buildOperationWithTwoInputs);
testWebNNOperation('max', './resources/test_data/max.json', buildOperationWithTwoInputs);
testWebNNOperation('min', './resources/test_data/min.json', buildOperationWithTwoInputs);
testWebNNOperation('pow', './resources/test_data/pow.json', buildOperationWithTwoInputs);

// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-unary

testWebNNOperation('abs', './resources/test_data/abs.json', buildOperationWithSingleInput);
testWebNNOperation('ceil', './resources/test_data/ceil.json', buildOperationWithSingleInput);
testWebNNOperation('cos', './resources/test_data/cos.json', buildOperationWithSingleInput);
testWebNNOperation('exp', './resources/test_data/exp.json', buildOperationWithSingleInput);
testWebNNOperation('floor', './resources/test_data/floor.json', buildOperationWithSingleInput);
testWebNNOperation('log', './resources/test_data/log.json', buildOperationWithSingleInput);
testWebNNOperation('neg', './resources/test_data/neg.json', buildOperationWithSingleInput);
testWebNNOperation('sin', './resources/test_data/sin.json', buildOperationWithSingleInput);
testWebNNOperation('tan', './resources/test_data/tan.json', buildOperationWithSingleInput);

// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-gemm

const buildGemm= (operationName, builder, resources) => {
  // MLOperand gemm(MLOperand a, MLOperand b, optional MLGemmOptions options = {});
  const namedOutputOperand = {};
  const [inputOperandA, inputOperandB] = createMultiInputOperands(builder, resources);
  let gemmOptions = {...resources.options};
  if (gemmOptions.c) {
    if (gemmOptions.c.shape) {
      gemmOptions.c = createConstantOperand(builder, gemmOptions.c);
    } else {
      // MLOperand c;
      // Create a single-value operand when c is a scalar
      gemmOptions.c = builder.constant(gemmOptions.c);
    }
  }
  namedOutputOperand[resources.expected.name] = builder[operationName](inputOperandA, inputOperandB, gemmOptions);
  return namedOutputOperand;
};

testWebNNOperation('gemm', './resources/test_data/gemm.json', buildGemm);

// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-leakyrelu

testWebNNOperation('leakyRelu', './resources/test_data/leaky_relu.json', buildOperationWithSingleInput);

// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-matmul

testWebNNOperation('matmul', './resources/test_data/matmul.json', buildOperationWithTwoInputs);

// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-pool2d

testWebNNOperation('averagePool2d', './resources/test_data/average_pool2d.json', buildOperationWithSingleInput);
testWebNNOperation('maxPool2d', './resources/test_data/max_pool2d.json', buildOperationWithSingleInput);

// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-reduce

testWebNNOperation('reduceMax', './resources/test_data/reduce_max.json', buildOperationWithSingleInput);
testWebNNOperation('reduceMean', './resources/test_data/reduce_mean.json', buildOperationWithSingleInput);
testWebNNOperation('reduceMin', './resources/test_data/reduce_min.json', buildOperationWithSingleInput);
testWebNNOperation('reduceProduct', './resources/test_data/reduce_product.json', buildOperationWithSingleInput);
testWebNNOperation('reduceSum', './resources/test_data/reduce_sum.json', buildOperationWithSingleInput);

// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-relu

testWebNNOperation('relu', './resources/test_data/relu.json', buildOperationWithSingleInput);

// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-reshape

const buildReshape = (operationName, builder, resources) => {
  // MLOperand reshape(MLOperand input, sequence<long> newShape);
  const namedOutputOperand = {};
  const inputOperand = createSingleInputOperand(builder, resources);
  // invoke builder.reshape()
  namedOutputOperand[resources.expected.name] = builder[operationName](inputOperand, resources.newShape);
  return namedOutputOperand;
};

testWebNNOperation('reshape', './resources/test_data/reshape.json', buildReshape);

// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-sigmoid

testWebNNOperation('sigmoid', './resources/test_data/sigmoid.json', buildOperationWithSingleInput);

// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-slice

const buildSlice = (operationName, builder, resources) => {
  // MLOperand slice(MLOperand input, sequence<long> starts, sequence<long> sizes, optional MLSliceOptions options = {});
  const namedOutputOperand = {};
  const inputOperand = createSingleInputOperand(builder, resources);
  // invoke builder.slice()
  namedOutputOperand[resources.expected.name] = builder[operationName](inputOperand, resources.starts, resources.sizes, resources.options);
  return namedOutputOperand;
};

testWebNNOperation('slice', './resources/test_data/slice.json', buildSlice);

// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-softmax

testWebNNOperation('softmax', './resources/test_data/softmax.json', buildOperationWithSingleInput);

// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-split

const buildSplit = (operationName, builder, resources) => {
  // sequence<MLOperand> split(MLOperand input,
  //                           (unsigned long or sequence<unsigned long>) splits,
  //                           optional MLSplitOptions options = {});
  const namedOutputOperand = {};
  const inputOperand = createSingleInputOperand(builder, resources);
  // invoke builder.split()
  const outputOperands = builder[operationName](inputOperand, resources.splits, resources.options);
  resources.expected.forEach((resourceDict, index) => {
    namedOutputOperand[resourceDict.name] = outputOperands[index];
  });
  return namedOutputOperand;
};

testWebNNOperation('split', './resources/test_data/split.json', buildSplit);

// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-squeeze

testWebNNOperation('squeeze', './resources/test_data/squeeze.json', buildOperationWithSingleInput);

// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-tanh

testWebNNOperation('tanh', './resources/test_data/tanh.json', buildOperationWithSingleInput);

// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-transpose

testWebNNOperation('transpose', './resources/test_data/transpose.json', buildOperationWithSingleInput);


// Part - test async
testWebNNOperation('batchNormalization', './resources/test_data/batch_normalization.json', buildBatchNorm);
testWebNNOperation('clamp', './resources/test_data/clamp.json', buildOperationWithSingleInput);
testWebNNOperation('concat', './resources/test_data/concat.json', buildConcat);
testWebNNOperation('conv2d', './resources/test_data/conv2d.json', buildConv2d);
testWebNNOperation('add', './resources/test_data/add.json', buildOperationWithTwoInputs);
testWebNNOperation('sub', './resources/test_data/sub.json', buildOperationWithTwoInputs);
testWebNNOperation('mul', './resources/test_data/mul.json', buildOperationWithTwoInputs);
testWebNNOperation('div', './resources/test_data/div.json', buildOperationWithTwoInputs);
testWebNNOperation('max', './resources/test_data/max.json', buildOperationWithTwoInputs);
testWebNNOperation('min', './resources/test_data/min.json', buildOperationWithTwoInputs);
testWebNNOperation('pow', './resources/test_data/pow.json', buildOperationWithTwoInputs);
testWebNNOperationP('abs', './resources/test_data/abs.json', buildOperationWithSingleInput);
testWebNNOperationP('ceil', './resources/test_data/ceil.json', buildOperationWithSingleInput);
testWebNNOperationP('cos', './resources/test_data/cos.json', buildOperationWithSingleInput);
testWebNNOperationP('exp', './resources/test_data/exp.json', buildOperationWithSingleInput);
testWebNNOperationP('floor', './resources/test_data/floor.json', buildOperationWithSingleInput);
testWebNNOperationP('log', './resources/test_data/log.json', buildOperationWithSingleInput);
testWebNNOperationP('neg', './resources/test_data/neg.json', buildOperationWithSingleInput);
testWebNNOperationP('sin', './resources/test_data/sin.json', buildOperationWithSingleInput);
testWebNNOperationP('tan', './resources/test_data/tan.json', buildOperationWithSingleInput);
testWebNNOperationP('gemm', './resources/test_data/gemm.json', buildGemm);
testWebNNOperationP('leakyRelu', './resources/test_data/leaky_relu.json', buildOperationWithSingleInput);
testWebNNOperationP('matmul', './resources/test_data/matmul.json', buildOperationWithTwoInputs);
testWebNNOperationP('averagePool2d', './resources/test_data/average_pool2d.json', buildOperationWithSingleInput);
testWebNNOperationP('maxPool2d', './resources/test_data/max_pool2d.json', buildOperationWithSingleInput);
testWebNNOperationP('reduceMax', './resources/test_data/reduce_max.json', buildOperationWithSingleInput);
testWebNNOperationP('reduceMean', './resources/test_data/reduce_mean.json', buildOperationWithSingleInput);
testWebNNOperationP('reduceMin', './resources/test_data/reduce_min.json', buildOperationWithSingleInput);
testWebNNOperationP('reduceProduct', './resources/test_data/reduce_product.json', buildOperationWithSingleInput);
testWebNNOperationP('reduceSum', './resources/test_data/reduce_sum.json', buildOperationWithSingleInput);
testWebNNOperationP('relu', './resources/test_data/relu.json', buildOperationWithSingleInput);
testWebNNOperationP('reshape', './resources/test_data/reshape.json', buildReshape);
testWebNNOperationP('sigmoid', './resources/test_data/sigmoid.json', buildOperationWithSingleInput);
testWebNNOperationP('slice', './resources/test_data/slice.json', buildSlice);
testWebNNOperationP('softmax', './resources/test_data/softmax.json', buildOperationWithSingleInput);
testWebNNOperationP('split', './resources/test_data/split.json', buildSplit);
testWebNNOperationP('squeeze', './resources/test_data/squeeze.json', buildOperationWithSingleInput);
testWebNNOperationP('tanh', './resources/test_data/tanh.json', buildOperationWithSingleInput);
testWebNNOperationP('transpose', './resources/test_data/transpose.json', buildOperationWithSingleInput);
