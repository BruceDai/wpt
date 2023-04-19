// META: title=test WebNN API instanceNormalization operation
// META: global=window,dedicatedworker
// META: script=./resources/utils.js
// META: timeout=long

'use strict';

// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-instancenorm

const buildInstanceNorm= (operationName, builder, resources) => {
  // MLOperand instanceNormalization(MLOperand input, optional MLInstanceNormalizationOptions options = {});
  const namedOutputOperand = {};
  const inputOperand = createSingleInputOperand(builder, resources);
  const instanceNormOptions = {...resources.options};
  if (instanceNormOptions.scale) {
    instanceNormOptions.scale = createConstantOperand(builder, instanceNormOptions.scale);
  }
  if (instanceNormOptions.bias) {
    instanceNormOptions.bias = createConstantOperand(builder, instanceNormOptions.bias);
  }
  // invoke builder.instanceNormalization()
  namedOutputOperand[resources.expected.name] = builder[operationName](inputOperand, instanceNormOptions);
  return namedOutputOperand;
};

testWebNNOperation('instanceNormalization', buildInstanceNorm);