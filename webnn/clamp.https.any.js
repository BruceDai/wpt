// META: title=test WebNN API clamp operation
// META: global=window,dedicatedworker
// META: script=./resources/utils.js
// META: timeout=long

'use strict';

// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-clamp

const computeClamp = (builder, resources) => {
  // MLOperand clamp(MLOperand x, optional MLClampOptions options = {});
  // use 'x' for input operand name
  const [inputOperand] = createInputOperands(builder, resources, ['x']);
  return builder.clamp(inputOperand, resources.options);
};

testWebNNOperation('clamp', '/webnn/resources/test_data/clamp.json', computeClamp);