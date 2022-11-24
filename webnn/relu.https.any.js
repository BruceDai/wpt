// META: title=test WebNN API relu operation
// META: global=window,dedicatedworker
// META: script=./resources/utils.js
// META: timeout=long

'use strict';

// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-relu

const computeRelu = (builder, resources) => {
  // MLOperand relu(MLOperand x);
  // use 'x' for input operand name
  const [inputOperand] = createInputOperands(builder, resources, ['x']);
  return builder.relu(inputOperand);
};

testWebNNOperation('relu', '/webnn/resources/test_data/relu.json', computeRelu);