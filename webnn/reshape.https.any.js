// META: title=test WebNN API reshape operation
// META: global=window,dedicatedworker
// META: script=./resources/utils.js
// META: timeout=long

'use strict';

// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-reshape

const computeReshape = (builder, resources) => {
  // MLOperand reshape(MLOperand input, sequence<long> newShape);
  // use 'input' for input operand name
  const [inputOperand] = createInputOperands(builder, resources, ['input']);
  return builder.reshape(inputOperand, resources.newShape);
};

testWebNNOperation('reshape', '/webnn/resources/test_data/reshape.json', computeReshape);

