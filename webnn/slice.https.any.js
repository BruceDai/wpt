// META: title=test WebNN API slice operation
// META: global=window,dedicatedworker
// META: script=./resources/utils.js
// META: timeout=long

'use strict';

// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-slice

const computeSlice = (builder, resources) => {
  // MLOperand slice(MLOperand input, sequence<long> starts, sequence<long> sizes, optional MLSliceOptions options = {});
  // use 'input' for input operand name
  const [inputOperand] = createInputOperands(builder, resources, ['input']);
  return builder.slice(inputOperand, resources.starts, resources.sizes, resources.options);
};

testWebNNOperation('slice', '/webnn/resources/test_data/slice.json', computeSlice);