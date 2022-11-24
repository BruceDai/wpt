// META: title=test WebNN API concat operation
// META: global=window,dedicatedworker
// META: script=./resources/utils.js
// META: script=./webnn-polyfill.js
// META: timeout=long

'use strict';

// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-concat

const computeConcat = (builder, resources) => {
  // MLOperand concat(sequence<MLOperand> inputs, long axis);
  const inputOperands = [];
  for (let input of resources.inputs) {
    inputOperands.push(builder.input(input.name, {type: resources.type, dimensions: input.shape}));
  }
  return builder.concat(inputOperands, resources.axis);
};

testWebNNOperation('concat', '/webnn/resources/test_data/concat.json', computeConcat);