// META: title=test WebNN API matmul operation
// META: global=window,dedicatedworker
// META: script=./resources/utils.js
// META: timeout=long

'use strict';

// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-matmul

const computeMatmul = (builder, resources) => {
  // MLOperand matmul(MLOperand a, MLOperand b);
  // use 'a' and 'b' for input operands name
  const [inputOperandA, inputOperandB] = createInputOperands(builder, resources, ['a', 'b']);
  return builder.matmul(inputOperandA, inputOperandB);
};

testWebNNOperation('matmul', '/webnn/resources/test_data/matmul.json', computeMatmul);