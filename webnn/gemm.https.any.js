// META: title=test WebNN API gemm operation
// META: global=window,dedicatedworker
// META: script=./resources/utils.js
// META: timeout=long

'use strict';

// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-gemm

const computeGemm = (builder, resources) => {
  // MLOperand gemm(MLOperand a, MLOperand b, optional MLGemmOptions options = {});
  // use 'a' and 'b' for input operands name
  const [inputOperandA, inputOperandB] = createInputOperands(builder, resources, ['a', 'b']);
  const operandType = resources.type;
  const TestTypedArray = TypedArrayDict[operandType];
  const options = resources.options;
  const gemmOptions = {};
  if (options !== undefined) {
    if (options.c !== undefined) {
      if (typeof options.c === 'number') {
        // scalar
        gemmOptions.c = new TestTypedArray([options.c])[0];
      } else {
        gemmOptions.c = builder.constant({type: operandType, dimensions: options.c.shape}, new TestTypedArray(options.c.data));
      }
    }
    if (options.alpha !== undefined) {
      // scalar
      gemmOptions.alpha = new TestTypedArray([options.alpha])[0];
    }
    if (options.beta !== undefined) {
      // scalar
      gemmOptions.beta = new TestTypedArray([options.beta])[0];
    }
    if (options.aTranspose !== undefined) {
      gemmOptions.aTranspose = options.aTranspose;
    }
    if (options.bTranspose !== undefined) {
      gemmOptions.bTranspose = options.bTranspose;
    }
  }
  return builder.gemm(inputOperandA, inputOperandB, gemmOptions);
};

testWebNNOperation('gemm', '/webnn/resources/test_data/gemm.json', computeGemm);