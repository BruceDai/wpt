// META: title=test WebNN API slice operation
// META: global=window,dedicatedworker
// META: script=./resources/utils.js
// META: timeout=long

'use strict';

// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-slice

const sliceTests = () => {
  const resources = loadTestData('/webnn/resources/test_data/slice.json');
  const tests = resources.tests;
  const inputsData = resources.inputsData;
  const expectedData = resources.expectedData;
  const targetTests = [];
  for (const test of tests) {
    const expectedDataSource = test.expected.data;
    const inputDataSource = test.input.data;
    targetTests.push(
      {
        name: test.name,
        operandType: test.type,
        input: {shape: test.input.shape, data: inputsData[inputDataSource]},
        starts: test.starts,
        sizes: test.sizes,
        options: test.options,
        expected: {shape: test.expected.shape, data: {outputOperand: expectedData[expectedDataSource]}}
      }
    );
  }
  return targetTests;
}

const buildGraph = (builder, resources) => {
  const operandType = resources.operandType;
  const TestTypedArray = TypedArrayDict[operandType];
  const inputOperand = builder.input('x', {type: operandType, dimensions: resources.input.shape});
  const outputOperand = builder.slice(inputOperand, resources.starts, resources.sizes, resources.options);
  const inputs = {x: new TestTypedArray(resources.input.data)};
  const outputs = {outputOperand: new TestTypedArray(sizeOfShape(resources.expected.shape))};
  return [{outputOperand}, inputs, outputs];
};

testWebNNOperation('slice', sliceTests(), buildGraph);