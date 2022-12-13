// META: title=test WebNN API conv2d operation
// META: global=window,dedicatedworker
// META: script=./resources/utils.js
// META: timeout=long

'use strict';

// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-conv2d

const buildConv2d= (operationName, builder, resources) => {
  // MLOperand conv2d(MLOperand input, MLOperand filter, optional MLConv2dOptions options = {});
  const namedOutputOperand = {};
  const [inputOperand, filterOperand] = createMultiInputOperands(builder, resources);
  const conv2dOptions = {};
  const options = resources.options;
  if (options) {
    if (options.padding) {
      conv2dOptions.padding = options.padding;
    }
    if (options.strides) {
      conv2dOptions.strides = options.strides;
    }
    if (options.dilations) {
      conv2dOptions.dilations = options.dilations;
    }
    if (options.autoPad) {
      conv2dOptions.autoPad = options.autoPad;
    }
    if (options.groups) {
      conv2dOptions.groups = options.groups;
    }
    if (options.inputLayout) {
      conv2dOptions.inputLayout = options.inputLayout;
    }
    if (options.filterLayout) {
      conv2dOptions.filterLayout = options.filterLayout;
    }
    if (options.bias) {
      conv2dOptions.bias = createConstantOperand(builder, options.bias);
    }
    if (options.activation) {
      conv2dOptions.activation = builder[options.activation]();
    }
  }
  namedOutputOperand[resources.expected.name] = builder[operationName](inputOperand, filterOperand, conv2dOptions);
  return namedOutputOperand;
};

testWebNNOperation('conv2d', '/webnn/resources/test_data/conv2d.json', buildConv2d);