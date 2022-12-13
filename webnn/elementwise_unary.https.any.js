// META: title=test WebNN API element-wise unary operations
// META: global=window,dedicatedworker
// META: script=./resources/utils.js
// META: timeout=long

'use strict';

// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-unary

testWebNNOperation('abs', '/webnn/resources/test_data/abs.json', buildOperationWithSingleInput);
testWebNNOperation('ceil', '/webnn/resources/test_data/ceil.json', buildOperationWithSingleInput);
testWebNNOperation('cos', '/webnn/resources/test_data/cos.json', buildOperationWithSingleInput);
testWebNNOperation('exp', '/webnn/resources/test_data/exp.json', buildOperationWithSingleInput);
testWebNNOperation('floor', '/webnn/resources/test_data/floor.json', buildOperationWithSingleInput);
testWebNNOperation('log', '/webnn/resources/test_data/log.json', buildOperationWithSingleInput);
testWebNNOperation('neg', '/webnn/resources/test_data/neg.json', buildOperationWithSingleInput);
testWebNNOperation('sin', '/webnn/resources/test_data/sin.json', buildOperationWithSingleInput);
testWebNNOperation('tan', '/webnn/resources/test_data/tan.json', buildOperationWithSingleInput);