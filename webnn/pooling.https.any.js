// META: title=test WebNN API pooling operations
// META: global=window,dedicatedworker
// META: script=./resources/utils.js
// META: timeout=long

'use strict';

// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-pool2d

testWebNNOperation('averagePool2d', '/webnn/resources/test_data/average_pool2d.json', buildOperationWithSingleInput);
testWebNNOperation('maxPool2d', '/webnn/resources/test_data/max_pool2d.json', buildOperationWithSingleInput);