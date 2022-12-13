// META: title=test WebNN API reduction  operation
// META: global=window,dedicatedworker
// META: script=./resources/utils.js
// META: timeout=long

'use strict';

// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-reduce

testWebNNOperation('reduceMax', '/webnn/resources/test_data/reduce_max.json', buildOperationWithSingleInput);
testWebNNOperation('reduceMean', '/webnn/resources/test_data/reduce_mean.json', buildOperationWithSingleInput);
testWebNNOperation('reduceMin', '/webnn/resources/test_data/reduce_min.json', buildOperationWithSingleInput);
testWebNNOperation('reduceProduct', '/webnn/resources/test_data/reduce_product.json', buildOperationWithSingleInput);
testWebNNOperation('reduceSum', '/webnn/resources/test_data/reduce_sum.json', buildOperationWithSingleInput);