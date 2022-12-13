// META: title=test WebNN API add operation
// META: global=window,dedicatedworker
// META: script=./resources/utils.js
// META: timeout=long

'use strict';

// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-binary

testWebNNOperation('add', './resources/test_data/add.json', buildOperationWithTwoInputs);