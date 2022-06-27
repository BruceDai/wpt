// META: title=test WebNN API relu operation
// META: global=window,dedicatedworker
// META: script=./resources/utils.js
// META: timeout=long

'use strict';

// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-relu

const inputData = [
  0.58585083,  1.1363881,   0.67161655,  -0.9741674,  -1.6196846,
  0.572627,    1.9026182,   -0.7756641,  -0.18808974, -1.0357478,
  1.1778295,   -2.305167,   -2.2636602,  0.3750199,   -0.08234365,
  -0.47962302, -0.3010948,  0.5369879,   -0.413804,   -1.096925,
  -0.9273629,  0.88833886,  -0.52474195, -1.3852776,  0.10217833,
  0.50499475,  1.3289608,   0.21790339,  -0.65971124, 0.47400787,
  0.7271749,   -0.03890531, -0.04459939, 0.2601329,   -0.06985649,
  0.2501139,   -1.0219133,  -1.1504377,  -0.83611137, 0.64221096,
  0.25879756,  1.040239,    -0.18669093, -1.1436414,  1.1445535,
  -0.01876706, 1.283455,    0.59794647,  2.1886187,   -0.21977298,
  0.90072393,  0.8913641,   -0.55512637, -0.17248231, -1.4617383,
  -1.5487962,  0.1265688,   0.7930071,   0.63802403,  0.3400246,
];
const expected = [
  0.58585083,  1.1363881,   0.67161655,  0.,          0.,
  0.572627,    1.9026182,   0.,          0.,          0.,
  1.1778295,   0.,          0.,          0.3750199,   0.,
  0.,          0.,          0.5369879,   0.,          0.,
  0.,          0.88833886,  0.,          0.,          0.10217833,
  0.50499475,  1.3289608,   0.21790339,  0.,          0.47400787,
  0.7271749,   0.,          0.,          0.2601329,   0.,
  0.2501139,   0.,          0.,          0.,          0.64221096,
  0.25879756,  1.040239,    0.,          0.,          1.1445535,
  0.,          1.283455,    0.59794647,  2.1886187,   0.,
  0.90072393,  0.8913641,   0.,          0.,          0.,
  0.,          0.1265688,   0.7930071,   0.63802403,  0.3400246,
];

const testRelu = async (inputShape, inputValue, expected, isAsync) => {
  let context;
  if (isAsync) {
    context = await navigator.ml.createContext();
  } else {
    context = navigator.ml.createContextSync();
  }
  const builder = new MLGraphBuilder(context);
  const x = builder.input('x', {type: 'float32', dimensions: inputShape});
  const y = builder.relu(x);
  let graph;
  if (isAsync) {
    graph = await builder.build({y});
  } else {
    graph = builder.buildSync({y});
  }
  const inputs = {
    'x': new Float32Array(inputValue),
  };
  const outputs = {'y': new Float32Array(sizeOfShape(inputShape))};
  if (isAsync) {
    await context.compute(graph, inputs, outputs);
  } else {
    context.computeSync(graph, inputs, outputs);
  }
  assert_array_approx_equals_ulp(outputs.y, expected, ULPTolerance.float32.relu, 'float32');
};

const tests = [
  {inputShape: [60], name: 'relu 1D'},
  {inputShape: [3, 20], name:'relu 2D'},
  {inputShape: [3, 4, 5], name:'relu 3D'},
  {inputShape: [3, 2, 2, 5], name:'relu 4D'},
  {inputShape: [3, 2, 2, 1, 5], name:'relu 5D'},
]

for (let i = 0; i < tests.length; i++) {
  if (self.GLOBAL.isWindow()) {
    promise_test(async () => {
      await testRelu(tests[i].inputShape, inputData, expected, true);
    }, tests[i].name);
  } else if (self.GLOBAL.isWorker()) {
    promise_test(async () => {
      await testRelu(tests[i].inputShape, inputData, expected, true);
    }, `${tests[i].name} async`);
    promise_test(async () => {
      await testRelu(tests[i].inputShape, inputData, expected, false);
    }, `${tests[i].name} sync`);
  }
}
