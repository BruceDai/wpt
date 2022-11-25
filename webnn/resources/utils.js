'use strict';

const ExecutionArray = ['sync', 'async'];

// https://webmachinelearning.github.io/webnn/#enumdef-mldevicetype
const DeviceTypeArray = ['cpu', 'gpu'];

// https://webmachinelearning.github.io/webnn/#enumdef-mloperandtype
const TypedArrayDict = {
  float32: Float32Array,
  int32: Int32Array,
  uint32: Uint32Array,
  int8: Int8Array,
  uint8: Uint8Array,
};

const sizeOfShape = (array) => {
  return array.reduce((accumulator, currentValue) => accumulator * currentValue, 1);
};

/**
 * Get JSON resources from specified test resources file.
 * @param {String} file - A test resources file path
 * @returns {Object} Test resources
 */
const loadResources = (file) => {
  const loadJSON = () => {
    let xmlhttp = new XMLHttpRequest();
    xmlhttp.open("GET", file, false);
    xmlhttp.overrideMimeType("application/json");
    xmlhttp.send();
    if (xmlhttp.status == 200 && xmlhttp.readyState == 4) {
      return xmlhttp.responseText;
    } else {
      throw new Error(`Failed to load ${file}`);
    }
  };

  const json = loadJSON();
  return JSON.parse(json.replace(/\\"|"(?:\\"|[^"])*"|(\/\/.*|\/\*[\s\S]*?\*\/)/g, (m, g) => g ? "" : m));
};

/**
 * Get exptected data source from given resources with output name.
 * @param {Array} resources - An array of expected resources
 * @param {String} name - An output name
 * @returns {Number[]} An expected data array
 */
const getExpectedData = (resources, name) => {
  let data;
  for (let subResources of resources) {
    if (subResources.name === name) {
      data = subResources.data;
      break;
    }
  }
  if (data === undefined) {
    throw new Error(`Failed to get expected data sources by ${name}`);
  }
  return data;
};

/**
 * Get ULP tolerance of gemm operation.
 * @param {Object} resources - Resources used for building a graph
 * @param {String} [name] - An operation name
 * @returns {Number} A tolerance number
 */
const getGemmPrecisionTolerance = (resources, name) => {
  // GEMM : alpha * (A x B) + beta * C
  // An upper bound for the worst serial ordering is bounded by
  // the number of lossy operations, where matrix multiplication
  // is a dot product (mul and add times the number of elements)
  // plus bias operations.
  const shapeA = resources.inputs['a'].shape;
  const defaultOptions = {
    c: 0.0,
    alpha: 1.0,
    beta: 1.0,
    aTranspose: false,
    bTranspose: false
  }
  const options = resources.options ? resources.options : defaultOptions;
  const width = options.aTranspose ? shapeA[0] : shapeA[1];
  let tolerance = width * 2;
  if (options.alpha !== undefined && options.alpha !== 1.0) {
    tolerance++;
  }
  if (options.c && options.beta !== 0.0) {
    if (options.beta !== undefined && options.beta !== 1.0) {
      tolerance++;
    }
    tolerance++;
  }
  return tolerance;
};

/**
 * Get ULP tolerance of matmul operation.
 * @param {Object} resources - Resources used for building a graph
 * @param {String} [name] - An operation name
 * @returns {Number} A tolerance number
 */
const getMatmulPrecisionTolerance = (resources, name) => {
  // Matmul : Compute the matrix product of two input tensors.
  // If a is 1-D, it is converted to a 2-D tensor by prepending a 1 to its dimensions, [n] -> [1, n]
  const shapeA = resources.inputs['a'].shape;
  const tolerance = shapeA[shapeA.length - 1] * 2;
  return tolerance;
};

// Refer to precision metrics on https://github.com/webmachinelearning/webnn/issues/265#issuecomment-1256242643
const PrecisionMetrics = {
  batchNormalization: {ULP: {float32: 6, float16: 6}},
  clamp: {ULP: {float32: 0, float16: 0}},
  concat: {ULP: {float32: 0, float16: 0}},
  // TODO getConv2dPrecisionTolerance
  // conv2d: {ULP: {float32: getConv2dPrecisionTolerance, float16: getConv2dPrecisionTolerance}},
  // element-wise binary operations
  add: {ULP: {float32: 1, float16: 1}},
  sub: {ULP: {float32: 1, float16: 1}},
  mul: {ULP: {float32: 1, float16: 1}},
  div: {ULP: {float32: 2, float16: 2}},
  max: {ULP: {float32: 0, float16: 0}},
  min: {ULP: {float32: 0, float16: 0}},
  pow: {ULP: {float32: 32, float16: 2}},
  // element-wise unary operations
  abs: {ULP: {float32: 0, float16: 0}},
  ceil: {ULP: {float32: 0, float16: 0}},
  cos: {ATOL: {float32: 1/1024, float16: 1/512}},
  exp: {ULP: {float32: 32, float16: 1}},
  floor: {ULP: {float32: 0, float16: 0}},
  log: {ATOL: {float32: 1/1024, float16: 1/1024}},
  neg: {ULP: {float32: 0, float16: 0}},
  sin: {ATOL: {float32: 1/1024, float16: 1/512}},
  tan: {ATOL: {float32: 1/1024, float16: 1/512}},
  gemm: {ULP: {float32: getGemmPrecisionTolerance, float16: getGemmPrecisionTolerance}},
  leakRelu: {ULP: {float32: 1, float16: 1}},
  matmul: {ULP: {float32: getMatmulPrecisionTolerance, float16: getMatmulPrecisionTolerance}},
  // pooling operations
  // TODO getAveragePool2dPrecisionTolerance
  // averagePool2d: {ULP: {float32: getAveragePool2dPrecisionTolerance, float16: getAveragePool2dPrecisionTolerance}},
  maxPool2d: {ULP: {float32: 0, float16: 0}},
  // reduction operations
  reduceMax: {ULP: {float32: 0, float16: 0}},
  // TODO getReducePrecisionTolerance(resources, name)
  // using second name parameter for reduceMean op, IEPOE + 2 ULP
  // reduceMean: {ULP: {float32: getReducePrecisionTolerance, float16: getReducePrecisionTolerance}},
  reduceMin: {ULP: {float32: 0, float16: 0}},
  // reduceProduct: {ULP: {float32: getReducePrecisionTolerance, float16: getReducePrecisionTolerance}},
  // reduceSum: {ULP: {float32: getReducePrecisionTolerance, float16: getReducePrecisionTolerance}},
  relu: {ULP: {float32: 0, float16: 0}},
  reshape: {ULP: {float32: 0, float16: 0}},
  // float32 (leaving a few ULP for roundoff)
  sigmoid: {ULP: {float32: 32+2, float16: 3}},
  slice: {ULP: {float32: 0, float16: 0}},
  // TODO getSoftmaxPrecisionTolerance
  // softmax: {ULP: {float32: getSoftmaxPrecisionTolerance, float16: getSoftmaxPrecisionTolerance}},
  split: {ULP: {float32: 0, float16: 0}},
  squeeze: {ULP: {float32: 0, float16: 0}},
  tanh: {ATOL: {float32: 1/1024, float16: 1/512}},
  transpose: {ULP: {float32: 0, float16: 0}},
};

/**
 * Get precison tolerance value.
 * @param {String} name - An operation name
 * @param {String} metricType - Value: 'ULP', 'ATOL'
 * @param {Object} resources - Resources used for building a graph
 * @returns {Number} A tolerance number
 */
const getPrecisonTolerance = (name, metricType, resources) => {
  let tolerance = PrecisionMetrics[name][metricType][resources.type];
  // If the tolerance is dynamic, then evaluate the function to get the value.
  if (tolerance instanceof Function) {
    tolerance = tolerance(resources, name);
  }
  return tolerance;
};

/**
 * Get bitwise of the given value.
 * @param {Number} value
 * @param {String} dataType - A data type string, like "float32", "float16",
 *     more types, please see:
 *     https://webmachinelearning.github.io/webnn/#enumdef-mloperandtype
 * @return {Number} A 64-bit signed integer.
 */
const getBitwise = (value, dataType) => {
  const buffer = new ArrayBuffer(8);
  const int64Array = new BigInt64Array(buffer);
  int64Array[0] = value < 0 ? ~BigInt(0) : BigInt(0);
  let typedArray;
  if (dataType === "float32") {
    typedArray = new Float32Array(buffer);
  } else {
    throw new AssertionError(`Data type ${dataType} is not supported`);
  }
  typedArray[0] = value;
  return int64Array[0];
};

/**
 * Assert that each array property in ``actual`` is a number being close enough to the corresponding
 * property in ``expected`` by the acceptable ULP distance ``nulp`` with given ``dataType`` data type.
 *
 * @param {Array} actual - Array of test values.
 * @param {Array} expected - Array of values expected to be close to the values in ``actual``.
 * @param {Number} nulp - A BigInt value indicates acceptable ULP distance.
 * @param {String} dataType - A data type string, value: "float32",
 *     more types, please see:
 *     https://webmachinelearning.github.io/webnn/#enumdef-mloperandtype
 * @param {String} description - Description of the condition being tested.
 */
const assert_array_approx_equals_ulp = (actual, expected, nulp, dataType, description) => {
  /*
    * Test if two primitive arrays are equal within acceptable ULP distance
    */
  assert_true(actual.length === expected.length,
              `assert_array_approx_equals_ulp: ${description} lengths differ, expected ${expected.length} but got ${actual.length}`);
  let actualBitwise, expectedBitwise, distance;
  for (let i = 0; i < actual.length; i++) {
    actualBitwise = getBitwise(actual[i], dataType);
    expectedBitwise = getBitwise(expected[i], dataType);
    distance = actualBitwise - expectedBitwise;
    distance = distance >= 0 ? distance : -distance;
    assert_true(distance <= nulp,
                `assert_array_approx_equals_ulp: ${description} actual ${actual[i]} should be close enough to expected ${expected[i]} by the acceptable ${nulp} ULP distance, but they have ${distance} ULP distance`);
  }
};

/**
 * Assert actual results with expected results.
 * @param {String} name - An operation name
 * @param {(Number[]|Number)} actual
 * @param {(Number[]|Number)} expected
 * @param {Number} tolerance
 * @param {String} operandType  - An operand type string, value: "float32",
 *     more types, please see:
 *     https://webmachinelearning.github.io/webnn/#enumdef-mloperandtype
 * @param {String} metricType - Value: 'ULP', 'ATOL'
 */
const doAssert = (name, actual, expected, tolerance, operandType, metricType) => {
  const description = `test ${name} ${operandType}`;
  if (typeof expected === 'number') {
    // for checking a scalar output by matmul 1D x 1D
    expected = [expected];
    actual = [actual];
  }
  if (metricType === 'ULP') {
    assert_array_approx_equals_ulp(actual, expected, tolerance, operandType, description);
  } else if (metricType === 'ATOL') {
    assert_array_approx_equals(actual, expected, tolerance, description);
  }
};

/**
 * Check computed results with expected data.
 * @param {String} name - An operation name
 * @param {Object.<String, MLOperand>} namedOutputOperands
 * @param {Object.<MLNamedArrayBufferViews>} outputs - The resources of required outputs
 * @param {Object} resources - Resources used for building a graph
 */
const checkResults = (name, namedOutputOperands, outputs, resources) => {
  const operandType = resources.type;
  const metricType = PrecisionMetrics[name] ? Object.keys(PrecisionMetrics[name])[0] : 'ULP';
  const tolerance = getPrecisonTolerance(name, metricType, resources);
  const expected = resources.expected;
  let outputData;
  let expectedData;
  if (Array.isArray(expected)) {
    // check outputs by split or gru
    for (let operandName in namedOutputOperands) {
      outputData = outputs[operandName];
      expectedData = getExpectedData(expected, operandName);
      doAssert(name, outputData, expectedData, tolerance, operandType, metricType)
    }
  } else {
    const outputName = expected.name ? expected.name : 'output';
    outputData = outputs[outputName];
    expectedData = expected.data;
    doAssert(name, outputData, expectedData, tolerance, operandType, metricType)
  }
};

/**
 * Create input operands for a graph.
 * @param {MLGraphBuilder} builder - A ML graph builder
 * @param {Object} resources - Resources used for building a graph
 * @param {String[]} nameArray
 * @returns {MLOperand[]} Input operands
 */
const createInputOperands = (builder, resources, nameArray) => {
  let operands = [];
  nameArray.forEach(name => {
    operands.push(builder.input(name, {type: resources.type, dimensions: resources.inputs[name].shape}));
  });
  return operands;
};

/**
 * Build a graph.
 * @param {MLGraphBuilder} builder - A ML graph builder
 * @param {Object} resources - Resources used for building a graph
 * @param {Function} computeFunc - A compute function
 * @returns [namedOperands, inputs, outputs]
 */
const buildGraph = (builder, resources, computeFunc) => {
  const operandType = resources.type;
  const TestTypedArray = TypedArrayDict[operandType];
  const outputOperand = computeFunc(builder, resources);
  let inputs = {};
  let namedOperands = {};
  if (Array.isArray(resources.inputs)) {
    // the inputs of concat() is a sequence
    for (let subInput of resources.inputs) {
      inputs[subInput.name] = new TestTypedArray(subInput.data);
    }
  } else {
    for (let inputName in resources.inputs) {
      inputs[inputName] = new TestTypedArray(resources.inputs[inputName].data);
    }
  }
  let outputs = {};
  if (Array.isArray(resources.expected)) {
    // the outputs of split() or gru() is a sequence
    for (let i = 0; i < resources.expected.length; i++) {
      outputs[resources.expected[i].name] = new TestTypedArray(sizeOfShape(resources.expected[i].shape));
      namedOperands[resources.expected[i].name] = outputOperand[i];
    }
  } else {
    const outputName = resources.expected.name ? resources.expected.name : 'output';
    // matmul 1D with 1D produces a scalar which doesn't have its shape
    const shape = resources.expected.shape ? resources.expected.shape : [];
    outputs[outputName] = new TestTypedArray(sizeOfShape(shape));
    namedOperands[outputName] = outputOperand;
  }
  return [namedOperands, inputs, outputs];
};

/**
 * Build a graph, synchronously compile graph and execute, then check computed results.
 * @param {String} name - An operation name
 * @param {MLContext} context - A ML context
 * @param {MLGraphBuilder} builder - A ML graph builder
 * @param {Object} resources - Resources used for building a graph
 * @param {Function} computeFunc - A compute function
 */
const runSync = (name, context, builder, resources, computeFunc) => {
  // build a graph
  const [namedOutputOperands, inputs, outputs] = buildGraph(builder, resources, computeFunc);
  // synchronously compile the graph up to the output operand
  const graph = builder.buildSync(namedOutputOperands);
  // synchronously execute the compiled graph.
  context.computeSync(graph, inputs, outputs);
  checkResults(name, namedOutputOperands, outputs, resources);
};

/**
 * Build a graph, asynchronously compile graph and execute, then check computed results.
 * @param {String} name - An operation name
 * @param {MLContext} context - A ML context
 * @param {MLGraphBuilder} builder - A ML graph builder
 * @param {Object} resources - Resources used for building a graph
 * @param {Function} computeFunc - A compute function
 */
const run = async (name, context, builder, resources, computeFunc) => {
  // build a graph
  const [namedOutputOperands, inputs, outputs] = buildGraph(builder, resources, computeFunc);
  // asynchronously compile the graph up to the output operand
  const graph = await builder.build(namedOutputOperands);
  // asynchronously execute the compiled graph
  await context.compute(graph, inputs, outputs);
  checkResults(name, namedOutputOperands, outputs, resources);
};

/**
 * Run WebNN operation tests.
 * @param {String} name - An operation name
 * @param {String} file - A test resources file path
 * @param {Function} computeFunc - A compute function
 */
const testWebNNOperation = (name, file, computeFunc) => {
  const resources = loadResources(file);
  const tests = resources.tests;
  ExecutionArray.forEach(executionType => {
    const isSync = executionType === 'sync';
    if (self.GLOBAL.isWindow() && isSync) {
      return;
    }
    let context;
    let builder;
    if (isSync) {
      // test sync
      DeviceTypeArray.forEach(deviceType => {
        setup(() => {
          context = navigator.ml.createContextSync({deviceType});
          builder = new MLGraphBuilder(context);
        });
        for (const subTest of tests) {
          test(() => {
            runSync(name, context, builder, subTest, computeFunc);
          }, `${subTest.name} / ${subTest.type} / ${deviceType} / ${executionType}`);
        }
      });
    } else {
      // test async
      DeviceTypeArray.forEach(deviceType => {
        promise_setup(async () => {
          context = await navigator.ml.createContext({deviceType});
          builder = new MLGraphBuilder(context);
        });
        for (const subTest of tests) {
          promise_test(async () => {
            await run(name, context, builder, subTest, computeFunc);
          }, `${subTest.name} / ${subTest.type} / ${deviceType} / ${executionType}`);
        }
      });
    }
  });
};