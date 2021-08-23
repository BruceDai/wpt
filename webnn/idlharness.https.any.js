// META: global=window,worker
// META: script=/resources/WebIDLParser.js
// META: script=/resources/idlharness.js
// META: script=./dist/webnn-polyfill.js
// META: timeout=long

// https://webmachinelearning.github.io/webnn/

'use strict';

idl_test(
  ['webnn'],
  ['html', 'WebIDL', 'webgl1', 'webgpu'],
  async idl_array => {
    if (self.GLOBAL.isWindow()) {
      idl_array.add_objects({ Navigator: ['navigator'] });
    } else if (self.GLOBAL.isWorker()) {
      idl_array.add_objects({ WorkerNavigator: ['navigator'] });
    }

    idl_array.add_objects({
      NavigatorML: [],
      ML: ['navigator.ml'],
      MLContext: ['context'],
      MLOperand: ['constant1', 'input1'],
      MLOperator: [],
      MLGraphBuilder: ['builder'],
      MLGraph: ['graph']
    });

    self.context = navigator.ml.createContext();
    self.builder = new MLGraphBuilder(context);
    const TENSOR_DIMS = [1, 2, 2, 2];
    const TENSOR_SIZE = 8;
    const desc = {type: 'float32', dimensions: TENSOR_DIMS};
    const constantBuffer1 = new Float32Array(TENSOR_SIZE).fill(0.5);
    self.constant1 = builder.constant(desc, constantBuffer1);
    self.input1 = builder.input('input1', desc);
    const constantBuffer2 = new Float32Array(TENSOR_SIZE).fill(0.5);
    const constant2 = builder.constant(desc, constantBuffer2);
    const input2 = builder.input('input2', desc);
    const intermediateOutput1 = builder.add(constant1, input1);
    const intermediateOutput2 = builder.add(constant2, input2);
    const output = builder.mul(intermediateOutput1, intermediateOutput2);
    self.graph = builder.build({'output': output});
  }
);
