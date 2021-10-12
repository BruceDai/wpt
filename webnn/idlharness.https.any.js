// META: global=window,worker
// META: script=/resources/WebIDLParser.js
// META: script=/resources/idlharness.js
// META: script=https://webmachinelearning.github.io/webnn-polyfill/dist/webnn-polyfill.js
// META: timeout=long

// https://webmachinelearning.github.io/webnn/

'use strict';

idl_test(
  ['webnn'],
  ['html', 'WebIDL', 'webgl1', 'webgpu'],
  idl_array => {
    if (self.GLOBAL.isWindow()) {
      idl_array.add_objects({ Navigator: ['navigator'] });
    } else if (self.GLOBAL.isWorker()) {
      idl_array.add_objects({ WorkerNavigator: ['navigator'] });
    }

    idl_array.add_objects({
      NavigatorML: [],
      ML: ['navigator.ml'],
      MLContext: ['context'],
      MLOperand: ['input', 'filter'],
      MLOperator: ['relu'],
      MLGraphBuilder: ['builder'],
      MLGraph: ['graph']
    });

    self.context = navigator.ml.createContext();
    self.builder = new MLGraphBuilder(context);
    self.input = builder.input('input', {type: 'float32', dimensions: [1, 1, 5, 5]});
    self.filter = builder.constant({type: 'float32', dimensions: [1, 1, 3, 3]},new Float32Array(9).fill(1));
    self.relu = builder.relu();
    self.output = builder.conv2d(input, filter, {activation: relu});
    self.graph = builder.build({'output': output});
  }
);
