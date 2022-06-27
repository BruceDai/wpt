// META: global=window,dedicatedworker
// META: script=/resources/WebIDLParser.js
// META: script=/resources/idlharness.js
// META: timeout=long

// https://webmachinelearning.github.io/webnn/

'use strict';

idl_test(
  ['webnn'],
  ['html', 'webidl', 'webgl1', 'webgpu'],
  async (idl_array) => {
    if (self.GLOBAL.isWindow()) {
      idl_array.add_objects({ Navigator: ['navigator'] });
    } else if (self.GLOBAL.isWorker()) {
      idl_array.add_objects({ WorkerNavigator: ['navigator'] });
    }

    idl_array.add_objects({
      NavigatorML: ['navigator'],
      ML: ['navigator.ml'],
      MLContext: ['context'],
      MLOperand: ['input', 'filter'],
      MLOperator: ['relu'],
      MLGraphBuilder: ['builder'],
      MLGraph: ['graph']
    });

    const desc = {type: 'float32', dimensions: [1, 1, 5, 5]};
    // async
    self.context = await navigator.ml.createContext();
    // const tf = context.tf;
    // await tf.setBackend('wasm');
    // await tf.ready();
    self.builder = new MLGraphBuilder(context);
    self.input = builder.input('input', desc);
    self.filter = builder.constant({type: 'float32', dimensions: [1, 1, 3, 3]}, new Float32Array(9).fill(1));
    self.relu = builder.relu();
    self.output = builder.conv2d(input, filter, {activation: relu,inputLayout: "nchw"});
    self.graph = await builder.build({output});

    if (self.GLOBAL.isWorker()) {
      // sync
      self.context = navigator.ml.createContextSync();
      // const tf2 = context.tf;
      // await tf2.setBackend('wasm');
      // await tf2.ready(); 
      self.builder = new MLGraphBuilder(context);
      self.input = builder.input('input', desc);
      self.filter = builder.constant({type: 'float32', dimensions: [1, 1, 3, 3]}, new Float32Array(9).fill(1));
      self.relu = builder.relu();
      self.output = builder.conv2d(input, filter, {activation: relu,inputLayout: "nchw"});
      self.graph = builder.buildSync({output});
    }
  }
);
