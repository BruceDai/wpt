// META: global=window,dedicatedworker
// META: script=/resources/WebIDLParser.js
// META: script=/resources/idlharness.js
// META: script=./resources/utils.js
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
      MLOperand: ['input', 'filter', 'output'],
      MLOperator: ['relu'],
      MLGraphBuilder: ['builder'],
      MLGraph: ['graph']
    });

    ExecuteArray.forEach(executeType => {
      const isSync = executeType === 'sync';
      if (self.GLOBAL.isWindow() && isSync) {
        return;
      }

      DeviceTypeArray.forEach(async (deviceType) => {
        if (isSync) {
          self.context = navigator.ml.createContextSync({deviceType});
        } else {
          self.context = await navigator.ml.createContext({deviceType});
        }

        self.builder = new MLGraphBuilder(context);
        self.input = builder.input('input', {type: 'float32', dimensions: [1, 1, 5, 5]});
        self.filter = builder.constant({type: 'float32', dimensions: [1, 1, 3, 3]}, new Float32Array(9).fill(1));
        self.relu = builder.relu();
        self.output = builder.conv2d(input, filter, {activation: relu, inputLayout: "nchw"});

        if (isSync) {
          self.graph = builder.buildSync({output});
        } else {
          self.graph = await builder.build({output});
        }
      });
    });
  }
);
