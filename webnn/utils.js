'use strict';

export function sizeOfShape(array) {
  return array.reduce(
      (accumulator, currentValue) => accumulator * currentValue, 1);
}

export function createActivation(
    builder, activation, input = undefined, options = {}) {
  if (activation === 'relu') {
    return input === undefined ? builder.relu() : builder.relu(input);
  } else if (activation === 'relu6') {
    const clampOptions = {minValue: 0, maxValue: 6};
    return input === undefined ? builder.clamp(clampOptions) :
                                 builder.clamp(input, clampOptions);
  } else if (activation === 'sigmoid') {
    return input === undefined ? builder.sigmoid() : builder.sigmoid(input);
  } else if (activation === 'leakyRelu') {
    return input === undefined ? builder.leakyRelu(options) :
                                 builder.leakyRelu(input, options);
  } else {
    throw new Error(`activation ${activation} is not supported`);
  }
}
