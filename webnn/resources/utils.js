'use strict';

// unit of least precision (ULP) is the spacing between two consecutive floating-point numbers.
// It is used as a measure of accuracy in numeric calculations.
const ULPTolerance = {
  // for single-precision floating-point
  'float32': {
    'clamp': 0,
    'concat': 0,
    'relu': 0,
    'reshape': 0,
    'slice': 0,
    'split': 0,
    'squeeze': 0,
    'transpose': 0
  }
}

function sizeOfShape(array) {
  return array.reduce((accumulator, currentValue) => accumulator * currentValue, 1);
}