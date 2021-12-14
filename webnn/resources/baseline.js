// 'use strict';
import sizeOfShape from './common.js'
import * as utils from './utils.js'

/**
 * Compute the binary addition, subtraction, multiplication, division, maximum
 * minimum and power by given two inputs and opertion name.
 * @param {number} a
 * @param {number} b
 * @param {string} name - An operation name, value:
 *     'add'|'sub'|'mul'|'div'|'max'|'min'|'pow'
 * @returns A computed number result.
 */
function computeBinary(a, b, name) {
  let out;
  switch(name) {
    case 'add':
      out = a + b;
      break;
    case 'sub':
      out = a - b;
      break;
    case 'mul':
      out = a * b;
      break;
    case 'div':
      out = a / b;
      break;
    case 'max':
      out = Math.max(a, b);
      break;
    case 'min':
      out = Math.min(a, b);
      break;
    case 'pow':
      out = Math.pow(a, b);
      break;
    default:
      throw new Error(`'${name}' is not supported`);
  }
  return out;
}

/**
 * Binary operation by given two inputs and opertion name.
 * @param { dimensions: Array, value: Array } a
 * @param { dimensions: Array, value: Array } b
 * @param {string} name - An operation name, value:
 *     'add'|'sub'|'mul'|'div'|'max'|'min'|'pow'
 * @returns A computed { dimensions: Array, value: Array } result.
 */
function binary(a, b, name) {
    const outShape = utils.assertAndGetBroadcastShape(a.dimensions, b.dimensions);
    const aBroadcastDims = utils.getBroadcastDims(a.dimensions, outShape);
    const bBroadcastDims = utils.getBroadcastDims(b.dimensions, outShape);
    const outSize = sizeOfShape(outShape);
    const outValue = new Array(outSize);

    if (aBroadcastDims.length === 0 && bBroadcastDims.length === 0) {
      // For example:
      //   1. a.dimensions = b.dimensions = [m,n,p,q]
      //   or
      //   2. a.dimensions = [m,n,p,q] b.dimensions = [q] or [p,q] or [n,p,q]
      for (let i = 0; i < outSize; ++i) {
        outValue[i] = computeBinary(
            a.value[i % a.value.length], b.value[i % b.value.length], name);
      }
    } else {
      const strides = utils.computeStrides(outShape);
      for (let i = 0; i < outSize; ++i) {
        const loc = utils.indexToLoc(i, outShape.length, strides);
        const aValue = utils.getValue(a, loc, aBroadcastDims);
        const bValue = utils.getValue(b, loc, bBroadcastDims);
        outValue[i] = computeBinary(aValue, bValue, name);
      }
    }

    return {dimensions: outShape, value: outValue};
  };

  export {binary};