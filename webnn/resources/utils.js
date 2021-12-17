'use strict';

/**
 * Get broadcast shape with two input shapes.
 * @param {Array} shapeA
 * @param {Array} shapeB
 * @returns An {Array} shape.
 */
function getBroadcastShape(shapeA, shapeB) {
  const outShape = [];
  const lenA = shapeA.length;
  const lenB = shapeB.length
  const maxLen = Math.max(lenA, lenB);
  for (let i = 0; i < maxLen; i++) {
    let a = shapeA[lenA - i - 1];
    if (a === undefined) {
      a = 1;
    }
    let b = shapeB[lenB - i - 1];
    if (b == undefined) {
      b = 1;
    }
    if (a === 1) {
      outShape.unshift(b);
    } else if (b === 1) {
      outShape.unshift(a);
    } else if (a !== b) {
      throw Error(
        `Operands could not be broadcast together with shapes ` +
        `[${shapeA}] and [${shapeB}].`);
    } else {
      outShape.unshift(a);
    }
  }
  return outShape;
}

/**
 * Get tobe broadcasted dimensions array from input shape with target shape.
 * @param {Array} inShape - Input shape array
 * @param {Array} outShape - Target Shape array
 * @returns A tobe broadcasted dimensions array.
 */
 function getBroadcastDims(inShape, outShape) {
  // For example
  //   inShape = [3, 1, 1, 3]
  //   outShape = [3, 224, 224, 3]
  //   result = [1, 2] inShape[1] and inShape[2] need broadcast from 1 to 224.
 const inRank = inShape.length;
 const result = [];
 for (let i = 0; i < inRank; i++) {
   const index = inRank - 1 - i;
   const a = inShape[index] || 1;
   const b = outShape[outShape.length - 1 - i] || 1;
   if (b > 1 && a === 1) {
     result.unshift(index);
   }
 }
 return result;
}

/**
 * Compute the strides by given shape array.
 * @param {Array} shape
 * @returns A strides array.
 */
function computeStrides(shape) {
  const rank = shape.length;
  if (rank < 2) {
    return [];
  }
  const strides = new Array(rank - 1);
  strides[rank - 2] = shape[rank - 1];
  for (let i = rank - 3; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

/**
 * Compute the flat index by given indices in N-dimentional array.
 * @param {Array} indices - The indices of element in N-dimensional array.
 * @param {number} n - A dimentions of N-dimensional array.
 * @param {Array} strides - A strides array.
 * @returns A index number.
 */
function indicesToIndex(indices, n, strides) {
  if (n === 0) {
    // scalar
    return 0;
  } else if (n === 1) {
    // 1D
    return indices[0];
  }
  let index = indices[indices.length - 1];
  for (let i = 0; i < indices.length - 1; ++i) {
    index += strides[i] * indices[i];
  }
  return index;
}
/**
* Compute the indices in N-dimentional array by given flat index.
* @param {number} index - Index in flat array.
* @param {number} n - A dimentions of N-dimensional array.
* @param {Array} strides - A strides array.
* @returns An indices array.
*/
function indexToIndices(index, n, strides) {
  if (n === 0) {
    return [];
  } else if (n === 1) {
    return [index];
  }
  const outIndices = new Array(n);
  for (let i = 0; i < outIndices.length - 1; ++i) {
    outIndices[i] = Math.floor(index / strides[i]);
    index -= outIndices[i] * strides[i];
  }
  outIndices[outIndices.length - 1] = index;
  return outIndices;
}

/**
 * Get value from input value array by given indices and broadcast shape info.
 * @param { dimensions: Array, value: Array }  x - An input
 * @param {Array} indices - An indices array.
 * @param {Array} broadcastShape
 * @returns A number value from x.value array.
 */
function getValue(x, indices, broadcastShape) {
  const shape = x.dimensions;
  const len = shape.length;
  const targetIndices = indices.slice(-len);
  const s = computeStrides(shape);
  broadcastShape.forEach(d => targetIndices[d] = 0);
  const index = indicesToIndex(targetIndices, len, s);
  return x.value[index];
}

export {
  computeStrides,
  getBroadcastDims,
  getBroadcastShape,
  getValue,
  indicesToIndex,
  indexToIndices
};