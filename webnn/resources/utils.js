'use strict';

function assertAndGetBroadcastShape(shapeA, shapeB) {
  const result = [];
  const l = Math.max(shapeA.length, shapeB.length);
  for (let i = 0; i < l; i++) {
      let a = shapeA[shapeA.length - i - 1];
      if (a == null) {
          a = 1;
      }
      let b = shapeB[shapeB.length - i - 1];
      if (b == null) {
          b = 1;
      }
      if (a === 1) {
          result.unshift(b);
      }
      else if (b === 1) {
          result.unshift(a);
      }
      else if (a !== b) {
          const errMsg = `Operands could not be broadcast together with shapes ` +
              `${shapeA} and ${shapeB}.`;
          throw Error(errMsg);
      }
      else {
          result.unshift(a);
      }
  }
  return result;
}

/**
 * Returns the dimensions in the input shape that are broadcasted to
 * produce the provided output shape.
 *
 * The returned dimensions are 0-indexed and sorted. An example:
 * inShape = [4, 1, 3]
 * outShape = [5, 4, 3, 3]
 * result = [1]. Dimension 1 (2nd dimension of input) gets broadcasted 1 => 3.
 */
function getBroadcastDims(inShape, outShape) {
  const inRank = inShape.length;
  const dims = [];
  for (let i = 0; i < inRank; i++) {
      const dim = inRank - 1 - i;
      const a = inShape[dim] || 1;
      const b = outShape[outShape.length - 1 - i] || 1;
      if (b > 1 && a === 1) {
          dims.unshift(dim);
      }
  }
  return dims;
}

function computeStrides(shape) {
  const rank = shape.length;
  if (rank < 2) {
      return [];
  }
  // Last dimension has implicit stride of 1, thus having D-1 (instead of D)
  // strides.
  const strides = new Array(rank - 1);
  strides[rank - 2] = shape[rank - 1];
  for (let i = rank - 3; i >= 0; --i) {
      strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

/**
 * Computes flat index for a given location (multidimentionsal index) in a
 * Tensor/multidimensional array.
 *
 * @param locs Location in the tensor.
 * @param rank Rank of the tensor.
 * @param strides Tensor strides.
 */
function locToIndex(locs, rank, strides) {
  if (rank === 0) {
      return 0;
  }
  else if (rank === 1) {
      return locs[0];
  }
  let index = locs[locs.length - 1];
  for (let i = 0; i < locs.length - 1; ++i) {
      index += strides[i] * locs[i];
  }
  return index;
}
/**
* Computes the location (multidimensional index) in a tensor/multidimentional
* array for a given flat index.
*
* @param index Index in flat array.
* @param rank Rank of tensor.
* @param strides Strides of tensor.
*/
function indexToLoc(index, rank, strides) {
  if (rank === 0) {
      return [];
  }
  else if (rank === 1) {
      return [index];
  }
  const locs = new Array(rank);
  for (let i = 0; i < locs.length - 1; ++i) {
      locs[i] = Math.floor(index / strides[i]);
      index -= locs[i] * strides[i];
  }
  locs[locs.length - 1] = index;
  return locs;
}

function getValue(x, indices, broadcastShape) {
  const shape = x.dimensions;
  const rank = shape.length;
  const targetIndices = indices.slice(-rank);
  const s = computeStrides(shape);
  broadcastShape.forEach(d => targetIndices[d] = 0);
  const index = locToIndex(targetIndices, rank, s);
  return x.value[index];
}

export {
  computeStrides,
  getBroadcastDims,
  assertAndGetBroadcastShape,
  getValue,
  locToIndex,
  indexToLoc
};