'use strict';
/**
 * Compute product size.
 * @param {Array} array
 * @returns A number size.
 */
function sizeOfShape(array) {
  return array.reduce(
      (accumulator, currentValue) => accumulator * currentValue, 1);
}

export {sizeOfShape};
export default sizeOfShape;