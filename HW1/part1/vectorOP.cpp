#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  __pp_vec_float x; // every lane value
  __pp_vec_int y;
  __pp_vec_int zeroI = _pp_vset_int(0);
  __pp_vec_int oneI  = _pp_vset_int(1);
  __pp_vec_float clampMax = _pp_vset_float(9.999999f);
  __pp_vec_int count;
  __pp_vec_float result;
  __pp_mask maskAll, maskEqualZero, maskNotEqualZero, maskClamp, maskDo;

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    // every turn lanes, avoid to outbound
    int lanes = (N - i >= VECTOR_WIDTH) ? VECTOR_WIDTH : (N - i);
    maskAll = _pp_init_ones(lanes);
    // Batch Load
    _pp_vload_float(x, values + i, maskAll); // x = values[i];
    _pp_vload_int(y, exponents + i, maskAll); // int y = exponents[i]

    _pp_veq_int(maskEqualZero, y, zeroI, maskAll); // if (y == 0)
    _pp_vset_float(result, 1.f, maskEqualZero); // output[i] = 1.f;

    maskNotEqualZero = _pp_mask_not(maskEqualZero); // else
    _pp_vmove_float(result, x, maskNotEqualZero); // result = x;
    _pp_vsub_int(count, y, oneI, maskNotEqualZero); // int count = y - 1;
    
    // maskDo : count > zero , so need to do 
    _pp_vgt_int(maskDo, count, zeroI, maskNotEqualZero);
    // maskDo = [1,0,0,1] -> _pp_cntbits(maskDo) = 2
    while(_pp_cntbits(maskDo) > 0) { // while (count > 0)
      _pp_vmult_float(result, result, x, maskDo); // result *= x;
      _pp_vsub_int(count, count, oneI, maskDo); // count--;
      _pp_vgt_int(maskDo, count, zeroI, maskNotEqualZero);
    }
    // result > 9.999999f 
    _pp_vgt_float(maskClamp, result, clampMax, maskAll); // if (result > 9.999999f)
    _pp_vmove_float(result, clampMax, maskClamp); // result = 9.999999f;
    // write back
    _pp_vstore_float(output + i, result, maskAll); // output[i] = result;
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
  }

  return 0.0;
}