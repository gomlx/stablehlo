package stablehlo

import (
	"github.com/gomlx/stablehlo/internal/optypes"
	"github.com/gomlx/stablehlo/shapeinference"
	"github.com/gomlx/stablehlo/types"
)

// Compare implements the corresponding standard binary operation.
func (f *Function) Compare(lhs, rhs *Value, direction types.ComparisonDirection, compareType types.ComparisonType) (*Value, error) {
	op := optypes.Compare
	outputShape, err := shapeinference.Compare(lhs.shape, rhs.shape, direction, compareType)
	if err != nil {
		return nil, err
	}
	stmt := f.addOp(op, outputShape, lhs, rhs)
	stmt.Attributes = map[string]any{
		"compare_type":         compareType,
		"comparison_direction": direction,
	}
	return stmt.Outputs[0], nil
}

// Complex returns the complex value by concatenating the real and imaginary parts element-wise.
func (f *Function) Complex(real, imag *Value) (*Value, error) {
	op := optypes.Complex
	outputShape, err := shapeinference.Complex(real.shape, imag.shape)
	if err != nil {
		return nil, err
	}
	return f.addOp(op, outputShape, real, imag).Outputs[0], nil
}

// Real returns the real part of the complex value.
func (f *Function) Real(complex *Value) (*Value, error) {
	op := optypes.Real
	outputShape, err := shapeinference.RealOrImag(complex.shape)
	if err != nil {
		return nil, err
	}
	return f.addOp(op, outputShape, complex).Outputs[0], nil
}

// Imag returns the real part of the complex value.
func (f *Function) Imag(complex *Value) (*Value, error) {
	op := optypes.Imag
	outputShape, err := shapeinference.RealOrImag(complex.shape)
	if err != nil {
		return nil, err
	}
	return f.addOp(op, outputShape, complex).Outputs[0], nil
}

// Clamp returns the minimum(maximum(x, min), max).
//
// The values max and min can either be a scalar or have the same shape as x.
//
// Clamp is not defined for booleans or complex numbers (the semantics would not be clear).
//
// Note: the order of the arguments in StableHLO is different from most ML libraries.
func (f *Function) Clamp(min, x, max *Value) (*Value, error) {
	op := optypes.Clamp
	outputShape, err := shapeinference.Clamp(min.shape, x.shape, max.shape)
	if err != nil {
		return nil, err
	}
	return f.addOp(op, outputShape, min, x, max).Outputs[0], nil
}
