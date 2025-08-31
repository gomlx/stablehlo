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
