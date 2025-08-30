package stablehlo

import (
	"github.com/gomlx/stablehlo/internal/optypes"
	"github.com/gomlx/stablehlo/shapeinference"
	"github.com/gomlx/stablehlo/types"
)

// Compare implements the corresponding standard binary operation.
func (f *Function) Compare(lhs, rhs *Value, direction types.ComparisonDirection, compareType types.ComparisonType) (*Value, error) {
	op := optypes.Compare
	outputShape, err := shapeinference.CompareOp(lhs.shape, rhs.shape, direction, compareType)
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
