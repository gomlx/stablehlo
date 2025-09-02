package stablehlo

import (
	"fmt"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/stablehlo/internal/optypes"
	"github.com/gomlx/stablehlo/shapeinference"
	"github.com/gomlx/stablehlo/types"
	"github.com/gomlx/stablehlo/types/shapes"
	"github.com/pkg/errors"
)

// Compare implements the corresponding standard binary operation.
func (fn *Function) Compare(lhs, rhs *Value, direction types.ComparisonDirection, compareType types.ComparisonType) (*Value, error) {
	op := optypes.Compare
	outputShape, err := shapeinference.Compare(lhs.shape, rhs.shape, direction, compareType)
	if err != nil {
		return nil, err
	}
	stmt := fn.addOp(op, outputShape, lhs, rhs)
	stmt.Attributes = map[string]any{
		"compare_type":         compareType,
		"comparison_direction": direction,
	}
	return stmt.Outputs[0], nil
}

// Complex returns the complex value by concatenating the real and imaginary parts element-wise.
func (fn *Function) Complex(real, imag *Value) (*Value, error) {
	op := optypes.Complex
	outputShape, err := shapeinference.Complex(real.shape, imag.shape)
	if err != nil {
		return nil, err
	}
	return fn.addOp(op, outputShape, real, imag).Outputs[0], nil
}

// Real returns the real part of the complex value.
func (fn *Function) Real(complex *Value) (*Value, error) {
	op := optypes.Real
	outputShape, err := shapeinference.RealOrImag(complex.shape)
	if err != nil {
		return nil, err
	}
	return fn.addOp(op, outputShape, complex).Outputs[0], nil
}

// Imag returns the real part of the complex value.
func (fn *Function) Imag(complex *Value) (*Value, error) {
	op := optypes.Imag
	outputShape, err := shapeinference.RealOrImag(complex.shape)
	if err != nil {
		return nil, err
	}
	return fn.addOp(op, outputShape, complex).Outputs[0], nil
}

// Clamp returns the minimum(maximum(x, min), max).
//
// The values max and min can either be a scalar or have the same shape as x.
//
// Clamp is not defined for booleans or complex numbers (the semantics would not be clear).
//
// Note: the order of the arguments in StableHLO is different from most ML libraries.
func (fn *Function) Clamp(min, x, max *Value) (*Value, error) {
	op := optypes.Clamp
	outputShape, err := shapeinference.Clamp(min.shape, x.shape, max.shape)
	if err != nil {
		return nil, err
	}
	return fn.addOp(op, outputShape, min, x, max).Outputs[0], nil
}

// DotGeneralBuilder is a builder for DotGeneral nodes. See DotGeneral for more details.
type DotGeneralBuilder struct {
	fn                               *Function
	lhs                              *Value
	lhsContractingAxes, lhsBatchAxes []int
	rhs                              *Value
	rhsContractingAxes, rhsBatchAxes []int

	precision   [2]types.DotGeneralPrecisionType
	outputDType dtypes.DType
	algorithm   *types.DotGeneralAlgorithm
}

// DotGeneral takes as input lhs (left-hand-side) and rhs (right-hand-side) specifications
// for a general vector product -- a generalized "Einsum". Each axis can be:
//   - Just aligned (batch axes), so the output has the same axes as the inputs. The dimensions
//     must match in lhs and rhs.
//   - Crossed (default), in which case the output is the combination (concatenation) of the
//     dimensions.
//   - Contracted (contracting axes), where the output does multiply the values and reduce sum
//     those dimensions.
//
// It follows that the resulting dimension number starts with the batch dimension, then the 'lhs'
// non-contracting/non-batch dimension and finally the 'rhs' non-contracting/non-batch dimension.
// It provides the basic means of implementing Einsum.
//
// Because there are optional parameters, this function returns a DotGeneralBuilder that can
// be further configured. Call DotGeneralBuilder.Done to get the final DotGeneral node.
//
// Example:
//
//	// Create a function with a single DotGeneral node.
//	f := NewFunction()
//	lhs := f.Constant(types.Float32, []float32{1, 2, 3, 4, 5, 6})
//	rhs := f.Constant(types.Float32, []float32{1, 2, 3, 4, 5, 6})
//	dot, err := f.DotGeneral(lhs, []int{-1}, []int{-2}, rhs, []int{-1}, []int{-2}).Done()
func (fn *Function) DotGeneral(
	lhsOp *Value, lhsContractingAxes, lhsBatchAxes []int,
	rhsOp *Value, rhsContractingAxes, rhsBatchAxes []int) *DotGeneralBuilder {
	return &DotGeneralBuilder{
		fn:                 fn,
		lhs:                lhsOp,
		lhsContractingAxes: lhsContractingAxes,
		lhsBatchAxes:       lhsBatchAxes,
		rhs:                rhsOp,
		rhsContractingAxes: rhsContractingAxes,
		rhsBatchAxes:       rhsBatchAxes,

		precision:   [2]types.DotGeneralPrecisionType{types.DotGeneralPrecisionDefault, types.DotGeneralPrecisionDefault},
		outputDType: lhsOp.shape.DType,
	}
}

// Precision sets the precision of the dot-general operation.
//
// Its default is described as "the fastest calculation, but the least accurate approximation to the original number."
//
// It controls the tradeoff between speed and accuracy for computations on accelerator backends.
// This can be one of the following (at the moment, the semantics of these enum values are underspecified,
// but they are planning to address this in #755 -- https://github.com/openxla/stablehlo/issues/755):
func (b *DotGeneralBuilder) Precision(lhsPrecision, rhsPrecision types.DotGeneralPrecisionType) *DotGeneralBuilder {
	b.precision[0] = lhsPrecision
	b.precision[1] = rhsPrecision
	return b
}

// OutputDType sets the output data type: for input types like BFloat16 one may want to increase the
// output precision.
func (b *DotGeneralBuilder) OutputDType(dtype dtypes.DType) *DotGeneralBuilder {
	b.outputDType = dtype
	return b
}

// Algorithm sets the algorithm settings to use for the dot-general operation.
//
// The default is not to set any of these parameters.
//
// See details in types.DotGeneralAlgorithm.
func (b *DotGeneralBuilder) Algorithm(algorithm *types.DotGeneralAlgorithm) *DotGeneralBuilder {
	b.algorithm = algorithm
	return b
}

// Done indicates the end of the DotGeneralBuilder configuration.
// It checks the validity of the parameters and shapes and returns the final DotGeneral node.
func (b *DotGeneralBuilder) Done() (*Value, error) {
	op := optypes.DotGeneral
	outputShape, err := shapeinference.DotGeneral(
		b.lhs.shape, b.lhsContractingAxes, b.lhsBatchAxes,
		b.rhs.shape, b.rhsContractingAxes, b.rhsBatchAxes,
		b.outputDType)
	if err != nil {
		return nil, err
	}
	stmt := b.fn.addOp(op, outputShape, b.lhs, b.rhs)
	dotConfig := fmt.Sprintf(`#stablehlo.dot<
      lhs_batching_dimensions = %s,
      rhs_batching_dimensions = %s,
      lhs_contracting_dimensions = %s,
      rhs_contracting_dimensions = %s
    >`, intSliceToStableHLO(b.lhsBatchAxes), intSliceToStableHLO(b.rhsBatchAxes),
		intSliceToStableHLO(b.lhsContractingAxes), intSliceToStableHLO(b.rhsContractingAxes))
	stmt.Attributes = map[string]any{
		"dot_dimension_numbers": literalStr(dotConfig),
	}
	//if b.algorithm != nil {
	//	stmt.Attributes["lhs_precision_type"] = b.algorithm.LhsPrecisionType
	//	stmt.Attributes["rhs_precision_type"] = b.algorithm.RhsPrecisionType
	//	stmt.Attributes["lhs_component_count"] = b.algorithm.LhsComponentCount
	//	stmt.Attributes["rhs_component_count"] = b.algorithm.RhsComponentCount
	//	stmt.Attributes["num_primitive_operations"] = b.algorithm.NumPrimitiveOperations
	//	stmt.Attributes["allow_imprecise_accumulation"] = b.algorithm.AllowImpreciseAccumulation
	//}
	return stmt.Outputs[0], nil
}

// Iota creates a constant of the given shape with increasing numbers (starting from 0)
// on the given axis. So Iota([2,2], 1) returns [[0 1][0 1]], while Iota([2,2], 0)
// returns [[0 0][1 1]].
func (fn *Function) Iota(shape shapes.Shape, axis int) (*Value, error) {
	op := optypes.Iota
	var err error
	axis, err = shapeinference.AdjustAxisToRank(shape.Rank(), axis)
	if err != nil {
		return nil, err
	}
	stmt := fn.addOp(op, shape)
	stmt.Attributes = map[string]any{"iota_dimension": int64(axis)}
	return stmt.Outputs[0], nil
}

// Reshape the operand to the given shape.
// The total size of the new shape must match the original shape.
//
// This has no effect on the data, no transposition is performed.
func (fn *Function) Reshape(operand *Value, shape shapes.Shape) (*Value, error) {
	op := optypes.Reshape
	if operand.shape.DType != shape.DType {
		return nil, errors.Errorf("Reshape() requires the operand and the shape to have the same data type, got operand=%s and shape=%s",
			operand.shape, shape)
	}
	if operand.shape.Size() != shape.Size() {
		return nil, errors.Errorf("Reshape() requires the total size of the new shape to match the original shape, got operand=%s and shape=%s",
			operand.shape, shape)
	}
	stmt := fn.addOp(op, shape, operand)
	return stmt.Outputs[0], nil
}
