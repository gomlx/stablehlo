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

// IsFinite tests whether each element of operand is finite, i.e., is not positive or negative infinity, and is not NaN.
// It returns an array of boolean values with the same shape as the input, where each element is true if and only if
// the corresponding input element is finite.
func (fn *Function) IsFinite(x *Value) (*Value, error) {
	op := optypes.IsFinite
	outputShape, err := shapeinference.IsFinite(x.shape)
	if err != nil {
		return nil, err
	}
	return fn.addOp(op, outputShape, x).Outputs[0], nil
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
	precisionConfig := fmt.Sprintf("[#stablehlo<precision %s>, #stablehlo<precision %s>]",
		b.precision[0].ToStableHLO(), b.precision[1].ToStableHLO())
	stmt.Attributes["precision_config"] = literalStr(precisionConfig)
	if b.algorithm != nil {
		algorithmConfig := fmt.Sprintf(`#stablehlo.dot_algorithm<
      lhs_precision_type = %s,
      rhs_precision_type = %s,
      accumulation_type = %s,
      lhs_component_count = %d,
      rhs_component_count = %d,
      num_primitive_operations = %d,
      allow_imprecise_accumulation = %v
    >`, b.algorithm.LhsPrecisionType.ToStableHLO(),
			b.algorithm.RhsPrecisionType.ToStableHLO(),
			b.algorithm.AccumulationType.ToStableHLO(),
			b.algorithm.LhsComponentCount,
			b.algorithm.RhsComponentCount,
			b.algorithm.NumPrimitiveOperations,
			b.algorithm.AllowImpreciseAccumulation)
		stmt.Attributes["algorithm"] = literalStr(algorithmConfig)
	}
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

// BroadcastInDim broadcasts dimensions from the operand to the target shape.
// It can also transpose axes and add new ones.
//
// The axesMapping should have one value per operand axes. It maps the axes from the operand to
// the corresponding value on the target shape.
func (fn *Function) BroadcastInDim(operand *Value, target shapes.Shape, axesMapping []int) (*Value, error) {
	op := optypes.BroadcastInDim
	err := shapeinference.BroadcastInDim(operand.shape, target, axesMapping)
	if err != nil {
		return nil, err
	}
	stmt := fn.addOp(op, target, operand)
	stmt.Attributes = map[string]any{"broadcast_dimensions": intSliceToArrayI64StableHLO(axesMapping)}
	return stmt.Outputs[0], nil
}

// Gather is a powerful but cumbersome Gather operation.
// Full details in https://openxla.org/stablehlo/spec#gather.
//
// The output of Gather has the same DType of the operand, from where we are pulling the data.
//
// Its output shape will be composed of 2 parts:
//
//   - Batch axes: they come from operandBatchingAxes/startIndicesBatchingAxes (they correspond to each other)
//     and from the other axes of startIndices, except the "indexVectorAxis" (usually the last)
//     that is used as the indices into the operand. (*)
//   - "Offset axes": these are axes that come from the operand, the sizes given by sliceSizes.
//     Notice that if sliceSizes for an axis is 1, and that axis is present in the collapsedSliceAxes list, this
//     axis gets omitted in the output.
//
// So in general output.Rank() = startIndices.Rank() - 1 + len(offsetAxes).
//
// (*) One exception is if indexVectorAxis == startIndices.Rank(), in which case we assume there is an
// extra virtual axis in startIndices of size 1, in which case output.Rank() = startIndices.Rank() + len(offsetAxes).
//
// (*) One exception is if indexVectorAxis == startIndices.Rank(), in which case we assume there is an
// extra implicit axis in startIndices of size 1, in which case output.Rank() = startIndices.Rank() + len(offsetAxes).
//
// Arguments:
//   - operand: the values from where we are gathering. The output DType will follow the operand one.
//   - startIndices: are the indices we want to gather. The axis pointed by indexVector
//     lists the indices of the slice to be gathered in the operand array (their values are mapped to the axis
//     in the operand according to startIndexMap).
//     All other axes are "batch dimensions" and they will have equivalent axes (same dimensions) in the output.
//   - indexVectorAxis: which of the axis in startIndices is collected and used as the start index for slices
//     to be gathered in the operand.
//     It is typically the last axis of startIndices, so startIndices.Shape.Rank()-1.
//     There is a special case where indexVectorAxis == startIndices.Rank() in which case we assume there is an
//     extra virtual axis in startIndices of size 1, in which case output.Rank() = startIndices.Rank() + len(offsetAxes).
//   - offsetOutputAxes: _output_ axes (not the operand's) that will hold the "offset slices", slices that are not
//     collapsed. It points in which position (axis) in the output these slices should show up.
//     The len(offsetOutputAxes) must match the dimension of indexVectorAxis (== startIndices.Dimensions[indexVectorAxis]).
//     Notice all axes in the operand will either become an "offset axis" in the output,
//     of optionally collapsed (or "squeezed") in the output, if included in collapsedSliceAxes.
//     The axes in the output (given in offsetAxes) to the axes in the operand (the axes not present in collapsedSliceAxes) sequentially.
//     One must have Rank(operand) == len(collapsedSliceAxes) + len(offsetAxes) + len(operandBatchingAxes).
//   - collapsedSliceAxes: _operand_ axes (for which sliceSizes are 1) not to be included in the output.
//     One must have sliceSizes[collapsedSliceAxes[i]] == 1 for all i.
//   - operandBatchingAxes: operand's batching axes that have corresponding batching axes in the startIndices, and that
//     will also be included in the output.
//     One must have sliceSizes[operandBatchingAxes[i]] == 1 for all i.
//     Also, one must have Rank(operand) == len(operandBatchingAxes) + len(collapsedSliceAxes) + len(offsetOutputAxes).
//   - startIndicesBatchingAxes: startIndices' batching axes have corresponding batching axes in the operand, and that
//     will also be included in the output.
//   - startIndexMap: this maps which value in startIndices is used for which axis in the operand, select the slice to be gathered.
//     Notice len(startIndexMap) must match the startIndices.Dimensions[indexVectorAxis].
//     Also, len(startIndexMap) == len(offsetOutputAxes) -- offsetOutputAxes maps the same axes in the output.
//     E.g.: if startIndices.shape=(2, 3), indexVectorAxis=1, and operand.rank=4 and startIndexMap=[]int{0, 1, 2},
//     this means each row of the startIndices will point to the first 3 axes (0,1 and 2) in the operand.
//     For those axes in the operand not explicitly set (so if len(startIndexMap) < operand.Rank()), and not part
//     of operandBatchingAxes, the corresponding axis start index is considered to be 0, and one sets the sliceSizes
//     to take the slice one wants (typically the full slice).
//   - sliceSizes: a size for each operand's axis, so len(sliceSize) = operand.Rank().
//     once the start index from where to gather is resolved, this defines how much data in each axis
//     to gather.
//     Constraints: sliceSizes[collapsedSliceAxes[i]] == 1, and sliceSizes[operandBatchingAxes[j]] == 1, for all i, j.
//   - indicesAreSorted: can be set to true if it's guaranteed that startIndices are sorted (in ascending order,
//     after scattering its values according to start_index_map) by the user. This allows for some optimizations
//     in some platforms.
func (fn *Function) Gather(operand, startIndices *Value, indexVectorAxis int,
	offsetOutputAxes, collapsedSliceAxes, operandBatchingAxes,
	startIndicesBatchingAxes, startIndexMap,
	sliceSizes []int, indicesAreSorted bool) (*Value, error) {
	op := optypes.Gather
	outputShape, err := shapeinference.Gather(
		operand.shape, startIndices.shape, indexVectorAxis,
		offsetOutputAxes, collapsedSliceAxes, operandBatchingAxes,
		startIndicesBatchingAxes, startIndexMap,
		sliceSizes, indicesAreSorted)
	if err != nil {
		return nil, err
	}
	stmt := fn.addOp(op, outputShape, operand, startIndices)
	stmt.Attributes = map[string]any{
		"dimension_numbers": literalStrF(
			`#stablehlo.gather<
      offset_dims = %s,
      collapsed_slice_dims = %s,
      operand_batching_dims = %s,
      start_indices_batching_dims = %s,
      start_index_map = %s,
      index_vector_dim = %d>`,
			intSliceToStableHLO(offsetOutputAxes),
			intSliceToStableHLO(collapsedSliceAxes),
			intSliceToStableHLO(operandBatchingAxes),
			intSliceToStableHLO(startIndicesBatchingAxes),
			intSliceToStableHLO(startIndexMap),
			indexVectorAxis),
		"slice_sizes":        intSliceToArrayI64StableHLO(sliceSizes),
		"indices_are_sorted": indicesAreSorted,
	}
	return stmt.Outputs[0], nil
}
