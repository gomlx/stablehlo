package stablehlo

import (
	"fmt"
	"slices"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/stablehlo/internal/optypes"
	"github.com/gomlx/stablehlo/shapeinference"
	"github.com/gomlx/stablehlo/types"
	"github.com/gomlx/stablehlo/types/shapes"
	"github.com/pkg/errors"
)

// addOp adds a new operation to the function.
func (fn *Function) addOp(opType optypes.OpType, outputShape shapes.Shape, inputs ...*Value) *Statement {
	stmt := &Statement{
		Builder:  fn.Builder,
		Function: fn,
		OpType:   opType,
		Inputs:   inputs,
		Outputs:  []*Value{fn.newValue(outputShape)},
	}
	fn.Statements = append(fn.Statements, stmt)
	return stmt
}

// addMultiOp adds a new operation with multiple outputs to the function.
func (fn *Function) addMultiOp(opType optypes.OpType, outputShapes []shapes.Shape, inputs []*Value) *Statement {
	outputs := make([]*Value, len(outputShapes))
	for i, shape := range outputShapes {
		outputs[i] = fn.newValue(shape)
	}
	stmt := &Statement{
		Builder:  fn.Builder,
		Function: fn,
		OpType:   opType,
		Inputs:   inputs,
		Outputs:  outputs,
	}
	fn.Statements = append(fn.Statements, stmt)
	return stmt
}

// binaryOp adds a new binary operation to the function.
func (fn *Function) binaryOp(op optypes.OpType, lhs, rhs *Value) (*Value, error) {
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	if lhs.fn != fn || rhs.fn != fn {
		return nil, errors.Errorf("cannot add operation %s to function %q, because the operands are not part of the function",
			op, fn.Name)
	}
	outputShape, err := shapeinference.BinaryOp(op, lhs.shape, rhs.shape)
	if err != nil {
		return nil, err
	}
	return fn.addOp(op, outputShape, lhs, rhs).Outputs[0], nil
}

// unaryOp adds a new unary operation to the function.
func (fn *Function) unaryOp(op optypes.OpType, operand *Value) (*Value, error) {
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	if operand.fn != fn {
		return nil, errors.Errorf("cannot add operation %s to function %q, because the operand is not part of the function",
			op, fn.Name)
	}
	outputShape, err := shapeinference.UnaryOp(op, operand.shape)
	if err != nil {
		return nil, err
	}
	return fn.addOp(op, outputShape, operand).Outputs[0], nil
}

// Compare implements the corresponding standard binary operation.
func Compare(lhs, rhs *Value, direction types.ComparisonDirection, compareType types.ComparisonType) (*Value, error) {
	op := optypes.Compare
	fn := lhs.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	if rhs.fn != fn {
		return nil, errors.Errorf("cannot add operation %s to function %q, because operands are from different functions (%q and %q)",
			op, fn.Name, fn.Name, rhs.fn.Name)
	}
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

func valuesToShapes(values []*Value) []shapes.Shape {
	s := make([]shapes.Shape, len(values))
	for i, v := range values {
		s[i] = v.shape
	}
	return s
}

// Complex returns the complex value by concatenating the real and imaginary parts element-wise.
func Complex(real, imag *Value) (*Value, error) {
	op := optypes.Complex
	fn := real.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	if imag.fn != fn {
		return nil, errors.Errorf("cannot add operation %s to function %q, because operands are from different functions (%q and %q)",
			op, fn.Name, fn.Name, imag.fn.Name)
	}
	outputShape, err := shapeinference.Complex(real.shape, imag.shape)
	if err != nil {
		return nil, err
	}
	return fn.addOp(op, outputShape, real, imag).Outputs[0], nil
}

// Real returns the real part of the complex value.
func Real(complex *Value) (*Value, error) {
	op := optypes.Real
	fn := complex.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	outputShape, err := shapeinference.RealOrImag(complex.shape)
	if err != nil {
		return nil, err
	}
	return fn.addOp(op, outputShape, complex).Outputs[0], nil
}

// Imag returns the real part of the complex value.
func Imag(complex *Value) (*Value, error) {
	op := optypes.Imag
	fn := complex.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	outputShape, err := shapeinference.RealOrImag(complex.shape)
	if err != nil {
		return nil, err
	}
	return fn.addOp(op, outputShape, complex).Outputs[0], nil
}

// IsFinite tests whether each element of operand is finite, i.e., if it is not positive nor negative infinity, and it is not NaN.
// It returns the same shape as the input, but with boolean values where each element is true if and only if
// the corresponding input element is finite.
func IsFinite(x *Value) (*Value, error) {
	op := optypes.IsFinite
	fn := x.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
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
func Clamp(min, x, max *Value) (*Value, error) {
	op := optypes.Clamp
	fn := x.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	if min.fn != fn || max.fn != fn {
		return nil, errors.Errorf("cannot add operation %s to function %q, because operands are from different functions (%q, %q and %q)",
			op, fn.Name, fn.Name, max.fn.Name, min.fn.Name)
	}
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
func DotGeneral(
	lhsOp *Value, lhsContractingAxes, lhsBatchAxes []int,
	rhsOp *Value, rhsContractingAxes, rhsBatchAxes []int) *DotGeneralBuilder {
	return &DotGeneralBuilder{
		fn:                 lhsOp.fn,
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
	fn := b.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	if b.lhs.fn != fn || b.rhs.fn != fn {
		return nil, errors.Errorf("cannot add operation %s to function %q, because operands are from different functions (%q and %q)",
			op, fn.Name, b.lhs.fn.Name, b.rhs.fn.Name)
	}
	outputShape, err := shapeinference.DotGeneral(
		b.lhs.shape, b.lhsContractingAxes, b.lhsBatchAxes,
		b.rhs.shape, b.rhsContractingAxes, b.rhsBatchAxes,
		b.outputDType)
	if err != nil {
		return nil, err
	}
	stmt := b.fn.addOp(op, outputShape, b.lhs, b.rhs)
	stmt.Attributes = map[string]any{
		"dot_dimension_numbers": literalStrF(
			"#stablehlo.dot<\n"+
				"\tlhs_batching_dimensions = %s,\n"+
				"\trhs_batching_dimensions = %s,\n"+
				"\tlhs_contracting_dimensions = %s,\n"+
				"\trhs_contracting_dimensions = %s\n>",
			intSliceToStableHLO(b.lhsBatchAxes),
			intSliceToStableHLO(b.rhsBatchAxes),
			intSliceToStableHLO(b.lhsContractingAxes),
			intSliceToStableHLO(b.rhsContractingAxes)),
	}
	precisionConfig := fmt.Sprintf("[#stablehlo<precision %s>, #stablehlo<precision %s>]",
		b.precision[0].ToStableHLO(), b.precision[1].ToStableHLO())
	stmt.Attributes["precision_config"] = literalStr(precisionConfig)
	if b.algorithm != nil {
		stmt.Attributes["algorithm"] = literalStrF("#stablehlo.dot_algorithm<\n"+
			"\tlhs_precision_type = %s,\n"+
			"\trhs_precision_type = %s,\n"+
			"\taccumulation_type = %s,\n"+
			"\tlhs_component_count = %d,\n"+
			"\trhs_component_count = %d,\n"+
			"\tnum_primitive_operations = %d,\n"+
			"\tallow_imprecise_accumulation = %v>",
			b.algorithm.LhsPrecisionType.ToStableHLO(),
			b.algorithm.RhsPrecisionType.ToStableHLO(),
			b.algorithm.AccumulationType.ToStableHLO(),
			b.algorithm.LhsComponentCount,
			b.algorithm.RhsComponentCount,
			b.algorithm.NumPrimitiveOperations,
			b.algorithm.AllowImpreciseAccumulation)
	}
	return stmt.Outputs[0], nil
}

// Reshape the operand to the given shape.
// The total size of the new shape must match the original shape.
//
// This has no effect on the data, no transposition is performed.
func Reshape(operand *Value, shape shapes.Shape) (*Value, error) {
	op := optypes.Reshape
	fn := operand.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
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
func BroadcastInDim(operand *Value, target shapes.Shape, axesMapping []int) (*Value, error) {
	op := optypes.BroadcastInDim
	fn := operand.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
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
func Gather(operand, startIndices *Value, indexVectorAxis int,
	offsetOutputAxes, collapsedSliceAxes, operandBatchingAxes,
	startIndicesBatchingAxes, startIndexMap,
	sliceSizes []int, indicesAreSorted bool) (*Value, error) {
	op := optypes.Gather
	fn := operand.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	if startIndices.fn != fn {
		return nil, errors.Errorf("cannot add operation %s to function %q, because startIndices is from different function (%q and %q)",
			op, fn.Name, startIndices.fn.Name, fn.Name)
	}

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
			"#stablehlo.gather<\n"+
				"\toffset_dims = %s,\n"+
				"\tcollapsed_slice_dims = %s,\n"+
				"\toperand_batching_dims = %s,\n"+
				"\tstart_indices_batching_dims = %s,\n"+
				"\tstart_index_map = %s,\n"+
				"\tindex_vector_dim = %d>",
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

// Slice extracts a subarray from the input array.
// The subarray is of the same rank as the input and contains the values inside a bounding box within the input array
// where the dimensions and indices of the bounding box are given as arguments to the slice operation.
// The strides set the input stride of the slice in each axis and must be >= 1.
// It is optional, and if missing, it is assumed to be 1 for every dimension.
// Examples:
//
//	Slice(x={0, 1, 2, 3, 4}, starts={2}, limits={4}, strides=nil) -> {2, 3}
//	Slice(x={0, 1, 2, 3, 4}, starts={2}, limits={5}, strides={2}) -> {2, 4}
func Slice(x *Value, starts, limits, strides []int) (*Value, error) {
	op := optypes.Slice
	fn := x.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	if len(strides) == 0 {
		strides = make([]int, x.shape.Rank())
		for i := range strides {
			strides[i] = 1
		}
	}
	outputShape, err := shapeinference.Slice(x.shape, starts, limits, strides)
	if err != nil {
		return nil, err
	}
	stmt := fn.addOp(op, outputShape, x)
	stmt.Attributes = map[string]any{
		"start_indices": intSliceToArrayI64StableHLO(starts),
		"limit_indices": intSliceToArrayI64StableHLO(limits),
		"strides":       intSliceToArrayI64StableHLO(strides),
	}
	return stmt.Outputs[0], nil
}

// Concatenate operands on the given axis.
//
// All axes that are not being concatenated must match dimensions, except on the axes being concatenated.
// It doesn't work with scalars -- use ExpandAxes.
// If there is only one operand, it is returned and this is a no-op.
func Concatenate(axis int, operands ...*Value) (*Value, error) {
	op := optypes.Concatenate
	if len(operands) == 0 {
		return nil, errors.New("Concatenate requires at least one operand")
	}
	fn := operands[0].fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	for i, operand := range operands {
		if operand.fn != fn {
			return nil, errors.Errorf("cannot add operation %s to function %q, because operand #%d is from different function (%q and %q)",
				op, fn.Name, i, operand.fn.Name, fn.Name)
		}
	}
	operandsShapes := make([]shapes.Shape, len(operands))
	for i, operand := range operands {
		operandsShapes[i] = operand.shape
	}
	outputShape, err := shapeinference.Concatenate(operandsShapes, axis)
	if err != nil {
		return nil, err
	}
	adjustedAxis, err := shapeinference.AdjustAxisToRank(axis, operands[0].shape.Rank())
	if err != nil {
		return nil, errors.WithMessage(err, "Concatenate axis for operands")
	}
	stmt := fn.addOp(op, outputShape, operands...)
	stmt.Attributes = map[string]any{
		"dimension": int64(adjustedAxis),
	}
	return stmt.Outputs[0], nil
}

// Reduce reduces the input along the given axes.
//
// Each resulting value is initialized with initValue (e.g.: for a sum, it's 0, for a product it's 1), and
// then each value is combined with it using the reduction function.
//
// The reduction function must be created with Builder.NewClosure, and it should take as input scalar
// values be associative and commutative.
//
// The initialValue and x must have the same DType. This initial dtype must be promotable to the dtype accepted
// by the reductions function. The result dtype is the same as the output of the reduction function.
// So one could reduce-sum a 4bit quantized tensor directly into a Float32.
//
// See MultiReduce for a version that accepts multiple inputs and outputs.
func Reduce(x, initialValue *Value, reduction *Function, axes ...int) (*Value, error) {
	results, err := MultiReduce([]*Value{x}, []*Value{initialValue}, reduction, axes...)
	if err != nil {
		return nil, err
	}
	return results[0], nil
}

// MultiReduce reduces the input along the given axes.
//
// Each resulting value i is initialized with initValues[i] (e.g.: for a sum, it's 0, for a product it is 1),
// and then each value is combined with it using the reduction function.
//
// The reduction function must be created with Builder.NewClosure.
// If there are N inputs and initialValues, the reduction function should have a signature
// (lhs_1, ... lhs_N, rhs_1, ... lhs_N) and output (out_1 ... out_N), where lhs_i and rhs_i are scalars
// taken from the inputs.
//
// It returns N results for each aggregated value.
//
// See Reduce for a version that accepts a single input.
//
// TODO: promotion of types doesn't seem to be working according to the spec in
// https://openxla.org/stablehlo/spec#reduce.
func MultiReduce(inputs, initialValues []*Value, reduction *Function, axes ...int) ([]*Value, error) {
	op := optypes.Reduce
	if len(inputs) == 0 {
		return nil, errors.New("MultiReduce requires at least one operand")
	}
	fn := inputs[0].fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	for i, operand := range inputs {
		if operand.fn != fn {
			return nil, errors.Errorf("cannot add operation %s to function %q, because input #%d is from different function (%q and %q)",
				op, fn.Name, i, operand.fn.Name, fn.Name)
		}
	}
	for i, operand := range initialValues {
		if operand.fn != fn {
			return nil, errors.Errorf("cannot add operation %s to function %q, because initialValues[%d] is from different function (%q and %q)",
				op, fn.Name, i, operand.fn.Name, fn.Name)
		}
	}
	outputsShapes, err := shapeinference.Reduce(
		valuesToShapes(inputs), valuesToShapes(initialValues),
		valuesToShapes(reduction.Inputs), reduction.Outputs,
		axes)
	if err != nil {
		return nil, err
	}
	allInputs := append(slices.Clone(inputs), initialValues...)
	stmt := fn.addMultiOp(op, outputsShapes, allInputs)
	stmt.Attributes = map[string]any{
		"dimensions": intSliceToArrayI64StableHLO(axes),
	}
	stmt.AddFunctionParameter("reduction", reduction)
	return stmt.Outputs, nil
}

// Select takes element-wise values from onTrue or onFalse depending on the value of the pred (must be boolean).
//
// The pred must be boolean and can be a scalar or have the same shape as isTrue and isFalse.
// isTrue and isFalse must have the same shape and dtypes.
func Select(pred, onTrue, onFalse *Value) (*Value, error) {
	op := optypes.Select
	fn := pred.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	if onTrue.fn != fn || onFalse.fn != fn {
		return nil, errors.Errorf("cannot add operation %s to function %q, because operands are from different functions (%q, %q and %q)",
			op, fn.Name, fn.Name, onTrue.fn.Name, onFalse.fn.Name)
	}
	outputShape, err := shapeinference.Select(pred.shape, onTrue.shape, onFalse.shape)
	if err != nil {
		return nil, err
	}
	stmt := fn.addOp(op, outputShape, pred, onTrue, onFalse)
	return stmt.Outputs[0], nil
}

// BitcastConvert performs an elementwise bit-cast operation from a dtype to another dtype.
//
// The Bitcast doesn't "convert", rather it just reinterprets the bits from x.DType() to the targetDType.
//
// If x.DType() and targetDType use the same number of bytes (targetDType.Size() == x.DType().Size()),
// the dimensions are not changed, simply the dtype is changed.
//
// If targetDType.Size() > x.DType().Size(), it requires x last axis to have a dimension of
// targetDType.Size() / x.DType().Size(), and the returned shape will trim the last axis.
//
// If targetDType.Size() < x.DType().Size(), the returned shape will have an extra axis in the end, with dimension of
// x.DType().Size() / targetDType.Size().
//
// E.g: Bitcast([1]uint32{0xdeadbeef}, dtypes.UInt16) -> [1][2]uint16{{0xbeef, 0xdead}} // Little-endian encoding.
func BitcastConvert(operand *Value, targetDtype dtypes.DType) (*Value, error) {
	op := optypes.BitcastConvert
	fn := operand.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	outputShape, err := shapeinference.BitcastConvert(operand.shape, targetDtype)
	if err != nil {
		return nil, err
	}
	stmt := fn.addOp(op, outputShape, operand)
	return stmt.Outputs[0], nil
}
