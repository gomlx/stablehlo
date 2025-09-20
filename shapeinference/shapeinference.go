// Package shapeinference calculates the shape resulting from operations and validates its inputs.
//
// This can be useful for new optypes.to test and help plan for buffer space for temporary or output buffers.
//
// It defines a BinaryOp function for shape inference for the majority of binary functions, using the standard
// broadcasting rules.
//
// The majority of the unary functions don't change the shape, except those that explicitly say that in their name,
// like Reshape, etc.
//
// For the remainder operations, each one gets its own shape inference function.
package shapeinference

import (
	"slices"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/stablehlo/internal/optypes"
	"github.com/gomlx/stablehlo/internal/utils"
	"github.com/gomlx/stablehlo/types"
	"github.com/gomlx/stablehlo/types/shapes"
	"github.com/pkg/errors"
)

var (
	// BooleanOrBitwiseOperations take booleans as input, aka. logical operations.
	BooleanOrBitwiseOperations = utils.SetWith(
		optypes.And,
		optypes.Or,
		optypes.Xor,
		optypes.Not,
	)

	// BitwiseOperations operates only on integer (binary) numbers and won't work on floats or complex numbers.
	BitwiseOperations = utils.SetWith(
		optypes.Popcnt,
		optypes.ShiftLeft,
		optypes.ShiftRightArithmetic,
		optypes.ShiftRightLogical,
		optypes.CountLeadingZeros,
	)

	// NumberOperations can take any type of number as input: integers, floats, or complex numbers.
	NumberOperations = utils.SetWith(
		optypes.Add,
		optypes.Subtract,
		optypes.Multiply,
		optypes.Divide,
		optypes.Power,
		optypes.Remainder,

		// Notice Abs and Sign works for unsigned ints: it's just a trivial implementation.
		optypes.Abs,
		optypes.Sign,

		optypes.Compare,
	)

	SignedNumberOperations = utils.SetWith(
		optypes.Negate,
	)

	// FloatOperations operates only on float (and not on complex numbers).
	FloatOperations = utils.SetWith(
		optypes.Erf,
		optypes.Logistic,
		optypes.Cosine,
		optypes.Sine,
		optypes.Tanh,
	)

	// FloatOrComplexOperations operates only on float or complex numbers and won't work on integer or boolean values.
	FloatOrComplexOperations = utils.SetWith(
		optypes.Exponential,
		optypes.ExponentialMinusOne,
		optypes.Log,
		optypes.LogPlusOne,
		optypes.Ceil,
		optypes.Floor,
		optypes.RoundNearestEven,
		optypes.Rsqrt,
		optypes.Sqrt,
		optypes.IsFinite,
	)

	// ComplexOperations operates only on complex numbers.
	ComplexOperations = utils.SetWith(
		optypes.Imag,
		optypes.Real,
	)

	// StandardBinaryOperations include all operations that have two operands usually named lhs (left-hand-side) and
	// rhs (right-hand-side) and are usually commutative (invariant to order).
	StandardBinaryOperations = utils.SetWith(
		optypes.Add,
		optypes.Atan2,
		optypes.Subtract,
		optypes.Multiply,
		optypes.Divide,
		optypes.Power,
		optypes.Remainder,
		optypes.And,
		optypes.Or,
		optypes.Xor,
		optypes.Maximum,
		optypes.Minimum,
		optypes.ShiftLeft,
		optypes.ShiftRightArithmetic,
		optypes.ShiftRightLogical,
	)

	// ComparisonOperations include all operations that take two inputs and returns booleans with the results of
	// a comparison.
	// For StableHLO they are converged in only one optypes.Compare, that takes as an attribute the comparison type.
	ComparisonOperations = utils.SetWith(optypes.Compare)

	// StandardUnaryOperations include all operations that have a single operand as input, and the return shape is the
	// same as the input (so no reductions).
	StandardUnaryOperations = utils.SetWith(
		optypes.Not,
		optypes.Popcnt,
		optypes.Cbrt,
		optypes.CountLeadingZeros,
		optypes.Erf,
		optypes.Exponential,
		optypes.ExponentialMinusOne,
		optypes.Log,
		optypes.LogPlusOne,
		optypes.Logistic,
		optypes.Ceil,
		optypes.Floor,
		optypes.RoundNearestEven,
		optypes.RoundNearestAfz,
		optypes.Rsqrt,
		optypes.Sqrt,
		optypes.Cosine,
		optypes.Sine,
		optypes.Tan,
		optypes.Tanh,
		optypes.Abs,
		optypes.Negate,
		optypes.Sign,
	)
)

// BinaryOp returns the expected output shape for ops in the StandardBinaryOperations set -- those include all
// operations that have two operands usually named lhs (left-hand-side) and rhs (right-hand-side), and they are usually
// commutative (invariant to order).
//
// It returns an error if the data type (shape.DType) is invalid for the operation -- e.g.: non-matching
// dtypes, or LogicalAnd not having booleans (dtype.Bool) as input.
func BinaryOp(opType optypes.OpType, lhsShape, rhsShape shapes.Shape) (output shapes.Shape, err error) {
	if !StandardBinaryOperations.Has(opType) && !ComparisonOperations.Has(opType) {
		err = errors.Errorf("operations %s is not in the StandardBinaryOperations set, cannot process it with BinaryOp", opType)
		return
	}
	if lhsShape.DType == dtypes.InvalidDType || rhsShape.DType == dtypes.InvalidDType {
		err = errors.Errorf("invalid shape for %s or %s for %q", lhsShape, rhsShape, opType)
		return
	}
	if !lhsShape.Equal(rhsShape) {
		err = errors.Errorf("shapes for %q must match, got %s and %s", opType, lhsShape, rhsShape)
		return
	}
	if BooleanOrBitwiseOperations.Has(opType) && lhsShape.DType != dtypes.Bool && !lhsShape.DType.IsInt() {
		err = errors.Errorf("Logical/Bitwise %q must have boolean (dtype.Bool) data types as input, got %s", opType, lhsShape)
		return
	}
	if BitwiseOperations.Has(opType) && !lhsShape.DType.IsInt() {
		err = errors.Errorf("bitwise BinaryOp %s must have an integer (Int8, UInt8, Int32, ...) data type as input, got %s", opType, lhsShape)
		return
	}
	if NumberOperations.Has(opType) && !(lhsShape.DType.IsInt() || lhsShape.DType.IsFloat() || lhsShape.DType.IsComplex()) {
		err = errors.Errorf("numeric BinaryOp %s must have a number (Int32, Float32, Complex64, ...) data type as input, got %s", opType, lhsShape)
		return
	}
	if FloatOperations.Has(opType) && !lhsShape.DType.IsFloat() {
		err = errors.Errorf("float BinaryOp %s must have a float (Float32, Float64, ...) data type as input, got %s", opType, lhsShape)
		return
	}
	if FloatOrComplexOperations.Has(opType) && !(lhsShape.DType.IsFloat() || lhsShape.DType.IsComplex()) {
		err = errors.Errorf("float/complex BinaryOp %s must have a float or complex (Float32, Complex64, ...) data type as input, got %s", opType, lhsShape)
		return
	}
	if ComplexOperations.Has(opType) && !lhsShape.DType.IsComplex() {
		err = errors.Errorf("complex BinaryOp %s must have a complex (Complex64, Complex128) data type as input, got %s", opType, lhsShape)
		return
	}

	return binaryOpImpl(opType, lhsShape, rhsShape)
}

func binaryOpImpl(opType optypes.OpType, lhsShape, rhsShape shapes.Shape) (output shapes.Shape, err error) {
	// Trivial cases: if one of the sides is a scalar, return the other side shape.
	if lhsShape.IsScalar() {
		return rhsShape, nil
	}
	if rhsShape.IsScalar() {
		return lhsShape, nil
	}

	// Other cases, either the dimensions match or one of them is 1.
	if lhsShape.Rank() != rhsShape.Rank() {
		err = errors.Errorf("if operands are not scalars, their rank must match for BinaryOp (%s), got shapes %s and %s",
			opType, lhsShape, rhsShape)
	}
	output = lhsShape.Clone()
	for axis := range output.Rank() {
		lhsDim := lhsShape.Dimensions[axis]
		rhsDim := rhsShape.Dimensions[axis]
		if lhsDim != 1 && rhsDim != 1 && lhsDim != rhsDim {
			err = errors.Errorf("dimension of axis #%d doesn't match and cannot be broadcast for BinaryOp (%s), got shapes %s and %s",
				axis, opType, lhsShape, rhsShape)
			return
		}
		output.Dimensions[axis] = max(lhsDim, rhsDim)
	}
	return
}

// Compare returns the broadcast shape with dtype set to Bool, for comparison operations (Equal, LessThan, GreaterOrEqual, etc.)
func Compare(lhsShape, rhsShape shapes.Shape, direction types.ComparisonDirection, compareType types.ComparisonType) (output shapes.Shape, err error) {
	if lhsShape.DType == dtypes.InvalidDType || rhsShape.DType == dtypes.InvalidDType {
		err = errors.Errorf("invalid shape for %s or %s for Compare", lhsShape, rhsShape)
		return
	}
	if lhsShape.DType != rhsShape.DType {
		err = errors.Errorf("data types (DType) for Compare must match, got %s and %s", lhsShape, rhsShape)
		return
	}
	dtype := lhsShape.DType
	switch compareType {
	case types.CompareFloat:
		if !dtype.IsFloat() && !dtype.IsComplex() {
			err = errors.Errorf("data type %s is not a float or complex, cannot process it with Compare(type=FLOAT)", dtype)
			return
		}
	case types.CompareTotalOrder:
		if !dtype.IsFloat() {
			err = errors.Errorf("data type %s is not a float, cannot process it with Compare(type=TOTAL_ORDER)", dtype)
			return
		}
	case types.CompareSigned:
		if !dtype.IsInt() || dtype.IsUnsigned() {
			err = errors.Errorf("data type %s is not a signed integer, cannot process it with Compare(type=SIGNED)", dtype)
			return
		}
	case types.CompareUnsigned:
		if !dtype.IsInt() || !dtype.IsUnsigned() {
			err = errors.Errorf("data type %s is not a signed integer, cannot process it with Compare(type=UNSIGNED)", dtype)
			return
		}
	default:
		err = errors.Errorf("invalid comparison type %d for Compare", compareType)
		return
	}
	if direction < types.CompareEQ || direction > types.CompareNE {
		err = errors.Errorf("invalid comparison direction %d for Compare", direction)
		return
	}
	output, err = BinaryOp(optypes.Compare, lhsShape, rhsShape)
	if err != nil {
		return
	}
	output.DType = dtypes.Bool
	return
}

// UnaryOp checks the validity of the data type for StandardUnaryOperations and returns either an error or
// the output shape, which is the same as the operand.
func UnaryOp(opType optypes.OpType, operand shapes.Shape) (output shapes.Shape, err error) {
	if !StandardUnaryOperations.Has(opType) {
		err = errors.Errorf("operation %s is not in the StandardUnaryOperations set, cannot process it with UnaryOp", opType)
		return
	}
	if operand.DType == dtypes.InvalidDType {
		err = errors.Errorf("invalid shape %s for UnaryOp %s", operand, opType)
		return
	}
	if BooleanOrBitwiseOperations.Has(opType) && operand.DType != dtypes.Bool && !operand.DType.IsInt() {
		err = errors.Errorf("logical UnaryOp %q must have boolean (dtype.Bool) data types as input, got %s", opType, operand)
		return
	}
	if BitwiseOperations.Has(opType) && !operand.DType.IsInt() {
		err = errors.Errorf("bitwise UnaryOp %s must have an integer (Int8, UInt8, Int32, ...) data type as input, got %s", opType, operand)
		return
	}
	if SignedNumberOperations.Has(opType) && (operand.DType.IsUnsigned() ||
		!(operand.DType.IsInt() || operand.DType.IsFloat() || operand.DType.IsComplex())) {
		err = errors.Errorf("signed UnaryOp %s must have a signed data type as input, got %s", opType, operand)
		return
	}
	if NumberOperations.Has(opType) && !(operand.DType.IsInt() || operand.DType.IsFloat() || operand.DType.IsComplex()) {
		err = errors.Errorf("numeric UnaryOp %s must have a number (Int32, Float32, Complex64, ...) data type as input, got %s", opType, operand)
		return
	}
	if FloatOperations.Has(opType) && !operand.DType.IsFloat() {
		err = errors.Errorf("float UnaryOp %s must have a float (Float32, Float64, ...) data type as input, got %s", opType, operand)
		return
	}
	if FloatOrComplexOperations.Has(opType) && !(operand.DType.IsFloat() || operand.DType.IsComplex()) {
		err = errors.Errorf("float/complex UnaryOp %s must have a float or complex (Float32, Complex64, ...) data type as input, got %s", opType, operand)
		return
	}
	if ComplexOperations.Has(opType) && !operand.DType.IsComplex() {
		err = errors.Errorf("complex UnaryOp %s must have a complex (Complex64, Complex128) data type as input, got %s", opType, operand)
		return
	}
	output = operand
	return
}

// Select returns the shape resulting from the Select operation.
//
// The pred must be boolean and can be a scalar or have the same shape as isTrue and isFalse.
// isTrue and isFalse must have the same shape and dtypes.
func Select(pred, onTrue, onFalse shapes.Shape) (output shapes.Shape, err error) {
	if pred.DType != dtypes.Bool {
		err = errors.Errorf("pred for Select() must be a boolean, got %s instead", pred)
		return
	}
	if !onTrue.Equal(onFalse) {
		err = errors.Errorf("onTrue (%s) and onFalse (%s) values for Select() must have the same shape",
			onTrue, onFalse)
		return
	}
	if !pred.IsScalar() && pred.CheckDims(onTrue.Dimensions...) != nil {
		err = errors.Errorf("pred for Select() must either be a scalar or match onTrue and onFalse shapes, instead got shapes pred=%s, onTrue=%s and onFalse=%s",
			pred, onTrue, onFalse)
	}
	if !onTrue.IsScalar() && !onFalse.IsScalar() && !onTrue.Equal(onFalse) {
		err = errors.Errorf("onTrue (%s) and onFalse (%s) values for Select() must either be scalar or match each other's shape",
			onTrue, onFalse)
		return
	}
	return onTrue.Clone(), nil
}

// Complex returns the shape resulting from the Complex operation.
func Complex(real, imag shapes.Shape) (output shapes.Shape, err error) {
	if real.DType != imag.DType {
		err = errors.Errorf("real and imaginary parts for Complex() must have the same data type, got %s and %s",
			real, imag)
		return
	}
	if real.DType != dtypes.Float32 && real.DType != dtypes.Float64 {
		err = errors.Errorf("real and imaginary parts for Complex() must have a float data type, got %s",
			real)
	}
	output = real.Clone()
	if real.DType == dtypes.Float32 {
		output.DType = dtypes.Complex64
	} else { // dtype = float64
		output.DType = dtypes.Complex128
	}
	return
}

// RealOrImag returns the shape resulting from the corresponding operations.
func RealOrImag(complexOperand shapes.Shape) (output shapes.Shape, err error) {
	if !complexOperand.DType.IsComplex() {
		err = errors.Errorf("Real() and Imag() require a complex data type, got %s", complexOperand)
	}
	output = complexOperand.Clone()
	if complexOperand.DType == dtypes.Complex64 {
		output.DType = dtypes.Float32
	} else { // dtype = complex128
		output.DType = dtypes.Float64
	}
	return
}

// Clamp returns the shape resulting from the corresponding operation.
func Clamp(min, operand, max shapes.Shape) (output shapes.Shape, err error) {
	if operand.DType != min.DType || operand.DType != max.DType {
		err = errors.Errorf("operand, min and max for Clamp() must have the same data type, got %s, %s and %s",
			operand, min, max)
		return
	}
	if operand.DType.IsComplex() || operand.DType == dtypes.Bool {
		err = errors.Errorf("Clamp() does not support complex or boolean data types, got %s", operand)
		return
	}
	if !min.IsScalar() && !min.Equal(operand) {
		err = errors.Errorf("min for Clamp() must either be a scalar or match the operand shape, instead got min=%s and operand=%s",
			min, operand)
		return
	}
	if !max.IsScalar() && !max.Equal(operand) {
		err = errors.Errorf("max for Clamp() must either be a scalar or match the operand shape, instead got max=%s and operand=%s",
			max, operand)
		return
	}
	output = operand.Clone()
	return
}

// Transpose all axes of the operand.
// There must be one value in permutations for each axis in the operand.
// The output will have: output.Shape.Dimension[ii] = operand.Shape.Dimension[permutations[i]].
func Transpose(operand shapes.Shape, permutation []int) (output shapes.Shape, err error) {
	rank := operand.Rank()
	if len(permutation) != rank {
		err = errors.Errorf("Transpose() requires all axes permutation to be defined, operand has shape %s, but %d permutation were given",
			operand, len(permutation))
		return
	}
	if rank == 0 {
		return operand, nil
	}

	// Check permutation axes are within range and unique.
	axesSet := slices.Clone(permutation)
	slices.Sort(axesSet)
	for ii, srcAxis := range axesSet {
		if srcAxis < 0 || srcAxis >= rank {
			err = errors.Errorf("invalid permutation axis %d given to Transpose(%s), it must be within the range of its rank",
				srcAxis, operand)
			return
		}
		if ii > 0 && srcAxis == axesSet[ii-1] {
			err = errors.Errorf("invalid permutation given to Transpose(%s, %v), there cannot be any repeated axis, each must appear exactly once",
				operand, permutation)
			return
		}
	}

	output = operand.Clone()
	for axis := range output.Dimensions {
		srcAxis := permutation[axis]
		output.Dimensions[axis] = operand.Dimensions[srcAxis]
	}
	return
}

// BroadcastInDim verifies that the arguments are valid.
// The output shape is already known, so nothing is returned.
//
// The axesMapping is changed in place, replacing negative axes with their positive equivalent.
func BroadcastInDim(operand, targetShape shapes.Shape, axesMapping []int) error {
	if operand.DType != targetShape.DType {
		return errors.Errorf("BroadcastInDim() requires the operand and the target shape to have the same data type, got operand=%s and targetShape=%s",
			operand, targetShape)
	}
	targetRank := targetShape.Rank()
	if targetRank < operand.Shape().Rank() {
		return errors.Errorf("BroadcastInDim() cannot be used to shrink the rank of the operand, got operand=%s and targetShape=%s",
			operand, targetShape)
	}
	if len(axesMapping) != operand.Shape().Rank() {
		return errors.Errorf("BroadcastInDim() requires all operand's axes mappings to be defined, operand has targetShape %s, but %d axes were given",
			operand, len(axesMapping))
	}
	usedAxis := utils.MakeSet[int](len(axesMapping))
	for operandAxis, targetAxis := range axesMapping {
		targetAxis, err := AdjustAxisToRank(targetAxis, targetRank)
		if err != nil {
			return errors.WithMessagef(err, "invalid axes mapping of operand axis %d to targetShape axis %d, targetShape targetShape is %s", operandAxis, targetAxis, targetShape)
		}
		if usedAxis.Has(targetAxis) {
			return errors.Errorf("BroadcastInDim() requires all targetShape axes to be unique, got duplicate axis %d", targetAxis)
		}
		usedAxis.Insert(targetAxis)
		operandDim := operand.Dimensions[operandAxis]
		targetDim := targetShape.Dimensions[targetAxis]
		if operandDim != 1 && operandDim != targetDim {
			return errors.Errorf("BroadcastInDim() requires all operand axes to be broadcast to be of dimension 1, but got operand.Dimensions[%d]=%d and targetShape.Dimension[%d]=%d",
				operandAxis, operandDim, targetAxis, targetDim)
		}
		axesMapping[operandAxis] = targetAxis
	}
	return nil
}

// Gather returns the output shape of a Gather operation.
func Gather(operand, startIndices shapes.Shape, indexVectorAxis int,
	offsetOutputAxes, collapsedSliceAxes, operandBatchingAxes,
	startIndicesBatchingAxes, startIndexMap,
	sliceSizes []int, indicesAreSorted bool) (output shapes.Shape, err error) {
	//fmt.Printf("Gather parameters:\n"+
	//	"  operand: %v\n"+
	//	"  startIndices: %v\n"+
	//	"  indexVectorAxis: %d\n"+
	//	"  offsetOutputAxes: %v\n"+
	//	"  collapsedSliceAxes: %v\n"+
	//	"  startIndexMap: %v\n"+
	//	"  sliceSizes: %v\n"+
	//	"  indicesAreSorted: %v\n",
	//	operand, startIndices, indexVectorAxis, offsetOutputAxes, collapsedSliceAxes,
	//	startIndexMap, sliceSizes, indicesAreSorted)
	_ = indicesAreSorted // Not used for shape inference.

	if operand.IsScalar() {
		return output, errors.Errorf("Gather() requires a non-scalar operand, got %s", operand)
	}

	// Check collapsedSliceAxes are all valid.
	setCollapsedAxes := utils.MakeSet[int]()
	for _, collapsedSliceAxis := range collapsedSliceAxes {
		if collapsedSliceAxis < 0 || collapsedSliceAxis >= operand.Rank() {
			return output, errors.Errorf("collapsed slice axis %d is out of range for operand %s", collapsedSliceAxis, operand)
		}
		if setCollapsedAxes.Has(collapsedSliceAxis) {
			return output, errors.Errorf("collapsed slice axis %d is defined more than once for operand %s", collapsedSliceAxis, operand)
		}
		setCollapsedAxes.Insert(collapsedSliceAxis)
	}

	// Check that batching axes are all valid, and that the batching axes in operand and startIndices match.
	setOperandBatchingAxes := utils.MakeSet[int]()
	for _, batchAxis := range operandBatchingAxes {
		if batchAxis < 0 || batchAxis >= operand.Rank() {
			return output, errors.Errorf("operand batch axis %d is out of range for operand %s", batchAxis, operand)
		}
		if setOperandBatchingAxes.Has(batchAxis) {
			return output, errors.Errorf("operand batch axis %d is defined more than once for operand %s", batchAxis, operand)
		}
		setCollapsedAxes.Insert(batchAxis)
	}
	setStartIndicesBatchingAxes := utils.MakeSet[int]()
	for _, batchAxis := range startIndicesBatchingAxes {
		if batchAxis < 0 || batchAxis >= startIndices.Rank() {
			return output, errors.Errorf("startIndices batch axis %d is out of range for startIndices %s", batchAxis, startIndices)
		}
		if setStartIndicesBatchingAxes.Has(batchAxis) {
			return output, errors.Errorf("startIndices batch axis %d is defined more than once for startIndices %s", batchAxis, startIndices)
		}
		if batchAxis == indexVectorAxis {
			return output, errors.Errorf("startIndices batch axis %d is the same as indexVectorAxis %d -- the same axis cannot be both", batchAxis, indexVectorAxis)
		}
		setStartIndicesBatchingAxes.Insert(batchAxis)
	}
	if len(operandBatchingAxes) != len(startIndicesBatchingAxes) {
		return output, errors.Errorf("operandBatchingAxes and startIndicesBatchingAxes must have the same number of axes (length), got %d and %d", len(operandBatchingAxes), len(startIndicesBatchingAxes))
	}
	for ii, operandBatchAxis := range operandBatchingAxes {
		startIndicesBatchAxis := startIndicesBatchingAxes[ii]
		if operand.Dim(operandBatchAxis) != startIndices.Dim(startIndicesBatchAxis) {
			return output, errors.Errorf("operand batch axis %d has dimension %d, but startIndices batch axis %d has dimension %d -- they must match",
				operandBatchAxis, operand.Dim(operandBatchAxis), startIndicesBatchAxis, startIndices.Dim(startIndicesBatchAxis))
		}
	}

	// Check slice sizes.
	if len(sliceSizes) != operand.Rank() {
		return output, errors.Errorf("sliceSizes must have one value per operand axes, so it length (%d) must match operand rank (%d)", len(sliceSizes), operand.Rank())
	}
	for axis, sliceSize := range sliceSizes {
		if sliceSize < 0 {
			return output, errors.Errorf("sliceSize %d for axis %d is negative, it must be non-negative", sliceSize, axis)
		}
		if operand.Dimensions[axis] < sliceSize {
			return output, errors.Errorf("sliceSize %d for axis %d is larger than the corresponding operand dimension %d", sliceSize, axis, operand.Dimensions[axis])
		}
	}
	for collapseAxis := range setCollapsedAxes {
		if sliceSizes[collapseAxis] != 1 {
			return output, errors.Errorf("collapsed slice axis %d must have sliceSize 1, but got %d", collapseAxis, sliceSizes[collapseAxis])
		}
	}
	for batchAxis := range operandBatchingAxes {
		if sliceSizes[batchAxis] != 1 {
			return output, errors.Errorf("operand's batching axis %d must have sliceSize 1, but got %d", batchAxis, sliceSizes[batchAxis])
		}
	}

	// Check that the operand's axes are all used.
	if operand.Rank() != len(offsetOutputAxes)+len(collapsedSliceAxes)+len(operandBatchingAxes) {
		return output, errors.Errorf("the number of collapsedSliceAxes (%d) + the number of offsetOutputAxes (%d) + the number of operandsBatchingAxes (%d) must be equal to the number of axes in the operand (operand.Rank()=%d)",
			len(collapsedSliceAxes), len(offsetOutputAxes), len(operandBatchingAxes), operand.Rank())
	}

	// Check indexVectorAxis: it is ok if it is equal to startIndices.rank, in which case we assume an implicit extra axis of dimension 1.
	if indexVectorAxis < 0 || indexVectorAxis > operand.Rank() {
		return output, errors.Errorf("indexVectorAxis=%d is out of range for operand %s", indexVectorAxis, operand)
	}

	// Check startIndexMap is set for the dimensions of indexVectorAxis in startIndices.
	numIndexedAxes := 1
	if indexVectorAxis < startIndices.Rank() {
		numIndexedAxes = startIndices.Dimensions[indexVectorAxis]
	}
	if len(startIndexMap) != numIndexedAxes {
		if indexVectorAxis == startIndices.Rank() {
			return output, errors.Errorf("when indexVectorAxis==startIndices.Rank() we assume only one axis is being indexed, so startIndexMap be of length 1, got %d instead",
				len(startIndexMap))
		}
		return output, errors.Errorf("startIndexMap must have one value per dimension of indexVectorAxis, so its length (%d) must match startIndices.Dimensions[%d] (==%d)",
			len(startIndexMap), indexVectorAxis, numIndexedAxes)
	}
	for idx, operandAxis := range startIndexMap {
		if operandAxis < 0 || operandAxis >= operand.Rank() {
			return output, errors.Errorf("startIndexMap[%d]=%d is out of range for operand %s", idx, operandAxis, operand)
		}
	}

	// The number of batch axes is usually the number of startIndices - 1, except if indexVectorAxis==rank,
	// in which case we assume an extra one in the end.
	batchRank := startIndices.Rank()
	if indexVectorAxis < startIndices.Rank() {
		batchRank--
	}

	// Build output shape: the order is defined as:
	//
	// - Axes in offsetOutputAxes are preset as offset, and their dimensions are taken sequentially from non-collapsed operand axes.
	// - Remaining axes are filled in order from the batch axes, taken from startIndices.
	output = shapes.Make(operand.DType)
	output.Dimensions = make([]int, batchRank+len(offsetOutputAxes))

	setOffsetOutputAxes := utils.MakeSet[int]()
	for _, offsetOutputAxis := range offsetOutputAxes {
		if offsetOutputAxis < 0 || offsetOutputAxis >= output.Rank() {
			return shapes.Invalid(), errors.Errorf("offset output axis %d is out of range for output of rank %d", offsetOutputAxis, output.Rank())
		}
		if setOffsetOutputAxes.Has(offsetOutputAxis) {
			return shapes.Invalid(), errors.Errorf("offset output axis %d is defined more than once: offsetOutputAxes=%v", offsetOutputAxis, offsetOutputAxes)
		}
		setOffsetOutputAxes.Insert(offsetOutputAxis)
	}
	offsetDims := make([]int, 0, len(offsetOutputAxes))
	for axis, sliceSize := range sliceSizes {
		if setCollapsedAxes.Has(axis) {
			// This is a collapsed axis and not used as an offset.
			continue
		}
		if setOperandBatchingAxes.Has(axis) {
			// This is a batch axis and not used as an offset.
			continue
		}
		offsetDims = append(offsetDims, sliceSize)
	}
	offsetDimsIdx := 0

	// Batch axes' dimensions are set from the inputIndices.
	batchDimsIdx := 0
	for axis := range output.Dimensions {
		if setOffsetOutputAxes.Has(axis) {
			// Take an offset dimension from sliceSizes:
			output.Dimensions[axis] = offsetDims[offsetDimsIdx]
			offsetDimsIdx++
		} else {
			// Take a batch dimension from startIndices:
			if batchDimsIdx == indexVectorAxis {
				// Skip the index axis.
				batchDimsIdx++
			}
			output.Dimensions[axis] = startIndices.Dimensions[batchDimsIdx]
			batchDimsIdx++
		}
	}
	return output, nil
}

// Concatenate calculates the output shape of a Concatenate operation.
// It takes a slice of input shapes and the dimension along which to concatenate.
func Concatenate(inputs []shapes.Shape, axis int) (output shapes.Shape, err error) {
	if len(inputs) == 0 {
		return shapes.Invalid(), errors.Errorf("Concatenate requires at least one input shape")
	}

	// Initialize output dimensions with the first shape.
	firstShape := inputs[0]
	dtype := firstShape.DType
	rank := firstShape.Rank()
	output = firstShape.Clone()
	if dtype == dtypes.InvalidDType {
		return shapes.Invalid(), errors.Errorf("invalid shape %s for first input of Concatenate", firstShape)
	}
	if len(inputs) == 1 {
		return firstShape, nil
	}

	if axis < 0 || axis >= rank {
		return shapes.Invalid(), errors.Errorf("invalid concatenation axis %d for shapes with rank %d", axis, rank)
	}

	// Validate further inputs and accumulate the concatenation axis size.
	for i := 1; i < len(inputs); i++ {
		currentShape := inputs[i]
		if currentShape.DType == dtypes.InvalidDType {
			return shapes.Invalid(), errors.Errorf("invalid shape %s for input #%d of Concatenate", currentShape, i)
		}
		if currentShape.DType != dtype {
			return shapes.Invalid(), errors.Errorf("mismatched DTypes for Concatenate: input #0 has %s, input #%d has %s",
				dtype, i, currentShape.DType)
		}
		if currentShape.Rank() != rank {
			return shapes.Invalid(), errors.Errorf("mismatched ranks for Concatenate: input #0 has rank %d, input #%d has rank %d",
				rank, i, currentShape.Rank())
		}

		for d := 0; d < rank; d++ {
			if d == axis {
				output.Dimensions[d] += currentShape.Dimensions[d]
			} else {
				if currentShape.Dimensions[d] != output.Dimensions[d] {
					return shapes.Invalid(), errors.Errorf("mismatched dimensions for Concatenate at axis %d (non-concatenation axis): input #0 has %d, input #%d has %d",
						d, output.Dimensions[d], i, currentShape.Dimensions[d])
				}
			}
		}
	}
	return output, nil
}

// Scatter checks that the parameters are consistent. The output shapes returned are the unchanged inputs -- the scattered
// updates are applied to the inputs, but their shapes are unchanged.
//
// The Scatter operations indicesAreSorted and uniqueIndices don't play a role in this.
func Scatter(inputs []shapes.Shape, scatterIndices shapes.Shape, updates []shapes.Shape,
	updateWindowAxes, insertedWindowAxes []int,
	inputBatchingAxes, scatterIndicesBatchingAxes []int,
	indexedInputAxes []int, indexVectorAxis int,
	updateComputationInputs, updateComputationOutputs []shapes.Shape) (outputs []shapes.Shape, err error) {
	// Check the number of inputs and updates.
	if len(inputs) == 0 {
		return nil, errors.Errorf("Scatter() requires at least one input")
	}
	if len(inputs) != len(updates) {
		return nil, errors.Errorf("Scatter() requires the same number of inputs and updates, got %d inputs and %d updates", len(inputs), len(updates))
	}

	// Check the dtypes match.
	if scatterIndices.DType == dtypes.InvalidDType {
		return nil, errors.Errorf("invalid shape for scatterIndices (%s)", scatterIndices)
	}
	input0 := inputs[0] // Shortcut, it will be used for the other checks.
	for i, input := range inputs {
		if input.DType == dtypes.InvalidDType {
			return nil, errors.Errorf("invalid shape for inputs[%d]=%s", i, input)
		}
		if slices.Compare(input0.Dimensions, input.Dimensions) != 0 {
			return nil, errors.Errorf("all inputs must have the same shape (even if different dtypes), "+
				"but inputs[0]=%s and inputs[%d]=%s", input0, i, input)
		}
	}
	updates0 := updates[0] // Shortcut, it will be used for the other checks.
	for i, update := range updates {
		if update.DType == dtypes.InvalidDType {
			return nil, errors.Errorf("invalid shape for updates[%d]=%s", i, update)
		}
		if update.DType != inputs[i].DType {
			return nil, errors.Errorf("data types (DType) for inputs[%d]=%s and corresponding updates[%d]=%s must match",
				i, inputs[i], i, update)
		}
		if slices.Compare(updates0.Dimensions, update.Dimensions) != 0 {
			return nil, errors.Errorf("all updates must have the same shape (even if different dtypes), "+
				"but updates[0]=%s and updates[%d]=%s", updates0, i, update)
		}
	}

	// Inputs rank:
	if input0.Rank() != len(updateWindowAxes)+len(inputBatchingAxes)+len(insertedWindowAxes) {
		return nil, errors.Errorf("the number of updateWindowAxes (%d) + the number of inputBatchingAxes (%d) "+
			"+ the number of insertedWindowAxes (%d) must be equal to the number of axes in the inputs (inputs rank is =%d)",
			len(updateWindowAxes), len(inputBatchingAxes), len(insertedWindowAxes), input0.Rank())
	}

	// TODO: perform the other checks in StableHLO specification in https://openxla.org/stablehlo/spec#scatter
	//       For now we rely on the checks that PJRT will perform anyway.
	_ = scatterIndicesBatchingAxes
	_ = indexedInputAxes
	_ = indexVectorAxis

	// Check updateComputation inputs and outputs.
	if len(updateComputationOutputs) != len(inputs) {
		return nil, errors.Errorf("updateComputation must have as many outputs (%d) as there are inputs (%d) to the Scatter operation",
			len(updateComputationOutputs), len(inputs))
	}
	if len(updateComputationInputs) != 2*len(inputs) {
		return nil, errors.Errorf(
			"updateComputation must have as many inputs (%d) as there are 2 * inputs (%d) = %d to the Scatter operation, "+
				"one value coming from the input, the other from the update",
			len(updateComputationInputs), len(inputs), 2*len(inputs))
	}
	for i := range len(inputs) {
		dtype := updateComputationInputs[i].DType
		if !inputs[i].DType.IsPromotableTo(dtype) {
			return nil, errors.Errorf(
				"inputs[%d].DType=%s is not promotable to updateComputationFn input parameter #%d's dtype (%s)",
				i, inputs[i].DType, i, dtype)
		}
		if dtype != updateComputationInputs[i+len(inputs)].DType {
			return nil, errors.Errorf(
				"updateComputation input #%d (%s) must match the dtype of the corresponding input #(%d + %d) (%s)",
				i, dtype, i, len(inputs), updateComputationInputs[i+len(inputs)].DType)
		}
		if dtype != updateComputationOutputs[i].DType {
			return nil, errors.Errorf(
				"updateComputation input #%d (%s) must match the dtype of the corresponding output #%d (%s)",
				i, dtype, i, updateComputationOutputs[i].DType)
		}
	}

	// Build output shapes based on the inputs and the outputs of the updateComputation.
	outputs = make([]shapes.Shape, len(inputs))
	for i, input := range inputs {
		outputs[i] = input.Clone()
		outputs[i].DType = updateComputationOutputs[i].DType
	}
	return
}

// Slice calculates the output shape for a Slice operation.
// It checks that starts, limits, and strides have the correct length (matching operand rank),
// and that the slice parameters are valid for the operand's dimensions.
// Strides must be positive.
func Slice(operand shapes.Shape, starts, limits, strides []int) (output shapes.Shape, err error) {
	rank := operand.Rank()
	opName := "Slice"
	if operand.DType == dtypes.InvalidDType {
		return shapes.Invalid(), errors.Errorf("%s: invalid operand shape %s", opName, operand)
	}
	if len(starts) != rank {
		return shapes.Invalid(), errors.Errorf("%s: len(starts)=%d, but operand rank is %d", opName, len(starts), rank)
	}
	if len(limits) != rank {
		return shapes.Invalid(), errors.Errorf("%s: len(limits)=%d, but operand rank is %d", opName, len(limits), rank)
	}
	if len(strides) != rank {
		return shapes.Invalid(), errors.Errorf("%s: len(strides)=%d, but operand rank is %d", opName, len(strides), rank)
	}

	output = shapes.Shape{
		DType:      operand.DType,
		Dimensions: make([]int, rank),
	}

	for axis := 0; axis < rank; axis++ {
		start, limit, stride := starts[axis], limits[axis], strides[axis]
		dimSize := operand.Dimensions[axis]

		if stride <= 0 {
			return shapes.Invalid(), errors.Errorf("%s: stride must be positive, but got stride[%d]=%d for operand shape %s",
				opName, axis, stride, operand)
		}
		if start < 0 || start >= dimSize {
			return shapes.Invalid(), errors.Errorf("%s: start index %d is out of bounds for axis %d with size %d (operand shape %s)",
				opName, start, axis, dimSize, operand)
		}
		// Limit can be equal to dimSize.
		if limit < start || limit > dimSize {
			return shapes.Invalid(), errors.Errorf("%s: limit index %d is out of bounds for axis %d (start=%d, size=%d, operand shape %s)",
				opName, limit, axis, start, dimSize, operand)
		}

		// The first one is always taken, so we use the ceiling of the division.
		outputDimSize := (limit - start + (stride - 1)) / stride
		output.Dimensions[axis] = outputDimSize
	}

	return output, nil
}

// ArgMinMax calculates the output shape for an ArgMinMax operation.
// It will be the shape of the operand minus the "reduce" axis.
func ArgMinMax(operand shapes.Shape, axis int, outputDType dtypes.DType) (output shapes.Shape, err error) {
	if !outputDType.IsInt() {
		err = errors.Errorf("ArgMinMax outputDType must be an integer type, got %s", outputDType)
		return
	}
	if !operand.DType.IsFloat() && !operand.DType.IsInt() {
		err = errors.Errorf("ArgMinMax operand DType must be a floating point or integer type, got %s", operand)
		return
	}
	if operand.IsScalar() {
		err = errors.Errorf("ArgMinMax requires a non-scalar operand, got %s", operand)
		return
	}
	if axis < 0 || axis >= operand.Rank() {
		err = errors.Errorf("ArgMinMax axis %d is out of range for operand %s", axis, operand)
		return
	}
	newDims := slices.Clone(operand.Dimensions)
	newDims = slices.Delete(newDims, axis, axis+1)
	output = shapes.Make(outputDType, newDims...)
	return
}

// ReduceWindow returns the expected output shape for the operation.
//
// Notice it doesn't take as input the reductionType parameter, since it doesn't affect the output shape.
func ReduceWindow(inputs, initialValues []shapes.Shape, reductionInputs, reductionOutputs []shapes.Shape,
	windowDimensions, strides, baseDilations, windowDilations []int, paddings [][2]int) (outputs []shapes.Shape, err error) {
	numReductions := len(inputs)
	if numReductions < 0 {
		return nil, errors.New("ReduceWindow requires at least one input")
	}
	baseShape := inputs[0]
	for i, input := range inputs {
		if !input.Ok() {
			return nil, errors.Errorf("ReduceWindow: invalid input[%d] shape %s", i, input)
		}
		err = input.CheckDims(baseShape.Dimensions...)
		if err != nil {
			err = errors.WithMessagef(err, "ReduceWindow: all inputs must have the same shape, inputs[0] has shape %s, but inputs[%d] has shape %s",
				baseShape, i, input)
			return
		}
	}
	rank := baseShape.Rank()
	for i, initialValue := range initialValues {
		if initialValue.DType != inputs[i].DType {
			return nil, errors.Errorf("ReduceWindow: initialValue[%d] has DType %s, but inputs[%d] has DType %s",
				i, initialValue.DType, i, inputs[i].DType)
		}
		if !initialValue.IsScalar() {
			return nil, errors.Errorf("ReduceWindow: initialValue[%d] must be a scalar, but got shape %s", i, initialValue)
		}
	}

	// Check that all reduction inputs and outputs are valid.
	if len(reductionInputs) != 2*numReductions {
		return nil, errors.Errorf("The reduction function for the ReduceWindow operation must have 2 inputs for each initialValue, but reduction has %d inputs for 2*%d=%d initial values",
			len(reductionInputs), len(initialValues), 2*len(initialValues))
	}
	if len(reductionOutputs) != numReductions {
		return nil, errors.Errorf("The reduction function for the ReduceWindow operation must have 1 output for each initialValue, but reduction has %d outputs for %d initial values",
			len(reductionOutputs), len(initialValues))
	}
	for i := range numReductions {
		dtype := reductionInputs[i].DType
		if dtype != reductionInputs[i+numReductions].DType || dtype != reductionOutputs[i].DType {
			return nil, errors.Errorf("ReduceWindow requires the same dtype for lhs[i], rhs[i] inputs and output[i], got lhs[%d]=%s and rhs[%d+%d]=%s and output[%d]=%s",
				i, reductionInputs[i], i, numReductions, reductionInputs[i+numReductions], i, reductionOutputs[i])
		}
		if !inputs[i].DType.IsPromotableTo(dtype) {
			return nil, errors.Errorf(
				"inputs[%d].DType=%s is not promotable to reductionFn input parameter #%d's dtype (%s)",
				i, inputs[i].DType, i, dtype)
		}
	}

	// Validate lengths of slice parameters against rank.
	if len(windowDimensions) != rank {
		return nil, errors.Errorf("ReduceWindow: len(windowDimensions)=%d, but inputs rank is %d", len(windowDimensions), rank)
	}
	if len(strides) != rank {
		return nil, errors.Errorf("ReduceWindow: len(strides)=%d, but inputs rank is %d", len(strides), rank)
	}
	if len(paddings) != rank {
		return nil, errors.Errorf("ReduceWindow: len(paddings)=%d, but inputs rank is %d", len(paddings), rank)
	}
	if len(baseDilations) != rank {
		return nil, errors.Errorf("ReduceWindow: baseDilations is not nil and len(baseDilations)=%d, but inputs rank is %d", len(baseDilations), rank)
	}
	if len(windowDilations) != rank {
		return nil, errors.Errorf("ReduceWindow: windowDilations is not nil and len(windowDilations)=%d, but inputs rank is %d", len(windowDilations), rank)
	}

	// If operand is a scalar (rank 0), the output is also a scalar of the same type.
	// All dimension-specific parameters (windowDimensions, strides, etc.) must be empty,
	// which is enforced by the length checks above (e.g., len(windowDimensions) == rank == 0).
	if rank == 0 {
		outputs = inputs
		return
	}

	// Each output dimension is calculated orthogonally to the others.
	outputDims := make([]int, rank)
	operand := inputs[0]
	for i := 0; i < rank; i++ {
		inputDim := operand.Dimensions[i] // Already validated to be > 0 by shapes.Make
		windowDim := windowDimensions[i]
		if windowDim < 1 {
			return nil, errors.Errorf("ReduceWindow: windowDimensions[%d]=%d must be >= 1 for operand shape %s", i, windowDim, operand)
		}
		stride := strides[i]
		if stride < 1 {
			return nil, errors.Errorf("ReduceWindow: strides[%d]=%d must be >= 1 for operand shape %s", i, stride, operand)
		}
		paddingLow := paddings[i][0]
		paddingHigh := paddings[i][1]
		if paddingLow < 0 || paddingHigh < 0 {
			return nil, errors.Errorf("ReduceWindow: paddings[%d]=[%d, %d] must be non-negative for operand shape %s", i, paddingLow, paddingHigh, operand)
		}
		baseDilation := baseDilations[i]
		if baseDilation < 1 {
			return nil, errors.Errorf("ReduceWindow: baseDilations[%d]=%d must be >= 1 for operand shape %s", i, baseDilation, operand)
		}
		windowDilation := windowDilations[i]
		if windowDilation < 1 {
			return nil, errors.Errorf("ReduceWindow: windowDilations[%d]=%d must be >= 1 for operand shape %s", i, windowDilation, operand)
		}

		// Effective input dimension after base dilation.
		// (size - 1) * dilation + 1
		effectiveInputDim := (inputDim-1)*baseDilation + 1

		// Effective window dimension after window dilation.
		effectiveWindowDim := (windowDim-1)*windowDilation + 1

		// Padded effective input size for this dimension.
		paddedEffectiveInputDim := effectiveInputDim + paddingLow + paddingHigh

		// Numerator for the output dimension formula.
		// output_dim = floor((padded_input_size - effective_window_size) / stride) + 1
		// The numerator must be non-negative for the output dimension to be at least 1.
		if effectiveWindowDim > paddedEffectiveInputDim {
			return nil, errors.Errorf(
				"ReduceWindow: effective window dimension %d for axis %d is larger than padded effective input dimension %d. (input_dim: %d, base_dilation: %d, window_dim: %d, window_dilation: %d, padding: [%d,%d]) for operand shape %s",
				effectiveWindowDim, i, paddedEffectiveInputDim, inputDim, baseDilation, windowDim, windowDilation, paddingLow, paddingHigh, operand)
		}

		numerator := paddedEffectiveInputDim - effectiveWindowDim
		outputDims[i] = numerator/stride + 1
	}

	outputs = make([]shapes.Shape, len(inputs))
	for i, output := range reductionOutputs {
		outputs[i] = shapes.Make(output.DType, outputDims...)
	}
	return
}

// Convolve returns the expected output shape for the Convolve operation.
func Convolve(input, kernel shapes.Shape,
	strides []int, paddings [][2]int, inputDilations, kernelDilations []int,
	inputBatchAxis, inputChannelsAxis int, inputSpatialAxes []int,
	kernelInputChannelsAxis, kernelOutputChannelsAxis int, kernelSpatialAxes []int,
	outputBatchAxis, outputChannelsAxis int, outputSpatialAxes []int,
	channelGroupCount, batchGroupCount int) (shapes.Shape, error) {
	// Convenient error returns.
	errorf := func(format string, args ...any) (shapes.Shape, error) {
		return shapes.Invalid(), errors.Errorf("Convolve:  "+format, args...)
	}

	if !input.Ok() {
		return errorf("invalid input (operand) shape %s", input)
	}
	if !kernel.Ok() {
		return errorf("invalid kernel shape %s", kernel)
	}

	// Check ranks.
	rank := input.Rank()
	spatialRank := rank - 2
	if rank < 3 {
		return errorf("input (operand) needs to be at least rank-3 with axes (in any order) batch, channels and spatial -- input shape is %s", input)
	}
	if kernel.Rank() != rank {
		return errorf("input (operand) and kernel have different rank!? -- input shape is %s and kernel shape is %s", input, kernel)
	}

	// Check axes configuration:
	if len(inputSpatialAxes) != spatialRank {
		return errorf("inputSpatialAxes (%v) must provide one value for each spatial axis (%d), input shape is %s",
			inputSpatialAxes, spatialRank, input)
	}
	inputAxes := utils.SetWith(inputBatchAxis, inputChannelsAxis)
	for _, inputAxis := range inputSpatialAxes {
		if inputAxis < 0 || inputAxis >= rank {
			return errorf("invalid input axes configuration (axis %d is out-of-bounds): batch=%d, channel=%d, spatial=%v", inputAxis, inputBatchAxis, inputChannelsAxis, inputSpatialAxes)
		}
		inputAxes.Insert(inputAxis)
	}
	if len(inputAxes) != rank {
		return errorf("duplicate input axes configuration: batch=%d, channel=%d, spatial=%v", inputBatchAxis, inputChannelsAxis, inputSpatialAxes)
	}

	if len(kernelSpatialAxes) != spatialRank {
		return shapes.Invalid(), errors.Errorf("Convolve: kernelSpatialAxes (%v) must provide one value for each spatial axis (%d), kernel shape is %s",
			inputSpatialAxes, spatialRank, kernel)
	}
	kernelAxes := utils.SetWith(kernelInputChannelsAxis, kernelOutputChannelsAxis)
	for _, kernelAxis := range kernelSpatialAxes {
		if kernelAxis < 0 || kernelAxis >= rank {
			return errorf("invalid kernel axes configuration (axis %d is out-of-bounds): input channel=%d, output channel=%d, spatial=%v",
				kernelAxis, kernelInputChannelsAxis, kernelOutputChannelsAxis, kernelSpatialAxes)
		}
		kernelAxes.Insert(kernelAxis)
	}
	if len(kernelAxes) != rank {
		return errorf("duplicate kernel axes configuration: input channel=%d, output channel=%d, spatial=%v",
			kernelInputChannelsAxis, kernelOutputChannelsAxis, kernelSpatialAxes)
	}

	if len(outputSpatialAxes) != spatialRank {
		return errorf("outputSpatialAxes (%v) must have one value for each spatial axis (%d), input shape is %s",
			outputSpatialAxes, spatialRank, input)
	}
	outputAxes := utils.SetWith(outputBatchAxis, outputChannelsAxis)
	for _, outputAxis := range outputSpatialAxes {
		if outputAxis < 0 || outputAxis >= rank {
			return errorf("invalid output axes configuration (axis %d is out-of-bounds): batch=%d, channels=%d, spatial=%v", outputAxis, outputBatchAxis, outputChannelsAxis, outputSpatialAxes)
		}
		outputAxes.Insert(outputAxis)
	}
	if len(outputAxes) != rank {
		return errorf("duplicate output axes configuration: batch=%d, channel=%d, spatial=%v",
			outputBatchAxis, outputChannelsAxis, outputSpatialAxes)
	}

	// Check strides, paddings, inputDilations and kernelDilations.
	if len(strides) != 0 && len(strides) != spatialRank {
		return errorf("strides (%v) must either be nil or provide one value for each spatial axis (%d), input shape is %s",
			strides, spatialRank, input.Shape())
	}
	if len(paddings) != 0 && len(paddings) != spatialRank {
		return errorf("paddings (%v) must either be nil or provide one value for each spatial axis (%d), input shape is %s",
			paddings, spatialRank, input.Shape())
	}
	if len(inputDilations) != 0 && len(inputDilations) != spatialRank {
		return errorf("inputDilations (%v) must either be nil or provide one value for each spatial axis (%d), input shape is %s",
			inputDilations, spatialRank, input.Shape())
	}
	for i, dilation := range inputDilations {
		if dilation < 1 {
			return errorf("inputDilations[%d]=%d must be >= 1 for input shape %s", i, dilation, input)
		}
	}
	if len(kernelDilations) != 0 && len(kernelDilations) != spatialRank {
		return errorf("kernelDilations (%v) must either be nil or provide one value for each spatial axis (%d), input shape is %s",
			kernelDilations, spatialRank, input.Shape())
	}
	for i, dilation := range kernelDilations {
		if dilation < 1 {
			return errorf("kernelDilations[%d]=%d must be >= 1 for input shape %s", i, dilation, input)
		}
	}

	if channelGroupCount > 1 && batchGroupCount > 1 {
		return errorf("at most one of channelGroupCount (%d) or batchGroupCount (%d) can be set to > 1", channelGroupCount, batchGroupCount)
	}

	// Check that channels (feature dimensions) are valid.
	inputChannels := input.Dim(inputChannelsAxis)
	outputChannels := kernel.Dim(kernelOutputChannelsAxis)
	if channelGroupCount < 1 {
		return errorf("channelGroupCount=%d must be >= 1 for input shape %s", channelGroupCount, input)
	}
	if inputChannels%channelGroupCount != 0 {
		return errorf("input channels dimension %d must be divisible by channelGroupCount %d", inputChannels, channelGroupCount)
	}
	if outputChannels%channelGroupCount != 0 {
		return errorf("kernel output channels dimension %d must be divisible by channelGroupCount %d", outputChannels, channelGroupCount)
	}
	kernelInputChannels := kernel.Dim(kernelInputChannelsAxis)
	if inputChannels != kernelInputChannels*channelGroupCount {
		return errorf("we must have inputChannels (=%d) = kernelInputChannels (=%d) * channelGroupCount (=%d) -- input shape is %s, kernel shape is %s",
			inputChannels, kernelInputChannels, channelGroupCount, input, kernel)
	}

	// Check batchGroupCount.
	inputBatch := input.Dim(inputBatchAxis)
	if batchGroupCount < 1 {
		return errorf("batchGroupCount=%d must be >= 1 for input shape %s", batchGroupCount, input)
	}
	if inputBatch%batchGroupCount != 0 {
		return errorf("input batch dimension %d must be divisible by batchGroupCount %d", inputBatch, batchGroupCount)
	}
	if outputChannels%batchGroupCount != 0 {
		return errorf("output channels dimension %d must be divisible by batchGroupCount %d", outputChannels, batchGroupCount)
	}

	// Find the output shape.
	output := input.Clone()
	output.Dimensions[outputBatchAxis] = inputBatch / batchGroupCount
	output.Dimensions[outputChannelsAxis] = outputChannels

	for spatialAxisIdx, inputAxis := range inputSpatialAxes {
		inputDim := input.Dim(inputAxis)
		filterAxis := kernelSpatialAxes[spatialAxisIdx]
		kernelDim := kernel.Dim(filterAxis)
		var (
			stride  int
			padding [2]int
		)
		if strides != nil {
			stride = strides[spatialAxisIdx]
		}
		if paddings != nil {
			padding = paddings[spatialAxisIdx]
		}
		inputDilation, kernelDilation := 1, 1
		if inputDilations != nil {
			inputDilation = inputDilations[spatialAxisIdx]
		}
		if kernelDilations != nil {
			kernelDilation = kernelDilations[spatialAxisIdx]
		}

		// Calculate outputDim of the convolution.
		if stride < 1 {
			return errorf("stride[%d]=%d must be >= 1 for input shape %s", spatialAxisIdx, stride, input)
		}

		// Calculate effective dimensions after dilations
		effectiveInputDim := (inputDim-1)*inputDilation + 1
		effectiveKernelDim := (kernelDim-1)*kernelDilation + 1

		// Calculate padded input size
		paddedEffectiveInputDim := effectiveInputDim + padding[0] + padding[1]

		// Calculate output dimension
		if effectiveKernelDim > paddedEffectiveInputDim {
			return errorf("effective kernel dimension %d for axis %d is larger than padded effective input dimension %d. "+
				"(input_dim: %d, input_dilation: %d, filter_dim: %d, filter_dilation: %d, padding: [%d,%d]) for input shape %s",
				effectiveKernelDim, inputAxis, paddedEffectiveInputDim, inputDim, inputDilation, kernelDim, kernelDilation,
				padding[0], padding[1], input)
		}
		outputDim := (paddedEffectiveInputDim-effectiveKernelDim)/stride + 1
		outputSpatialAxis := outputSpatialAxes[spatialAxisIdx]
		output.Dimensions[outputSpatialAxis] = outputDim
	}

	return output, nil
}

// AdjustAxisToRank returns a positive axis, adjusting negative numbers to the correct rank.
func AdjustAxisToRank(axis, rank int) (int, error) {
	if axis < -rank || axis >= rank {
		return -1, errors.Errorf("axis %d is out of range for the rank %d", axis, rank)
	}
	if axis < 0 {
		axis += rank
	}
	return axis, nil
}

// DotGeneral returns the shape resulting from the corresponding operations.
//
// It also has a side effect on the axes' specifications: it converts negative axes to their
// corresponding positive axes, and it sorts the axes in ascending order.
func DotGeneral(
	lhs shapes.Shape, lhsContractingAxes, lhsBatchAxes []int,
	rhs shapes.Shape, rhsContractingAxes, rhsBatchAxes []int,
	outputDType dtypes.DType) (output shapes.Shape, err error) {
	dtype := lhs.DType
	if dtype != rhs.DType {
		err = errors.Errorf("DotGeneral lhs (left-hand-side) and rhs operands don't match data types: %s and %s", dtype, rhs.DType)
		return
	}
	if len(lhsContractingAxes) != len(rhsContractingAxes) {
		err = errors.Errorf("DotGeneral number of contracting axes for lhs (%d) doesn't match rhs (%d)",
			len(lhsContractingAxes), len(rhsContractingAxes))
		return
	}
	if len(lhsBatchAxes) != len(rhsBatchAxes) {
		err = errors.Errorf("DotGeneral number of contracting axes for lhs (%d) doesn't match rhs (%d)",
			len(lhsContractingAxes), len(rhsContractingAxes))
	}
	lhsRank := lhs.Rank()
	rhsRank := rhs.Rank()

	// Validate and adjust axes.
	for ii, axis := range lhsContractingAxes {
		lhsContractingAxes[ii], err = AdjustAxisToRank(axis, lhsRank)
		if err != nil {
			err = errors.WithMessagef(err, "while adjusting contractingAxes for DotGeneral(lhs=%s, lhsContractingAxes=%v)", lhs, lhsContractingAxes)
			return
		}
	}
	for ii, axis := range lhsBatchAxes {
		lhsBatchAxes[ii], err = AdjustAxisToRank(axis, lhsRank)
		if err != nil {
			err = errors.WithMessagef(err, "while adjusting batchAxes for DotGeneral(lhs=%s, lhsBatchAxes=%v)", lhs, lhsBatchAxes)
			return
		}
	}
	for ii, axis := range rhsContractingAxes {
		rhsContractingAxes[ii], err = AdjustAxisToRank(axis, rhsRank)
		if err != nil {
			err = errors.WithMessagef(err, "while adjusting contractingAxes for DotGeneral(rhs=%s, rhsContractingAxes=%v)", rhs, rhsContractingAxes)
			return
		}
	}
	for ii, axis := range rhsBatchAxes {
		rhsBatchAxes[ii], err = AdjustAxisToRank(axis, rhsRank)
		if err != nil {
			err = errors.WithMessagef(err, "while adjusting batchAxes for DotGeneral(rhs=%s, rhsBatchAxes=%v)", rhs, rhsBatchAxes)
			return
		}
	}

	// Check that batch and contracting dimensions from lhs and rhs match.
	batchDims := make([]int, len(lhsBatchAxes))
	contractingDims := make([]int, len(lhsContractingAxes))
	for ii, lhsAxis := range lhsContractingAxes {
		rhsAxis := rhsContractingAxes[ii]
		if lhs.Dimensions[lhsAxis] != rhs.Dimensions[rhsAxis] {
			err = errors.Errorf("DotGeneral contracting dimensions don't match: lhs[%d]=%d != rhs[%d]=%d",
				lhsAxis, lhs.Dimensions[lhsAxis], rhsAxis, rhs.Dimensions[rhsAxis])
			return
		}
		contractingDims[ii] = lhs.Dimensions[lhsAxis]
	}
	for ii, lhsAxis := range lhsBatchAxes {
		rhsAxis := rhsBatchAxes[ii]
		if lhs.Dimensions[lhsAxis] != rhs.Dimensions[rhsAxis] {
			err = errors.Errorf("DotGeneral batch dimensions don't match: lhs[%d]=%d != rhs[%d]=%d",
				lhsAxis, lhs.Dimensions[lhsAxis], rhsAxis, rhs.Dimensions[rhsAxis])
			return
		}
		batchDims[ii] = lhs.Dimensions[lhsAxis]
	}

	// Find sizes of the normalized operands ([batchSize, crossSize, contractSize]).
	var lhsCrossDims, rhsCrossDims []int
	batchSize, lhsCrossSize, contractingSize, lhsCrossDims := dotGeneralFindSizes(lhs, lhsContractingAxes, lhsBatchAxes)
	_, rhsCrossSize, _, rhsCrossDims := dotGeneralFindSizes(rhs, rhsContractingAxes, rhsBatchAxes)

	// Check that all sizes are positive
	if batchSize < 0 || lhsCrossSize < 0 || contractingSize < 0 || rhsCrossSize < 0 {
		err = errors.Errorf("DotGeneral sizes must be positive: lhs(batch=%d, cross=%d, contracting=%d), rhs(cross=%d)",
			batchSize, lhsCrossSize, contractingSize, rhsCrossSize)
		return
	}

	// Reshape result to recover batch and cross dimensions.
	resultingDims := make([]int, 0, len(batchDims)+len(lhsCrossDims)+len(rhsCrossDims))
	resultingDims = append(resultingDims, batchDims...)
	resultingDims = append(resultingDims, lhsCrossDims...)
	resultingDims = append(resultingDims, rhsCrossDims...)
	output = shapes.Make(outputDType, resultingDims...)
	return
}

func dotGeneralFindSizes(shape shapes.Shape, contractingAxes, batchAxes []int) (batchSize, crossSize, contractingSize int, crossDims []int) {
	rank := shape.Rank()
	axesTypes := make([]int, rank)

	// Mark axes types: 1 for contracting, 2 for batch
	for _, axis := range contractingAxes {
		axesTypes[axis] = 1
	}
	for _, axis := range batchAxes {
		axesTypes[axis] = 2
	}

	// Calculate sizes by multiplying dimensions according to the axis type.
	batchSize, crossSize, contractingSize = 1, 1, 1
	crossDims = make([]int, 0, rank-len(contractingAxes)-len(batchAxes))
	for axis, axisType := range axesTypes {
		dim := shape.Dimensions[axis]
		switch axisType {
		case 0: // Cross axes (unmarked)
			crossSize *= dim
			crossDims = append(crossDims, dim)
		case 1: // Contracting axes
			contractingSize *= dim
		case 2: // Batch axes
			batchSize *= dim
		}
	}
	return
}

func IsFinite(operand shapes.Shape) (output shapes.Shape, err error) {
	dtype := operand.DType
	if !dtype.IsFloat() {
		err = errors.Errorf("IsFinite: operand data type %s is a floating point type", dtype)
		return
	}
	output = operand.Clone()
	output.DType = dtypes.Bool
	return
}

// Reduce returns the operation's output shapes and checks all shapes and dtypes are valid.
// The axes are also normalized to positive in-place.
func Reduce(inputs, initialValues, reductionInputs, reductionOutputs []shapes.Shape, axes []int) (outputs []shapes.Shape, err error) {
	// Check inputs and initialValues.
	numReductions := len(inputs)
	if numReductions == 0 {
		return nil, errors.New("Reduce requires at least one input")
	}
	if len(initialValues) != numReductions {
		return nil, errors.Errorf("Reduce requires the same number of initial values as inputs, got %d initial values and %d inputs",
			len(initialValues), len(inputs))
	}
	baseDimensions := inputs[0].Dimensions
	for i, input := range inputs {
		if input.DType != initialValues[i].DType {
			return nil, errors.Errorf("Reduce requires the same dtype for initial values and inputs, got %s and %s for input #%d",
				initialValues[i].DType, input.DType, i)
		}
		if !slices.Equal(input.Dimensions, baseDimensions) {
			return nil, errors.Errorf("Reduce requires the same shape (dimensions only) for all inputs, got %s and %s for inputs #0 and #%d",
				inputs[0], input, i)
		}
	}

	// Check that all reduction inputs and outputs are valid.
	if len(reductionInputs) != 2*numReductions {
		return nil, errors.Errorf("The reduction function for the Reduce operation must have 2 inputs for each initialValue, but reduction has %d inputs for 2*%d=%d initial values",
			len(reductionInputs), len(initialValues), 2*len(initialValues))
	}
	if len(reductionOutputs) != numReductions {
		return nil, errors.Errorf("The reduction function for the Reduce operation must have 1 output for each initialValue, but reduction has %d outputs for %d initial values",
			len(reductionOutputs), len(initialValues))
	}
	for i := range numReductions {
		if reductionInputs[i].DType != reductionInputs[i+numReductions].DType || reductionInputs[i].DType != reductionOutputs[i].DType {
			return nil, errors.Errorf("Reduce requires the same dtype for lhs[i], rhs[i] inputs and output[i], got lhs[%d]=%s and rhs[%d+%d]=%s and output[%d]=%s",
				i, reductionInputs[i], i, numReductions, reductionInputs[i+numReductions], i, reductionOutputs[i])
		}
	}

	// Check the axis are valid.
	rank := inputs[0].Rank()
	if len(axes) > rank {
		return nil, errors.Errorf("input for Reduce has rank=%d, but %d axes for reduction were given", rank, len(axes))
	}
	axesSet := utils.MakeSet[int]()
	for i, axis := range axes {
		adjustedAxis, err := AdjustAxisToRank(axis, rank)
		if err != nil {
			return nil, errors.WithMessagef(err, "invalid value for axes[%d]=%d for Reduce, inputs[0].shape=%s)",
				i, axis, inputs[0])
		}
		if axesSet.Has(adjustedAxis) {
			return nil, errors.Errorf("duplicate value for axes[%d]=%d for Reduce, axes=%v)",
				i, axis, axes)
		}
		axesSet.Insert(adjustedAxis)
		axes[i] = adjustedAxis
	}

	// Build the output shapes.
	reducedDims := slices.Clone(inputs[0].Dimensions)
	var toAxis int
	for axis, dim := range reducedDims {
		if axesSet.Has(axis) {
			// This axis will be reduced, and it disappears from the output shape.
			continue
		}
		reducedDims[toAxis] = dim
		toAxis++
	}
	reducedDims = reducedDims[:toAxis]
	outputs = make([]shapes.Shape, len(inputs))
	for ii, outputBase := range reductionOutputs {
		outputs[ii] = shapes.Make(outputBase.DType, reducedDims...)
	}
	return
}

func BitcastConvert(operand shapes.Shape, targetDType dtypes.DType) (outputShape shapes.Shape, err error) {
	if operand.DType == dtypes.INVALID {
		return shapes.Invalid(), errors.New("BitcastConvert: operand data type is invalid")
	}
	sourceDType := operand.DType
	outputShape = operand.Clone()
	outputShape.DType = targetDType
	if sourceDType.Bits() == targetDType.Bits() {
		// No changes in shape.
		return
	}
	if sourceDType.Bits() > targetDType.Bits() {
		// Convert to a smaller data type, append to a new dimension.
		newDim := sourceDType.Bits() / targetDType.Bits()
		outputShape.Dimensions = append(outputShape.Dimensions, newDim)
		return
	}

	// Convert to a larger data type, shrink the last dimension.
	if outputShape.Dim(-1) != (targetDType.Bits()+sourceDType.Bits()-1)/sourceDType.Bits() {
		return shapes.Invalid(), errors.Errorf("BitcastConvert: cannot convert from %d x %s (%d bits) to %s (%d bits)",
			outputShape.Dim(-1), sourceDType, sourceDType.Bits(), targetDType, targetDType.Bits())
	}
	outputShape.Dimensions = outputShape.Dimensions[:len(outputShape.Dimensions)-1]
	return
}

func Pad(x, fill shapes.Shape, paddingStart, paddingEnd, paddingInterior []int) (outputShape shapes.Shape, err error) {
	if !x.Ok() || !fill.Ok() {
		return shapes.Invalid(), errors.Errorf("Pad: invalid input shapes %s and %s", x, fill)
	}
	if x.DType != fill.DType {
		return shapes.Invalid(), errors.Errorf("Pad: operand (%s) and padding value (%s) must have the same dtype", x, fill)
	}
	if !fill.IsScalar() {
		return shapes.Invalid(), errors.Errorf("Pad: padding value (%s) must be a scalar", fill)
	}
	rank := x.Rank()
	if len(paddingStart) != rank || len(paddingEnd) != rank || len(paddingInterior) != rank {
		return shapes.Invalid(), errors.Errorf("Pad: number of padding values (%d, %d, %d) must match input rank %d",
			len(paddingStart), len(paddingEnd), len(paddingInterior), rank)
	}

	// Check that interior padding values are non-negative.
	for axis := range rank {
		if paddingInterior[axis] < 0 {
			return shapes.Invalid(), errors.Errorf("Pad: interior padding values must be non-negative, got start=%d, end=%d, interior=%d for axis %d",
				paddingStart[axis], paddingEnd[axis], paddingInterior[axis], axis)
		}
	}

	// Calculate output dimensions.
	outputDims := make([]int, rank)
	for axis := 0; axis < rank; axis++ {
		inputDim := x.Dimensions[axis]
		if inputDim <= 1 {
			outputDims[axis] = paddingStart[axis] + paddingEnd[axis] + inputDim
		} else {
			outputDims[axis] = paddingStart[axis] + paddingEnd[axis] + inputDim + (inputDim-1)*paddingInterior[axis]
		}
	}
	return shapes.Make(x.DType, outputDims...), nil
}

func FFT(x shapes.Shape, fftType types.FFTType, fftLength []int) (output shapes.Shape, err error) {
	if !x.Ok() {
		return shapes.Invalid(), errors.Errorf("FFT: invalid input shape %s", x)
	}

	// Check the FFT lengths are valid and match the input rank.
	rank := x.Rank()
	if len(fftLength) > rank {
		return shapes.Invalid(), errors.Errorf("FFT: number of FFT lengths (%d) cannot exceed input rank (%d)", len(fftLength), rank)
	}
	for i, length := range fftLength {
		if length <= 0 {
			return shapes.Invalid(), errors.Errorf("FFT: fftLength[%d]=%d must be positive", i, length)
		}
	}

	// Check input dtype matches FFT type.
	switch fftType {
	case types.FFTForward, types.FFTInverse:
		if !x.DType.IsComplex() {
			return shapes.Invalid(), errors.Errorf("FFT: FFTForward and FFTInverse require complex input, got %s", x.DType)
		}
	case types.FFTForwardReal:
		if !x.DType.IsFloat() {
			return shapes.Invalid(), errors.Errorf("FFT: FFTForwardReal requires real (float) input, got %s", x.DType)
		}
	case types.FFTInverseReal:
		if !x.DType.IsComplex() {
			return shapes.Invalid(), errors.Errorf("FFT: FFTInverseReal requires complex input, got %s", x.DType)
		}
	default:
		return shapes.Invalid(), errors.Errorf("FFT: invalid FFT type %d", fftType)
	}

	// Calculate output shape:
	output = x.Clone()
	switch fftType {
	case types.FFTForward, types.FFTInverse:
		// Output shape is the same as input.
		return

	case types.FFTForwardReal:
		// Output is complex, with the last FFT dimension halved and rounded up.
		if len(fftLength) == 0 {
			return shapes.Invalid(), errors.New("FFT: FFTForwardReal requires at least one FFT length")
		}
		lastFFTDim := fftLength[len(fftLength)-1]
		output.Dimensions[output.Rank()-1] = lastFFTDim/2 + 1
		if x.DType == dtypes.Float32 {
			output.DType = dtypes.Complex64
		} else { // Float64
			output.DType = dtypes.Complex128
		}

	case types.FFTInverseReal:
		// Input must be complex with the last axis dimension being fftLength/2+1
		if len(fftLength) == 0 {
			return shapes.Invalid(), errors.New("FFT: FFTInverseReal requires at least one FFT length")
		}
		lastFFTDim := fftLength[len(fftLength)-1]
		if x.Dim(-1) != lastFFTDim/2+1 {
			return shapes.Invalid(), errors.Errorf("FFT: FFTInverseReal input dimension %d must be equal to fftLength/2+1=%d",
				x.Dim(-1), lastFFTDim/2+1)
		}
		output.Dimensions[output.Rank()-1] = lastFFTDim
		switch x.DType {
		case dtypes.Complex64:
			output.DType = dtypes.Float32
		case dtypes.Complex128:
			output.DType = dtypes.Float64
		default:
			return shapes.Invalid(), errors.Errorf("FFT: FFTInverseReal dtype not supported: %s", output.DType)
		}

	default:
		return shapes.Invalid(), errors.Errorf("FFT: FFTType=%s not supported", fftType)
	}
	return
}
