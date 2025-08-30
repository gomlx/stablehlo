// Package optypes defines OpType and lists the supported operations.
package optypes

import (
	"fmt"

	"github.com/gomlx/stablehlo/internal/utils"
)

// OpType is an enum of all generic operations that ToStableHLO can support -- not all are implemented yet.
type OpType int

//go:generate go tool enumer -type=OpType optypes.go

const (
	Invalid OpType = iota
	FuncReturn
	Constant
	Identity

	ReduceWindow
	RngBitGenerator
	BatchNormForInference
	BatchNormForTraining
	BatchNormGradient
	BitCount

	Abs
	Add
	ArgMinMax
	Bitcast
	BitwiseAnd
	BitwiseNot
	BitwiseOr
	BitwiseXor
	Broadcast
	BroadcastInDim
	Ceil
	Clz
	Complex
	Concatenate
	Conj
	ConvGeneralDilated
	ConvertDType
	Cos
	Div
	Dot
	DotGeneral
	DynamicSlice
	DynamicUpdateSlice
	Equal
	EqualTotalOrder
	Erf
	Exp
	Expm1
	FFT
	Floor
	Gather
	GreaterOrEqual
	GreaterOrEqualTotalOrder
	GreaterThan
	GreaterThanTotalOrder
	Imag
	Iota
	IsFinite
	LessOrEqual
	LessOrEqualTotalOrder
	LessThan
	LessThanTotalOrder
	Log
	Log1p
	LogicalAnd
	LogicalNot
	LogicalOr
	LogicalXor
	Logistic
	Max
	Min
	Mul
	Negate
	NotEqual
	NotEqualTotalOrder
	Pad
	Pow
	Real
	ReduceBitwiseAnd
	ReduceBitwiseOr
	ReduceBitwiseXor
	ReduceLogicalAnd
	ReduceLogicalOr
	ReduceLogicalXor
	ReduceMax
	ReduceMin
	ReduceProduct
	ReduceSum
	Rem
	Reshape
	Reverse
	Round
	Rsqrt
	ScatterMax
	ScatterMin
	ScatterSum
	SelectAndScatterMax
	SelectAndScatterMin
	SelectAndScatterSum
	ShiftLeft
	ShiftRightArithmetic
	ShiftRightLogical
	Sign
	Sin
	Slice
	Sqrt
	Sub
	Tanh
	Transpose
	Where

	// Last should always be kept the last, it is used as a counter/marker for .
	Last
)

var (
	// stableHLOMappings maps OpType to the corresponding StableHLO name, when the default
	// "snake case" doesn't work.
	stableHLOMappings = map[OpType]string{
		FuncReturn: "func.return",
	}
)

// ToStableHLO returns the ToStableHLO name of the operation.
func (op OpType) ToStableHLO() string {
	name, ok := stableHLOMappings[op]
	if !ok {
		name = fmt.Sprintf("stablehlo.%s", utils.ToSnakeCase(op.String()))
	}
	return name
}
