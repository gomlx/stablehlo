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

	Abs
	Add
	And
	ArgMinMax
	BatchNormForInference
	BatchNormForTraining
	BatchNormGradient
	BitCount
	Bitcast
	Broadcast
	BroadcastInDim
	Ceil
	CountLeadingZeros
	Compare
	Conj
	Cosine
	Divide
	Erf
	Exponential
	ExponentialMinusOne
	Floor
	Imag
	Iota
	IsFinite
	Log
	Log1p
	Logistic
	Maximum
	Minimum
	Multiply
	Negate
	Not
	Or
	Pad
	Power
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
	ReduceWindow
	Remainder
	Reshape
	Reverse
	RngBitGenerator
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
	Sine
	Slice
	Sqrt
	Subtract
	Tanh
	Transpose
	Where
	Xor

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
