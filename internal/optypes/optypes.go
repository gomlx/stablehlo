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
	Atan2
	BroadcastInDim
	Cbrt
	Ceil
	Clamp
	Compare
	Complex
	Cosine
	CountLeadingZeros
	Divide
	DotGeneral
	Erf
	Exponential
	ExponentialMinusOne
	Floor
	Imag
	IsFinite
	Iota
	Log
	LogPlusOne
	Logistic
	Maximum
	Minimum
	Multiply
	Negate
	Not
	Or
	Popcnt
	Power
	Real
	Remainder
	Reshape
	RoundNearestAfz
	RoundNearestEven
	Rsqrt
	ShiftLeft
	ShiftRightArithmetic
	ShiftRightLogical
	Sign
	Sine
	Sqrt
	Subtract
	Tan
	Tanh
	Xor

	// Here the ones not implemented yet:

	ArgMinMax
	BatchNormForInference
	BatchNormForTraining
	BatchNormGradient
	BitcastConvert
	Case
	Cholesky
	CollectiveBroadcast
	CollectivePermute
	Composite
	Concatenate
	Convert
	Convolution
	CustomCall
	DynamicBroadcastInDim
	DynamicConv
	DynamicGather
	DynamicIota
	DynamicPad
	DynamicReshape
	DynamicUpdateSlice
	Fft
	Gather
	GetDimensionSize
	GetTupleElement
	If
	Infeed
	OptimizationBarrier
	Outfeed
	Pad
	PartitionId
	Recv
	Reduce
	ReducePrecision
	ReduceScatter
	ReduceWindow
	Reverse
	RngBitGenerator
	Scatter
	Select
	SelectAndScatter
	Send
	Slice
	Transpose
	TriangularSolve
	Tuple
	UniformDequantize
	UniformQuantize
	While

	// Last should always be kept the last, it is used as a counter/marker for .
	Last
)

var (
	// stableHLOMappings maps OpType to the corresponding StableHLO name, when the default
	// "snake case" doesn't work.
	stableHLOMappings = map[OpType]string{
		FuncReturn: "func.return",
		Erf:        "chlo.erf",
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
