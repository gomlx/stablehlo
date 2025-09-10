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
	BitcastConvert
	BroadcastInDim
	Cbrt
	Ceil
	Clamp
	Compare
	Complex
	Concatenate
	Convert
	Convolution
	Cosine
	CountLeadingZeros
	Divide
	DotGeneral
	Erf
	Exponential
	ExponentialMinusOne
	Floor
	Gather
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
	Pad
	Popcnt
	Power
	Real
	Remainder
	Reduce
	Reshape
	RngBitGenerator
	RoundNearestAfz
	RoundNearestEven
	Rsqrt
	Scatter
	ShiftLeft
	ShiftRightArithmetic
	ShiftRightLogical
	Sign
	Sine
	Slice
	Sqrt
	Subtract
	Tan
	Tanh
	Transpose
	Xor

	// Here the ones not implemented yet, please add an issue in the repo if you need them.

	BatchNormForInference
	BatchNormForTraining
	BatchNormGradient
	Case
	Cholesky
	CollectiveBroadcast
	CollectivePermute
	Composite
	CustomCall
	DynamicBroadcastInDim
	DynamicConv
	DynamicGather
	DynamicIota
	DynamicPad
	DynamicReshape
	DynamicUpdateSlice
	Fft
	GetDimensionSize
	GetTupleElement
	If
	Infeed
	OptimizationBarrier
	Outfeed
	PartitionId
	Recv
	ReducePrecision
	ReduceScatter
	ReduceWindow
	Reverse
	Select
	SelectAndScatter
	Send
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
		FuncReturn: "stablehlo.return",
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
