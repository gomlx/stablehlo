package types

import (
	"fmt"
	"slices"
	"strings"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/stablehlo/internal/utils"
)

// ComparisonType enum defined for the Compare op.
type ComparisonType int

//go:generate go tool enumer -type=ComparisonType ops.go

const (
	// CompareFloat are used for floating point comparisons.
	CompareFloat ComparisonType = iota

	// CompareTotalOrder version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
	CompareTotalOrder

	CompareSigned
	CompareUnsigned
)

// ToStableHLO returns the StableHLO representation of the comparison type.
func (c ComparisonType) ToStableHLO() string {
	switch c {
	case CompareFloat:
		return "#stablehlo<comparison_type FLOAT>"
	case CompareTotalOrder:
		return "#stablehlo<comparison_type TOTALORDER>"
	case CompareSigned:
		return "#stablehlo<comparison_type SIGNED>"
	case CompareUnsigned:
		return "#stablehlo<comparison_type UNSIGNED>"
	}
	return fmt.Sprintf("#stablehlo<comparison_type UNKNOWN %d>", c)
}

// ComparisonDirection enum defined for the Compare op.
type ComparisonDirection int

//go:generate go tool enumer -type=ComparisonDirection -trimprefix=Compare ops.go

const (
	CompareEQ ComparisonDirection = iota
	CompareGE
	CompareGT
	CompareLE
	CompareLT
	CompareNE
)

func (c ComparisonDirection) ToStableHLO() string {
	switch c {
	case CompareEQ:
		return "#stablehlo<comparison_direction EQ>"
	case CompareLE:
		return "#stablehlo<comparison_direction LE>"
	case CompareNE:
		return "#stablehlo<comparison_direction NE>"
	case CompareLT:
		return "#stablehlo<comparison_direction LT>"
	case CompareGT:
		return "#stablehlo<comparison_direction GT>"
	case CompareGE:
		return "#stablehlo<comparison_direction GE>"
	}
	return fmt.Sprintf("#stablehlo<comparison_direction UNKNOWN %d>", c)
}

// ConvolveAxesConfig defines the interpretation of the input/kernel/output tensor axes.
// There must be the same number of spatial dimensions (axes) for each of the 3 tensors.
// Input and output have batch and channel axes. Kernel has inputChannel and outputChannel axes.
//
// See Builder.ConvGeneral
type ConvolveAxesConfig struct {
	InputBatch, InputChannels int
	InputSpatial              []int

	KernelInputChannels, KernelOutputChannels int
	KernelSpatial                             []int

	OutputBatch, OutputChannels int
	OutputSpatial               []int
}

// Clone returns a deep copy of the structure.
func (c ConvolveAxesConfig) Clone() ConvolveAxesConfig {
	var c2 ConvolveAxesConfig
	c2 = c
	c2.InputSpatial = slices.Clone(c.InputSpatial)
	c2.KernelSpatial = slices.Clone(c.KernelSpatial)
	c2.OutputSpatial = slices.Clone(c.OutputSpatial)
	return c2
}

// DotGeneralPrecisionType defines the precision of the dot product.
//
// It controls the tradeoff between speed and accuracy for computations on accelerator backends.
// This can be one of the following (at the moment, the semantics of these enum values are underspecified,
// but they are planning to address this in #755 -- https://github.com/openxla/stablehlo/issues/755):
type DotGeneralPrecisionType int

//go:generate go tool enumer -type=DotGeneralPrecisionType -trimprefix=DotGeneralPrecision ops.go

const (
	// DotGeneralPrecisionDefault is the fastest calculation, but the least accurate approximation to the original number.
	DotGeneralPrecisionDefault DotGeneralPrecisionType = iota
	DotGeneralPrecisionHigh
	DotGeneralPrecisionHighest
)

func (p DotGeneralPrecisionType) ToStableHLO() string {
	return strings.ToUpper(p.String())
}

// FloatPrecisionType defines the precision used during floating point operations.
// In particular, modern GPUs accept the TF32 type which sacrifices some accuracy for
// significant speed improvements.
type FloatPrecisionType struct {
	// TF32 is used for the TF32 precision type.
	TF32 bool

	// DType is used for non-TF32 precision types.
	// It must be a float type.
	DType dtypes.DType
}

func (f FloatPrecisionType) ToStableHLO() string {
	if f.TF32 {
		return "tf32"
	}
	return utils.DTypeToStableHLO(f.DType)
}

// DotGeneralAlgorithm defines fine-control of the algorithm used for the dot product.
type DotGeneralAlgorithm struct {
	// LhsPrecisionType, RhsPrecisionType that the LHS and RHS of the operation are rounded to.
	// Precision types are independent of the storage types of the inputs and the output.
	LhsPrecisionType, RhsPrecisionType FloatPrecisionType

	// AccumulationType defines the type of the accumulator used for the dot product.
	AccumulationType FloatPrecisionType

	// LhsComponentCount, RhsComponentCount and NumPrimitiveOperations apply when we are doing an algorithm which
	// decomposes the LHS and/or RHS into multiple components and does multiple "primitive" dot operations on those values -
	// usually to emulate a higher precision (e.g.: Leveraging the bfloat16 Artificial Intelligence Datatype For
	// Higher-Precision Computations: bf16_6x tf32_3x -- https://arxiv.org/pdf/1904.06376, etc).
	// For algorithms with no decomposition, these values should be set to 1
	LhsComponentCount, RhsComponentCount, NumPrimitiveOperations int

	// AllowImpreciseAccumulation to specify if accumulation in lower precision is permitted for some steps
	// (e.g. CUBLASLT_MATMUL_DESC_FAST_ACCUM).
	AllowImpreciseAccumulation bool
}

// RngBitGeneratorAlgorithm used by the RngBitGenerator operation.
type RngBitGeneratorAlgorithm int

const (
	RngDefault RngBitGeneratorAlgorithm = iota
	RngPhilox
	RngThreeFry
)

//go:generate go tool enumer -type=RngBitGeneratorAlgorithm -trimprefix=Rng -transform=snake ops.go
