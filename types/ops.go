package types

import (
	"fmt"
	"slices"
)

// ComparisonType enum defined for the Compare op.
type ComparisonType int

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
