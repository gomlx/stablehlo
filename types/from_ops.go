package types

import "slices"

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
