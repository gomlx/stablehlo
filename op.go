package stablehlo

import (
	"github.com/gomlx/stablehlo/types/optypes"
	"github.com/gomlx/stablehlo/types/shapes"
)

// Op holds an operation definition with all its inputs, and the result name and shape.
type Op struct {
	opType optypes.OpType
	inputs []*Op
	shape  shapes.Shape
}
