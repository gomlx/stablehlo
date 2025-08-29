package stablehlo

import (
	"github.com/gomlx/stablehlo/internal/optypes"
	"github.com/gomlx/stablehlo/shapeinference"
	"github.com/gomlx/stablehlo/types/shapes"
)

// addOp adds a new operation to the function.
func (f *Function) addOp(opType optypes.OpType, outputShape shapes.Shape, inputs ...*Value) *Value {
	inputShapes := make([]shapes.Shape, len(inputs))
	for i, input := range inputs {
		inputShapes[i] = input.shape
	}

	stmt := &Statement{
		OpType:  opType,
		Inputs:  inputs,
		Outputs: []*Value{f.newValue(outputShape)},
	}
	f.Statements = append(f.Statements, stmt)
	return stmt.Outputs[0]
}

// binaryOp adds a new binary operation to the function.
func (f *Function) binaryOp(op optypes.OpType, lhs, rhs *Value) (*Value, error) {
	outputShape, err := shapeinference.BinaryOp(op, lhs.shape, rhs.shape)
	if err != nil {
		return nil, err
	}
	return f.addOp(op, outputShape, lhs, rhs), nil
}

// Add adds two values together.
func (f *Function) Add(lhs, rhs *Value) (*Value, error) {
	return f.binaryOp(optypes.Add, lhs, rhs)
}
