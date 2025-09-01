package stablehlo

import (
	"github.com/gomlx/stablehlo/internal/optypes"
	"github.com/gomlx/stablehlo/shapeinference"
	"github.com/gomlx/stablehlo/types/shapes"
)

// addOp adds a new operation to the function.
func (fn *Function) addOp(opType optypes.OpType, outputShape shapes.Shape, inputs ...*Value) *Statement {
	inputShapes := make([]shapes.Shape, len(inputs))
	for i, input := range inputs {
		inputShapes[i] = input.shape
	}

	stmt := &Statement{
		OpType:  opType,
		Inputs:  inputs,
		Outputs: []*Value{fn.newValue(outputShape)},
	}
	fn.Statements = append(fn.Statements, stmt)
	return stmt
}

// binaryOp adds a new binary operation to the function.
func (fn *Function) binaryOp(op optypes.OpType, lhs, rhs *Value) (*Value, error) {
	outputShape, err := shapeinference.BinaryOp(op, lhs.shape, rhs.shape)
	if err != nil {
		return nil, err
	}
	return fn.addOp(op, outputShape, lhs, rhs).Outputs[0], nil
}

// unaryOp adds a new unary operation to the function.
func (fn *Function) unaryOp(op optypes.OpType, operand *Value) (*Value, error) {
	outputShape, err := shapeinference.UnaryOp(op, operand.shape)
	if err != nil {
		return nil, err
	}
	return fn.addOp(op, outputShape, operand).Outputs[0], nil
}
