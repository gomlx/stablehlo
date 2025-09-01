package stablehlo

import (
	"fmt"
	"io"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/stablehlo/internal/optypes"
	"github.com/gomlx/stablehlo/types/shapes"
	"github.com/pkg/errors"
)

// Function represents a `func.func` in ToStableHLO.
type Function struct {
	// Name of the function. It should not include the "@" prefix.
	Name string

	// Inputs to the function.
	Inputs []*Value

	// Outputs types of the function.
	Outputs []shapes.Shape

	// Statements in the function body.
	Statements []*Statement

	// values holds all the values (e.g. %0, %1, %arg0) created in the function's scope.
	values []*Value

	// nextID is the next ID to be assigned to a new value.
	nextID int
}

// newValue creates a new value with the given shape and assigns it to the next available id.
func (fn *Function) newValue(shape shapes.Shape) *Value {
	v := &Value{
		id:    fn.nextID,
		shape: shape,
	}
	fn.nextID++
	fn.values = append(fn.values, v)
	return v
}

// NewConstant creates a new constant statement and returns the resulting value.
func (fn *Function) NewConstant(value any) (*Value, error) {
	// The shape of the constant is inferred from the value.
	dtype := dtypes.FromAny(value)
	if dtype == dtypes.INVALID {
		return nil, errors.Errorf("unsupported constant value type %T", value)
	}
	shape := shapes.Make(dtype)
	c := &Statement{
		OpType: optypes.Constant,
		Attributes: map[string]any{
			"value": value,
		},
		Outputs: []*Value{fn.newValue(shape)},
	}
	fn.Statements = append(fn.Statements, c)
	return c.Outputs[0], nil
}

// Return adds a return statement to the function with the given return values.
// There must be at least one return value.
func (fn *Function) Return(firstValue *Value, otherValues ...*Value) {
	allValues := make([]*Value, len(otherValues)+1)
	allValues[0] = firstValue
	allValues = append(allValues, otherValues...)

	outputShapes := make([]shapes.Shape, len(allValues))
	for i, value := range allValues {
		outputShapes[i] = value.shape
	}
	fn.Outputs = outputShapes

	stmt := &Statement{
		OpType: optypes.FuncReturn,
		Inputs: allValues,
	}
	fn.Statements = append(fn.Statements, stmt)
}

func (fn *Function) Write(writer io.Writer) error {
	var err error
	w := func(format string, args ...any) {
		if err != nil {
			// No op if an error was encountered earlier
			return
		}
		_, err = fmt.Fprintf(writer, format, args...)
	}
	we := func(e elementWriter) {
		if err != nil {
			// No op if an error was encountered earlier
			return
		}
		err = e.Write(writer)
	}

	w("func.func @%s(", fn.Name)
	for i, input := range fn.Inputs {
		if i > 0 {
			w(", ")
		}
		we(input)
		w(": %s", input.shape.ToStableHLO())
	}
	w(") -> ")
	if len(fn.Outputs) > 1 {
		w("(")
	}
	for i, output := range fn.Outputs {
		if i > 0 {
			w(", ")
		}
		w("%s", output.ToStableHLO())
	}
	if len(fn.Outputs) > 1 {
		w(")")
	}
	w(" {\n")

	for _, stmt := range fn.Statements {
		we(stmt)
		w("\n")
	}

	w("}")
	return err
}
