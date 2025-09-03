package stablehlo

import (
	"fmt"
	"io"
	"reflect"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/stablehlo/internal/optypes"
	"github.com/gomlx/stablehlo/types/shapes"
	"github.com/pkg/errors"
)

// Function represents a `func.func` in ToStableHLO.
type Function struct {
	Builder *Builder

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

// NewInput creates a new input parameter for a function.
// The order matter, since during execution of a compiled function,
// the input parameters must be given in the same order they were created.
//
// These add to the inputs already created during the function creation.
//
// It picks a default unique name for the input parameter, you can also
// provide a name with NewNamedInput.
func (fn *Function) NewInput(shape shapes.Shape) *Value {
	return fn.NewNamedInput(fmt.Sprintf("input_%d", len(fn.Inputs)), shape)
}

// NewNamedInput creates a new input parameter for a function with the given name -- it
// must be a unique input name.
//
// Names are used in the StableHLO code and may be helpful for debugging, but
// otherwise have no impact.
func (fn *Function) NewNamedInput(name string, shape shapes.Shape) *Value {
	value := NamedValue(name, shape)
	fn.Inputs = append(fn.Inputs, value)
	return value
}

// ConstantFromScalar creates a new constant statement and returns the resulting value.
func (fn *Function) ConstantFromScalar(value any) (*Value, error) {
	// The shape of the constant is inferred from the value.
	dtype := dtypes.FromAny(value)
	if dtype == dtypes.INVALID {
		return nil, errors.Errorf("unsupported constant value type %T", value)
	}
	shape := shapes.Make(dtype)
	c := &Statement{
		OpType: optypes.Constant,
		Attributes: map[string]any{
			"value": newTensorLiteral(value),
		},
		Outputs: []*Value{fn.newValue(shape)},
	}
	fn.Statements = append(fn.Statements, c)
	return c.Outputs[0], nil
}

// ConstantFromFlatAndDimensions creates a new constant statement from a flat slice with the raw values and the dimensions of the shape.
func (fn *Function) ConstantFromFlatAndDimensions(flat any, dimensions ...int) (*Value, error) {
	flatV := reflect.ValueOf(flat)
	dtype := dtypes.FromGoType(flatV.Type().Elem())
	if dtype == dtypes.INVALID {
		return nil, errors.Errorf("unsupported constant flat values type %T -- expected a slice of a basic data type", flat)
	}
	shape := shapes.Make(dtype, dimensions...)
	if shape.Size() != flatV.Len() {
		return nil, errors.Errorf("flat values size %d doesn't match shape size %d (%s)", flatV.Len(), shape.Size(), shape)
	}
	c := &Statement{
		OpType: optypes.Constant,
		Attributes: map[string]any{
			"value": newTensorLiteral(flat, dimensions...),
		},
		Outputs: []*Value{fn.newValue(shape)},
	}
	fn.Statements = append(fn.Statements, c)
	return c.Outputs[0], nil
}

// Return adds a return statement to the function with the given return values.
// There must be at least one return value.
//
// There can be only one return statement from a Function, and it must be the last
// operation of a function.
func (fn *Function) Return(firstValue *Value, otherValues ...*Value) {
	allValues := make([]*Value, 1, len(otherValues)+1)
	allValues[0] = firstValue
	allValues = append(allValues, otherValues...)
	outputShapes := make([]shapes.Shape, len(allValues))
	for i, value := range allValues {
		fmt.Printf("%d: %s\n", i, value.shape)
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
