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
func (f *Function) newValue(shape shapes.Shape) *Value {
	v := &Value{
		id:    f.nextID,
		shape: shape,
	}
	f.nextID++
	f.values = append(f.values, v)
	return v
}

// NewConstant creates a new constant statement and returns the resulting value.
func (f *Function) NewConstant(value any) (*Value, error) {
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
		Outputs: []*Value{f.newValue(shape)},
	}
	f.Statements = append(f.Statements, c)
	return c.Outputs[0], nil
}

// Return adds a return statement to the function with the given return values.
// There must be at least one return value.
func (f *Function) Return(firstValue *Value, otherValues ...*Value) {
	allValues := make([]*Value, len(otherValues)+1)
	allValues[0] = firstValue
	allValues = append(allValues, otherValues...)

	outputShapes := make([]shapes.Shape, len(allValues))
	for i, value := range allValues {
		outputShapes[i] = value.shape
	}
	f.Outputs = outputShapes

	stmt := &Statement{
		OpType: optypes.FuncReturn,
		Inputs: allValues,
	}
	f.Statements = append(f.Statements, stmt)
}

func (f *Function) Write(writer io.Writer) error {
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

	w("func.func @%s(", f.Name)
	for i, input := range f.Inputs {
		if i > 0 {
			w(", ")
		}
		we(input)
		w(": %s", input.shape.ToStableHLO())
	}
	w(") -> ")
	if len(f.Outputs) > 1 {
		w("(")
	}
	for i, output := range f.Outputs {
		if i > 0 {
			w(", ")
		}
		w("%s", output.ToStableHLO())
	}
	if len(f.Outputs) > 1 {
		w(")")
	}
	w(" {\n")

	for _, stmt := range f.Statements {
		we(stmt)
		w("\n")
	}

	w("}")
	return err
}

// scalarShapeForValue is a local helper to get the shape for a scalar value.
func scalarShapeForValue(value any) (shapes.Shape, error) {
	var dtype dtypes.DType
	switch value.(type) {
	case bool:
		dtype = dtypes.Bool
	case int:
		dtype = dtypes.Int64 // Assume int is 64-bit.
	case int8:
		dtype = dtypes.S8
	case int16:
		dtype = dtypes.S16
	case int32:
		dtype = dtypes.S32
	case int64:
		dtype = dtypes.S64
	case uint8:
		dtype = dtypes.U8
	case uint16:
		dtype = dtypes.U16
	case uint32:
		dtype = dtypes.U32
	case uint64:
		dtype = dtypes.U64
	case float32:
		dtype = dtypes.F32
	case float64:
		dtype = dtypes.F64
	default:
		return shapes.Shape{}, errors.Errorf("unsupported scalar value type %T", value)
	}
	return shapes.Make(dtype), nil
}
