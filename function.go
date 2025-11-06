package stablehlo

import (
	"fmt"
	"io"
	"reflect"
	"strconv"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/stablehlo/internal/optypes"
	"github.com/gomlx/stablehlo/shapeinference"
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

	// values holds all the values (e.g., %0, %1, %arg0) created in the function's scope.
	values []*Value

	// Parent of a closure function. It is only set if the function is a closure, and it's the function that created it.
	Parent *Function

	// nextArgID is the next ID to be assigned to new input arguments.
	nextArgID int

	// nextTmpID is the next ID to be assigned to new intermediary values.
	nextTmpID int

	// nextClosureID is the next ID to be assigned to new closures.
	nextClosureID int

	// Returned indicates if the function has a return statement, so it can no longer be changed.
	Returned bool
}

// findRootFn returns the root function of a function tree.
//
// There are no cases where it is more than 1-level deep, but it would work for more.
func (fn *Function) findRootFn() *Function {
	rootFn := fn
	for rootFn.Parent != nil {
		rootFn = rootFn.Parent
	}
	return rootFn
}

// newValue creates a new value with the given shape and assigns it to the next available id.
func (fn *Function) newValue(shape shapes.Shape) (v *Value) {
	rootFn := fn.findRootFn()
	v = &Value{
		fn:    fn,
		name:  strconv.Itoa(rootFn.nextTmpID),
		shape: shape,
	}
	rootFn.nextTmpID++
	fn.values = append(fn.values, v)
	return v
}

// Input creates a new input parameter for a function.
//
// If creating multiple inputs (one at a time), the order matters, since during execution of a compiled function,
// the input parameters must be given in the same order they were created.
//
// These add to the inputs already created during the function creation.
//
// It picks a default unique name for the input parameter, you can also
// provide a name with NamedInput.
func (fn *Function) Input(shape shapes.Shape) *Value {
	rootFn := fn.findRootFn()
	value := fn.NamedInput(fmt.Sprintf("arg%d", rootFn.nextArgID), shape)
	rootFn.nextArgID++
	return value
}

// NamedInput creates a new input parameter for a function with the given name -- it
// must be a unique input name.
//
// The name is passed through ConvertToValidName, which converts any non-digit or ASCII letter to an underscore.
//
// Names with the format "%d" and "arg%d" are reserved for the default input parameters.
//
// Names are used in the StableHLO code and may be helpful for debugging, but
// otherwise have no impact.
func (fn *Function) NamedInput(name string, shape shapes.Shape) *Value {
	value := &Value{
		fn:    fn,
		name:  ConvertToValidName(name),
		shape: shape,
	}
	fn.Inputs = append(fn.Inputs, value)
	return value
}

// ConstantFromScalar creates a new constant statement and returns the resulting value.
func (fn *Function) ConstantFromScalar(value any) (*Value, error) {
	if fn.Returned {
		return nil, errors.Errorf("Function.Return already called for %q", fn.Name)
	}

	// The shape of the constant is inferred from the value.
	dtype := dtypes.FromAny(value)
	if dtype == dtypes.INVALID {
		return nil, errors.Errorf("unsupported constant value type %T", value)
	}
	shape := shapes.Make(dtype)
	t, err := newTensorLiteralFromFlatAndDimensions(value)
	if err != nil {
		return nil, err
	}
	c := &Statement{
		Builder:  fn.Builder,
		Function: fn,
		OpType:   optypes.Constant,
		Attributes: map[string]any{
			"value": t,
		},
		Outputs: []*Value{fn.newValue(shape)},
	}
	fn.Statements = append(fn.Statements, c)
	return c.Outputs[0], nil
}

// ConstantFromFlatAndDimensions creates a new constant statement from a flat slice with the raw values and the dimensions of the shape.
func (fn *Function) ConstantFromFlatAndDimensions(flat any, dimensions ...int) (*Value, error) {
	if fn.Returned {
		return nil, errors.Errorf("Function.Return already called for %q", fn.Name)
	}
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
		Builder:    fn.Builder,
		Function:   fn,
		OpType:     optypes.Constant,
		Attributes: make(map[string]any, 1),
		Outputs:    []*Value{fn.newValue(shape)},
	}
	var err error
	if shape.IsScalar() {
		c.Attributes["value"], err = newTensorLiteralFromFlatAndDimensions(flatV.Index(0).Interface())
	} else {
		c.Attributes["value"], err = newTensorLiteralFromFlatAndDimensions(flat, dimensions...)
	}
	if err != nil {
		return nil, err
	}
	fn.Statements = append(fn.Statements, c)
	return c.Outputs[0], nil
}

// Return adds a return statement to the function with the given return values.
// There must be at least one return value.
//
// There can be only one return statement from a Function, and it must be the last
// operation of a function.
func (fn *Function) Return(firstValue *Value, otherValues ...*Value) error {
	if fn.Returned {
		return errors.Errorf("Function.Return already called for %q", fn.Name)
	}
	fn.Returned = true
	allValues := make([]*Value, 1, len(otherValues)+1)
	allValues[0] = firstValue
	allValues = append(allValues, otherValues...)
	outputShapes := make([]shapes.Shape, len(allValues))
	for i, value := range allValues {
		if value.fn != fn {
			return errors.New("Function.Return given values that are not owned by the function")
		}
		outputShapes[i] = value.shape
	}
	fn.Outputs = outputShapes

	stmt := &Statement{
		Builder:  fn.Builder,
		Function: fn,
		OpType:   optypes.FuncReturn,
		Inputs:   allValues,
	}
	fn.Statements = append(fn.Statements, stmt)
	return nil
}

// Iota creates a constant of the given shape with increasing numbers (starting from 0)
// on the given axis. So Iota([2,2], 1) returns [[0 1][0 1]], while Iota([2,2], 0)
// returns [[0 0][1 1]].
func (fn *Function) Iota(shape shapes.Shape, axis int) (*Value, error) {
	op := optypes.Iota
	adjustedAxis, err := shapeinference.AdjustAxisToRank(axis, shape.Rank())
	if err != nil {
		return nil, errors.WithMessagef(err, "Iota axis is invalid for shape %s", shape)
	}
	stmt := fn.addOp(op, shape)
	stmt.Attributes = map[string]any{"iota_dimension": int64(adjustedAxis)}
	return stmt.Outputs[0], nil
}

// Closure creates an unnamed closure function that can be used as an argument to operations like
// Reduce, ReduceWindow, ScatterAndUpdate, etc.
//
// After created, the Closure should not be changed. But it can be used multiple times within the same parent function.
//
// The function body is defined by calling ops on the function object, as a usual Function object.
func (fn *Function) Closure() *Function {
	rootFn := fn.findRootFn()

	// the name gets overwritten in StableHLO code by the statement taking the closure as a parameter,
	// it's just for debugging purposes.
	name := fmt.Sprintf("closure%d", rootFn.nextClosureID)
	rootFn.nextClosureID++
	closureFn := fn.Builder.NewFunction(name)
	closureFn.Parent = fn
	return closureFn
}

// Write the function as StableHLO code, with the given indentation.
func (fn *Function) Write(writer io.Writer, indentation string) error {
	// Create the formatting w() and we() internal functions to facilitate handling error while generating the statement code.
	var err error
	w := func(format string, args ...any) {
		// Do nothing if an error was encountered earlier.
		if err != nil {
			// No op if an error was encountered earlier
			return
		}
		_, err = fmt.Fprintf(writer, format, args...)
	}
	we := func(e elementWriter, indentation string) {
		// Do nothing if an error was encountered earlier.
		if err != nil {
			// No op if an error was encountered earlier
			return
		}
		err = e.Write(writer, indentation)
	}
	nextIndent := indentation + IndentationStep

	// Now write the function code.
	normalFunction := fn.Parent == nil
	isClosure := fn.Parent != nil
	if normalFunction {
		w("%sfunc.func @%s(", indentation, fn.Name)
	} else if isClosure {
		w("(")
	}
	for i, input := range fn.Inputs {
		if i > 0 {
			w(", ")
		}
		we(input, nextIndent)
		w(": %s", input.shape.ToStableHLO())
	}

	if isClosure {
		w(") :\n")
	} else if normalFunction {
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
	}

	for _, stmt := range fn.Statements {
		we(stmt, nextIndent)
		w("\n")
	}

	if normalFunction {
		w("%s}", indentation)
	}
	return err
}
