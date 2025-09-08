package stablehlo

import (
	"fmt"
	"io"

	"github.com/gomlx/stablehlo/types/shapes"
)

// Value represents a value in a StableHLO program, like `%0` or `%arg0`.
// These values can be inputs, outputs or intermediary values of functions.
//
// It is always associated with a function (where it's being used) and must be uniquely identified by a string with
// digits '0'-'9', 'A'-'Z', 'a'-'z' or '_'.
//
// For inlined functions (for instance, the one passed to a Reduce operation), the names cannot clash with the parent
// function name (!?). But the names can be reused in different inline functions.
//
// It also carries its shape information.
type Value struct {
	fn    *Function
	name  string
	shape shapes.Shape
}

// Shape returns the shape of the value.
func (v *Value) Shape() shapes.Shape {
	return v.shape
}

// Write writes the value in ToStableHLO text format to the given writer.
func (v *Value) Write(w io.Writer, indentation string) error {
	_ = indentation
	_, err := fmt.Fprintf(w, "%%%s", v.name)
	return err
}

// String implements fmt.Stringer.
func (v *Value) String() string {
	return "%" + v.name
}

// NamedValue creates a new named value with the given shape.
// These are meant to be used as inputs for functions.
func NamedValue(name string, shape shapes.Shape) *Value {
	return &Value{
		shape: shape,
		name:  name,
	}
}
