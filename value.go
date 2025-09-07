package stablehlo

import (
	"fmt"
	"io"

	"github.com/gomlx/stablehlo/types/shapes"
)

// Value represents a value in a ToStableHLO program, like `%0` or `%arg0`.
// It has a name, shape and an optional descriptive name that can contain letters, digits and underscore.
type Value struct {
	id    int
	shape shapes.Shape
	name  string // Optional name composed of letters, digits and underscore
}

// Shape returns the shape of the value.
func (v *Value) Shape() shapes.Shape {
	return v.shape
}

// Write writes the value in ToStableHLO text format to the given writer.
func (v *Value) Write(w io.Writer, indentation string) error {
	_ = indentation
	if v.name != "" {
		_, err := fmt.Fprintf(w, "%%%s", v.name)
		return err
	}
	_, err := fmt.Fprintf(w, "%%%d", v.id)
	return err
}

// String implements fmt.Stringer.
func (v *Value) String() string {
	if v.name != "" {
		return "%" + v.name
	}
	return fmt.Sprintf("%%%d", v.id)
}

// NamedValue creates a new named value with the given shape.
// These are meant to be used as inputs for functions.
func NamedValue(name string, shape shapes.Shape) *Value {
	return &Value{
		shape: shape,
		name:  name,
		id:    -1,
	}
}
