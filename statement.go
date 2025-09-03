package stablehlo

import (
	"fmt"
	"io"
	"math"
	"reflect"
	"slices"
	"strconv"
	"strings"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/stablehlo/internal/optypes"
	"github.com/gomlx/stablehlo/internal/utils"
	"github.com/gomlx/stablehlo/types/shapes"
)

// Statement represents a single operation line in ToStableHLO.
type Statement struct {
	// OpType is the type of the operation.
	OpType optypes.OpType

	// Inputs to the operation.
	Inputs []*Value

	// Attributes of the operation.
	Attributes map[string]any

	// Outputs of the operation. It may be nil for operations like func.return.
	Outputs []*Value
}

// Write writes a string representation of the statement to the given writer.
func (s *Statement) Write(writer io.Writer) error {
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

	// Output values are written first:
	w("  ") // Indentation of functions.
	if len(s.Outputs) > 0 {
		for i, output := range s.Outputs {
			if i > 0 {
				w(", ")
			}
			we(output)
		}
		w(" = ")
	}

	// Write op name and arguments:
	w("%q(", s.OpType.ToStableHLO())
	for i, input := range s.Inputs {
		if i > 0 {
			w(", ")
		}
		we(input)
	}
	w(")")

	// Write attributes:
	if len(s.Attributes) > 0 {
		if len(s.Attributes) == 1 {
			for key, value := range s.Attributes {
				literalValue := literalToStableHLO(value)
				if strings.Index(literalValue, "\n") == -1 {
					w(" { %s = %s }", key, literalToStableHLO(value))
				} else {
					w(" {\n    %s = %s\n  }", key, literalToStableHLO(value))
				}
			}
		} else {
			// One attribute per line:
			w(" {")
			first := true
			for key, value := range s.Attributes {
				if !first {
					w(",")
				}
				first = false
				w("\n    %s = %s", key, literalToStableHLO(value))
			}
			w("\n  }")
		}
	}

	// Write signature:
	w(" : (")
	for i, input := range s.Inputs {
		if i > 0 {
			w(", ")
		}
		w(input.shape.ToStableHLO())
	}
	w(")")
	w(" -> ")
	if len(s.Outputs) == 0 {
		w("()")
	} else {
		// There are outputs: we use "(" and ")" only if there are more than one.
		if len(s.Outputs) > 1 {
			w("(")
		}
		for i, output := range s.Outputs {
			if i > 0 {
				w(", ")
			}
			w(output.shape.ToStableHLO())
		}
		if len(s.Outputs) > 1 {
			w(")")
		}
	}

	return err
}

// hasToStableHLO is implemented by types that can be converted to a stablehlo string.
type hasToStableHLO interface {
	ToStableHLO() string
}

// literalStr represents a value already rendered in StableHLO format.
type literalStr string

// ToStableHLO returns the string representation of the literal.
func (str literalStr) ToStableHLO() string {
	return string(str)
}

// literalToStableHLO converts a literal value, usually used in attributes, to its ToStableHLO string representation.
func literalToStableHLO(attr any) string {
	switch v := attr.(type) {
	case string:
		return fmt.Sprintf("%q", v)
	case bool, float32, float64, int, int8, int16, int32, int64, uint8, uint16, uint32, uint64:
		dtype := dtypes.FromAny(v)
		return fmt.Sprintf("%s : %s",
			podToStableHLO(v),
			utils.DTypeToStableHLO(dtype))

	case hasToStableHLO:
		// For types that implement their own conversion to stablehlo, use that.
		return v.ToStableHLO()

	default:
		return fmt.Sprintf("Unknown literal type: %t %#v", v, v)
	}
}

// intSliceToStableHLO converts a slice of ints to a string with comma-separated values, as used
// by StableHLO for attribute values that are an array of ints.
func intSliceToStableHLO(ints []int) literalStr {
	str := fmt.Sprint(ints) // Produces "[1 2 3]"
	return literalStr(strings.Replace(str, " ", ", ", -1))
}

// intSliceToArrayI64StableHLO converts a slice of ints to a string with comma-separated values, as used
// by StableHLO for attribute values that are an array of int64.
func intSliceToArrayI64StableHLO(ints []int) literalStr {
	var sb strings.Builder
	sb.WriteString("array<i64")
	for i, v := range ints {
		if i == 0 {
			sb.WriteString(": ")
		} else {
			sb.WriteString(", ")
		}
		sb.WriteString(strconv.Itoa(v))
	}
	sb.WriteString(">")
	return literalStr(sb.String())
}

// floatToStableHLO converts a float to a string. f must be a float32 or float64.
func floatToStableHLO(fAny any) string {
	var f float64
	if f32, ok := fAny.(float32); ok {
		f = float64(f32)
	} else {
		f = fAny.(float64)
	}
	format := "%g"
	if f == math.Trunc(f) {
		// f is an integer, make sure we add a decimal point.
		format = "%.1f"
	}
	return fmt.Sprintf(format, f)
}

// podToStableHLO convert a POD (plain-old-data) value (scalar floats, ints, bool and complex) to a stableHLO string,
// with no types attached.
func podToStableHLO(pod any) string {
	switch v := pod.(type) {
	case float32, float64:
		return floatToStableHLO(v)

	case int, int8, int16, int32, int64, uint8, uint16, uint32, uint64:
		return fmt.Sprintf("%d", v)

	case bool:
		if v {
			return "true"
		}
		return "false"

	case complex64, complex128:
		var c complex128
		if c64, ok := v.(complex64); ok {
			c = complex128(c64)
		} else {
			c = v.(complex128)
		}
		return fmt.Sprintf("(%s, %s)",
			floatToStableHLO(real(c)), floatToStableHLO(imag(c)))

	default:
		return fmt.Sprintf("*don't know how to present data type*: %t %#v", v, v)
	}
}

// tensorLiteral represents a literal tensor value, used to define constants.
//
// It has a different representation than other literals.
type tensorLiteral struct {
	// value is either a scalar value or a flat slice of the values.
	value any

	// dims has the dimensions of the tensor or nil if the value is a scalar.
	dims []int
}

// newTensorLiteral creates a new tensorLiteral that can be used to render constants.
//
// Args:
// - value is either a scalar value or a flat slice of the values.
// - dims has the dimensions of the tensor or nil if the value is a scalar.
func newTensorLiteral(value any, dims ...int) tensorLiteral {
	return tensorLiteral{value: value, dims: dims}
}

// ToStableHLO returns the string representation of the tensor literal.
func (t tensorLiteral) ToStableHLO() string {
	valueV := reflect.ValueOf(t.value)
	var shape shapes.Shape
	if valueV.Kind() != reflect.Slice {
		// Scalar value:
		shape.DType = dtypes.FromGoType(valueV.Type())
		return fmt.Sprintf("dense<%s> : %s", podToStableHLO(t.value), shape.ToStableHLO())
	}

	shape.DType = dtypes.FromGoType(valueV.Type().Elem())
	shape.Dimensions = slices.Clone(t.dims)
	var flatIdx int
	var sb strings.Builder
	recursiveTensorToStableHLO(valueV, shape, flatIdx, 0, &sb)
	return fmt.Sprintf("dense<%s> : %s", sb.String(), shape.ToStableHLO())
}

func recursiveTensorToStableHLO(valueV reflect.Value, shape shapes.Shape, flatIdx, axis int, sb *strings.Builder) int {
	sb.WriteString("[")
	if axis == shape.Rank()-1 {
		// Case 1: the last axis we actually print the values.
		for axisIdx := range shape.Dimensions[axis] {
			if axisIdx > 0 {
				sb.WriteString(", ")
			}
			sb.WriteString(podToStableHLO(valueV.Index(flatIdx).Interface()))
			flatIdx++
		}

	} else {
		// Case 2: we recursively print the sub-tensors.
		for axisIdx := range shape.Dimensions[axis] {
			if axisIdx > 0 {
				sb.WriteString(", ")
			}
			flatIdx = recursiveTensorToStableHLO(valueV, shape, flatIdx, axis+1, sb)
		}
	}
	sb.WriteString("]")
	return flatIdx
}
