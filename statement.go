package stablehlo

import (
	"fmt"
	"io"
	"math"

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
		w("{")
		first := true
		for key, value := range s.Attributes {
			if !first {
				w(", ")
			}
			first = false
			w("%s = %s", key, literalToStableHLO(value))
		}
		w("}")
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

type hasToStableHLO interface {
	ToStableHLO() string
}

// literalToStableHLO converts a literal value, usually used in attributes, to its ToStableHLO string representation.
func literalToStableHLO(attr any) string {
	switch v := attr.(type) {
	case string:
		return fmt.Sprintf("%q", v)
	case float32, float64:
		var f float64
		if f32, ok := v.(float32); ok {
			f = float64(f32)
		} else {
			f = v.(float64)
		}
		shape := shapes.Make(dtypes.FromAny(v))
		format := "dense<%g> : %s"
		if f == math.Trunc(f) {
			// f is an integer, make sure we add a decimal point.
			format = "dense<%.1f> : %s"
		}
		return fmt.Sprintf(format, v, shape.ToStableHLO())
	case int, int8, int16, int32, int64, uint8, uint16, uint32, uint64:
		dtype := dtypes.FromAny(v)
		return fmt.Sprintf("%d : %s", v, utils.DTypeToStableHLO(dtype))
	case bool:
		if v {
			return "true"
		}
		return "false"

	case hasToStableHLO:
		// For types that implement their own conversion to stablehlo, use that.
		return v.ToStableHLO()

	default:
		return fmt.Sprintf("Unknown literal type: %t %#v", v, v)
	}
}
