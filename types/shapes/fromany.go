package shapes

import (
	"fmt"
	"reflect"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/stablehlo/shapes"
)

// FromValue tries to infer the shape of a Go value.
// Multidimensional slices must be dense.
func FromValue(v any) (shape shapes.Shape, err error) {
	err = shapeForValueRecursive(&shape, reflect.ValueOf(v), reflect.TypeOf(v))
	return
}

func shapeForValueRecursive(shape *shapes.Shape, v reflect.Value, t reflect.Type) error {
	if t.Kind() == reflect.Slice {
		// Recurse into inner slices.
		t = t.Elem()
		shape.Dimensions = append(shape.Dimensions, v.Len())
		shapePrefix := shape.Clone()

		// The first element is the reference
		if v.Len() == 0 {
			return fmt.Errorf("value with empty slice not valid for Tensor conversion: %T: %v -- notice it's impossible to represent tensors with zero-dimensions generically using Go slices - try shapes.Make maybe ", v.Interface(), v)
		}
		v0 := v.Index(0)
		err := shapeForValueRecursive(shape, v0, t)
		if err != nil {
			return err
		}

		// Test that other elements have the same shape as the first one.
		for ii := 1; ii < v.Len(); ii++ {
			shapeTest := shapePrefix.Clone()
			err = shapeForValueRecursive(&shapeTest, v.Index(ii), t)
			if err != nil {
				return err
			}
			if !shape.Equal(shapeTest) {
				return fmt.Errorf("sub-slices have irregular shapes, found shapes %q, and %q", shape, shapeTest)
			}
		}
	} else if t.Kind() == reflect.Pointer {
		return fmt.Errorf("cannot convert Pointer (%s) to a concrete value for tensors", t)
	} else {
		shape.DType = dtypes.FromGoType(t)
		if shape.DType == dtypes.InvalidDType {
			return fmt.Errorf("cannot convert type %s to a value concrete tensor type (maybe type not supported yet?)", t)
		}
	}
	return nil
}
