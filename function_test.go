package stablehlo

import (
	"fmt"
	"testing"

	"github.com/gomlx/stablehlo/types/optypes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func must[T any](value T, err error) T {
	if err != nil {
		panic(err)
	}
	return value
}

func TestBuilder_Programs(t *testing.T) {
	t.Run("no inputs", func(t *testing.T) {
		b := New("test_program")
		fn := b.NewFunction("main", nil)
		c1 := must(fn.NewConstant(1.0))
		c2 := must(fn.NewConstant(2.0))
		sum := must(fn.AddOp(optypes.Add, c1, c2))
		fn.Return(sum)
		program := must(b.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		assert.Contains(t, program,
			`func.func @main() -> (tensor<f64>) {
  %0 = "stablehlo.constant"(){value = dense<1> : tensor<f64>} : () -> tensor<f64>
  %1 = "stablehlo.constant"(){value = dense<2> : tensor<f64>} : () -> tensor<f64>
  %2 = "stablehlo.add"(%0, %1) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  "func.return"(%2) : (tensor<f64>) -> ()
}`)
	})
}

func TestBuilder_Errors(t *testing.T) {
	t.Run("no main", func(t *testing.T) {
		b := New("test_program")
		fn := b.NewFunction("not_main", nil)
		c1 := must(fn.NewConstant(1.0))
		fn.Return(c1)
		_, err := b.Build()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "program must have a main function")
	})
}
