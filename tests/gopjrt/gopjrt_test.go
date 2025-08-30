package gopjrt

import (
	"flag"
	"fmt"
	"strings"
	"testing"

	D "github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/pjrt"
	"github.com/gomlx/stablehlo"
	"github.com/gomlx/stablehlo/types"
	S "github.com/gomlx/stablehlo/types/shapes"
	"github.com/stretchr/testify/require"
)

var flagPluginNames = flag.String("plugins", "cpu", "List (|-separated) of PRJT plugin names or full paths. E.g. \"cpu|cuda\"")

func must[T any](value T, err error) T {
	if err != nil {
		panic(err)
	}
	return value
}

func getPluginNames() []string {
	names := strings.Split(*flagPluginNames, "|")
	var to int
	for _, name := range names {
		if name != "" {
			names[to] = name
			to++
		}
	}
	if to == 0 {
		panic("no XLA plugin names defined with -plugins")
	}
	names = names[:to]
	return names
}

func TestRun(t *testing.T) {
	for _, pluginName := range getPluginNames() {
		plugin, err := pjrt.GetPlugin(pluginName)
		require.NoError(t, err, "failed to load plugin %q", pluginName)
		client, err := plugin.NewClient(nil)
		require.NoError(t, err, "failed to create client for plugin %q", pluginName)
		t.Run(pluginName, func(t *testing.T) {
			t.Run("SpecialOps", func(t *testing.T) {
				testSpecialOps(t, client)
			})
			t.Run("BinaryOps", func(t *testing.T) {
				testBinaryOps(t, client)
			})
			t.Run("Compare", func(t *testing.T) {
				testCompare(t, client)
			})
		})
		require.NoError(t, client.Destroy())
	}
}

// compileAndExecute program with PJRT. All inputs are donated.
func compileAndExecute(t *testing.T, client *pjrt.Client, program []byte, inputs ...*pjrt.Buffer) []*pjrt.Buffer {
	loadedExec, err := client.Compile().WithStableHLO(program).Done()
	require.NoErrorf(t, err, "failed to compile program: \n%s", program)
	defer func() {
		err := loadedExec.Destroy()
		if err != nil {
			t.Errorf("failed to destroy loaded exec: %+v", err)
		}
	}()
	outputBuffers, err := loadedExec.Execute(inputs...).DonateAll().Done()
	require.NoErrorf(t, err, "failed to execute program: \n%s", program)
	return outputBuffers
}

type FlatAndDims struct {
	Flat any
	Dims []int
}

// requireBuffersEqual checks that the actual buffers contents match the expected flat values.
// It destroys the buffers.
func requireBuffersEqual(t *testing.T, expected []FlatAndDims, got []*pjrt.Buffer) {
	defer func() {
		for _, b := range got {
			err := b.Destroy()
			if err != nil {
				t.Errorf("failed to destroy buffer: %+v", err)
			}
		}
	}()

	require.Len(t, got, len(expected))
	for i, b := range got {
		gotFlat, gotDims, err := b.ToFlatDataAndDimensions()
		fmt.Printf("\t - output #%d: dims=%v, flat_values=%v\n", i, gotDims, gotFlat)
		require.NoErrorf(t, err, "failed to get buffer contents for output #%d, expected flat value %v", i, expected[i].Flat)
		require.Equalf(t, expected[i].Dims, gotDims, "output #%d dims don't match", i)
		require.Equalf(t, expected[i].Flat, gotFlat, "output #%d flat values don't match", i)
	}
}

func testSpecialOps(t *testing.T, client *pjrt.Client) {
	t.Run("Constant", func(t *testing.T) {
		b := stablehlo.New(t.Name())
		fn := b.NewFunction("main")
		c1 := must(fn.NewConstant(1.0))
		c2 := must(fn.NewConstant(2.0))
		sum := must(fn.Add(c1, c2))
		fn.Return(sum)
		program := must(b.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		output := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{{[]float64{3}, nil}}, output)
	})
}

func testBinaryOps(t *testing.T, client *pjrt.Client) {
	testBinaryOp := func(t *testing.T, opName string,
		op func(fn *stablehlo.Function, lhs, rhs *stablehlo.Value) (*stablehlo.Value, error),
		dtype D.DType, lhs, rhs any, expected any) {
		builder := stablehlo.New(t.Name())
		shape := S.Make(dtype)
		lhsV, rhsV := stablehlo.NamedValue("lhs", shape), stablehlo.NamedValue("rhs", shape)
		fn := builder.NewFunction("main", lhsV, rhsV)
		result := must(op(fn, lhsV, rhsV))
		fn.Return(result)
		program := must(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		a := must(client.BufferFromHost().FromFlatDataWithDimensions(lhs, []int{}).Done())
		b := must(client.BufferFromHost().FromFlatDataWithDimensions(rhs, []int{}).Done())
		output := compileAndExecute(t, client, program, a, b)
		requireBuffersEqual(t, []FlatAndDims{{expected, nil}}, output)
	}

	t.Run("Add", func(t *testing.T) {
		testBinaryOp(t, "Add", (*stablehlo.Function).Add, D.Float32, []float32{3.0}, []float32{7.0}, []float32{10.0})
	})

	t.Run("Subtract", func(t *testing.T) {
		testBinaryOp(t, "Subtract", (*stablehlo.Function).Subtract, D.Float32, []float32{7.0}, []float32{3.0}, []float32{4.0})
	})

	t.Run("Multiply", func(t *testing.T) {
		testBinaryOp(t, "Multiply", (*stablehlo.Function).Multiply, D.Float32, []float32{3.0}, []float32{4.0}, []float32{12.0})
	})

	t.Run("Divide", func(t *testing.T) {
		testBinaryOp(t, "Divide", (*stablehlo.Function).Divide, D.Float32, []float32{12.0}, []float32{3.0}, []float32{4.0})
	})

	t.Run("Power", func(t *testing.T) {
		testBinaryOp(t, "Power", (*stablehlo.Function).Power, D.Float32, []float32{2.0}, []float32{3.0}, []float32{8.0})
	})

	t.Run("And_Uint32", func(t *testing.T) {
		testBinaryOp(t, "And", (*stablehlo.Function).And, D.Uint32, []uint32{0b1100}, []uint32{0b1010}, []uint32{0b1000})
	})

	t.Run("Or_Uint32", func(t *testing.T) {
		testBinaryOp(t, "Or", (*stablehlo.Function).Or, D.Uint32, []uint32{0b1100}, []uint32{0b1010}, []uint32{0b1110})
	})

	t.Run("Xor_Uint32", func(t *testing.T) {
		testBinaryOp(t, "Xor", (*stablehlo.Function).Xor, D.Uint32, []uint32{0b1100}, []uint32{0b1010}, []uint32{0b0110})
	})

	t.Run("And_Bool", func(t *testing.T) {
		testBinaryOp(t, "And", (*stablehlo.Function).And, D.Bool, []bool{true}, []bool{false}, []bool{false})
	})

	t.Run("Or_Bool", func(t *testing.T) {
		testBinaryOp(t, "Or", (*stablehlo.Function).Or, D.Bool, []bool{true}, []bool{false}, []bool{true})
	})

	t.Run("Xor_Bool", func(t *testing.T) {
		testBinaryOp(t, "Xor", (*stablehlo.Function).Xor, D.Bool, []bool{true}, []bool{false}, []bool{true})
	})

	t.Run("Reminder", func(t *testing.T) {
		testBinaryOp(t, "Remainder", (*stablehlo.Function).Remainder, D.Float32, []float32{7.0}, []float32{4.0}, []float32{3.0})
	})

	t.Run("Maximum", func(t *testing.T) {
		testBinaryOp(t, "Maximum", (*stablehlo.Function).Maximum, D.Float32, []float32{3.0}, []float32{7.0}, []float32{7.0})
	})

	t.Run("Minimum", func(t *testing.T) {
		testBinaryOp(t, "Minimum", (*stablehlo.Function).Minimum, D.Float32, []float32{3.0}, []float32{7.0}, []float32{3.0})
	})
}

func testCompare(t *testing.T, client *pjrt.Client) {
	runTest := func(t *testing.T, opName string,
		direction types.ComparisonDirection, compareType types.ComparisonType,
		dtype D.DType, lhs, rhs any, expected any) {
		builder := stablehlo.New(t.Name())
		shape := S.Make(dtype)
		lhsV, rhsV := stablehlo.NamedValue("lhs", shape), stablehlo.NamedValue("rhs", shape)
		fn := builder.NewFunction("main", lhsV, rhsV)
		result := must(fn.Compare(lhsV, rhsV, direction, compareType))
		fn.Return(result)
		program := must(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		a := must(client.BufferFromHost().FromFlatDataWithDimensions(lhs, []int{}).Done())
		b := must(client.BufferFromHost().FromFlatDataWithDimensions(rhs, []int{}).Done())
		output := compileAndExecute(t, client, program, a, b)
		requireBuffersEqual(t, []FlatAndDims{{expected, nil}}, output)
	}

	t.Run("Float_EQ", func(t *testing.T) {
		runTest(t, "Compare", types.CompareEQ, types.CompareFloat,
			D.Float32, []float32{3.0}, []float32{3.0}, []bool{true})
	})

	t.Run("Signed_GT", func(t *testing.T) {
		runTest(t, "Compare", types.CompareGT, types.CompareSigned,
			D.Int32, []int32{7}, []int32{3}, []bool{true})
	})

	t.Run("Unsigned_LT", func(t *testing.T) {
		runTest(t, "Compare", types.CompareLT, types.CompareUnsigned,
			D.Uint32, []uint32{3}, []uint32{7}, []bool{true})
	})

	t.Run("TotalOrder_GE", func(t *testing.T) {
		runTest(t, "Compare", types.CompareGE, types.CompareTotalOrder,
			D.Float32, []float32{3.0}, []float32{3.0}, []bool{true})
	})

	t.Run("Float_NE", func(t *testing.T) {
		runTest(t, "Compare", types.CompareNE, types.CompareFloat,
			D.Float32, []float32{3.0}, []float32{7.0}, []bool{true})
	})

	t.Run("Signed_LE", func(t *testing.T) {
		runTest(t, "Compare", types.CompareLE, types.CompareSigned,
			D.Int32, []int32{3}, []int32{7}, []bool{true})
	})
}
