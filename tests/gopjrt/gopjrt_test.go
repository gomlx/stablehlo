package gopjrt

import (
	"flag"
	"fmt"
	"iter"
	"math"
	"reflect"
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

func pjrtClientsIterator(t *testing.T) iter.Seq2[string, *pjrt.Client] {
	return func(yield func(string, *pjrt.Client) bool) {
		for _, pluginName := range getPluginNames() {
			plugin, err := pjrt.GetPlugin(pluginName)
			require.NoError(t, err, "failed to load plugin %q", pluginName)
			client, err := plugin.NewClient(nil)
			require.NoError(t, err, "failed to create client for plugin %q", pluginName)
			done := yield(pluginName, client)
			require.NoError(t, client.Destroy())
			if done {
				return
			}
		}
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
		expectedShape, err := S.FromAnyValue(expected[i].Flat)
		require.NoErrorf(t, err, "failed to get shape for output #%d: %v", i, expected[i].Flat)
		dtype := expectedShape.DType
		fmt.Printf("\t - output #%d:\n\t   - Got: dims=%v, flat_values=%v\n", i, gotDims, gotFlat)
		fmt.Printf("\t   - Want(%s): dims=%v, flat_values=%v\n", dtype, expected[i].Dims, expected[i].Flat)
		require.NoErrorf(t, err, "failed to get buffer contents for output #%d, expected flat value %v", i, expected[i].Flat)
		require.Equalf(t, expected[i].Dims, gotDims, "output #%d dims don't match", i)
		switch dtype {
		case D.Float64, D.Float32:
			require.InDeltaSlicef(t, expected[i].Flat, gotFlat, 1e-4, "output #%d flat values don't match", i)
		default:
			require.Equalf(t, expected[i].Flat, gotFlat, "output #%d flat values don't match", i)
		}
	}
}

func TestUniqueOps(t *testing.T) {
	for pluginName, client := range pjrtClientsIterator(t) {
		t.Run(pluginName, func(t *testing.T) {
			testUniqueOps(t, client)
		})
	}
}

func testUniqueOps(t *testing.T, client *pjrt.Client) {
	t.Run("Constant", func(t *testing.T) {
		b := stablehlo.New(t.Name())
		fn := b.NewFunction("main")
		c1 := must(fn.NewScalarConstant(1.0))
		c2 := must(fn.NewScalarConstant(2.0))
		sum := must(fn.Add(c1, c2))
		fn.Return(sum)
		program := must(b.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		output := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{{[]float64{3}, nil}}, output)
	})

	t.Run("Complex", func(t *testing.T) {
		builder := stablehlo.New(t.Name())
		shape := S.Make(D.Float64)
		lhsV, rhsV := stablehlo.NamedValue("lhs", shape), stablehlo.NamedValue("rhs", shape)
		fn := builder.NewFunction("main", lhsV, rhsV)
		fn.Return(must(fn.Complex(lhsV, rhsV)))
		program := must(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		a := must(client.BufferFromHost().FromFlatDataWithDimensions([]float64{1.0}, nil).Done())
		b := must(client.BufferFromHost().FromFlatDataWithDimensions([]float64{-1.0}, nil).Done())
		output := compileAndExecute(t, client, program, a, b)
		requireBuffersEqual(t, []FlatAndDims{{[]complex128{1 - 1i}, nil}}, output)
	})

	t.Run("Clamp", func(t *testing.T) {
		builder := stablehlo.New(t.Name())
		minV, maxV := stablehlo.NamedValue("min", S.Make(D.Float32)), stablehlo.NamedValue("max", S.Make(D.Float32))
		xV := stablehlo.NamedValue("x", S.Make(D.Float32, 3))
		fn := builder.NewFunction("main", minV, xV, maxV)
		fn.Return(must(fn.Clamp(minV, xV, maxV)))
		program := must(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		min := must(client.BufferFromHost().FromFlatDataWithDimensions([]float32{-1.0}, nil).Done())
		max := must(client.BufferFromHost().FromFlatDataWithDimensions([]float32{1.0}, nil).Done())
		x := must(client.BufferFromHost().FromFlatDataWithDimensions([]float32{0.1, -2.2, 3.3}, []int{3}).Done())
		output := compileAndExecute(t, client, program, min, x, max)
		requireBuffersEqual(t, []FlatAndDims{{[]float32{0.1, -1, 1}, []int{3}}}, output)
	})

	t.Run("Iota", func(t *testing.T) {
		builder := stablehlo.New(t.Name())
		fn := builder.NewFunction("main")
		fn.Return(
			must(fn.Iota(S.Make(D.F32, 2, 2), 0)),
			must(fn.Iota(S.Make(D.F32, 2, 2), 1)),
			must(fn.Iota(S.Make(D.F32, 4), 0)),
		)
		program := must(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{0, 0, 1, 1}, []int{2, 2}},
			{[]float32{0, 1, 0, 1}, []int{2, 2}},
			{[]float32{0, 1, 2, 3}, []int{4}},
		}, outputs)
	})

	t.Run("Reshape", func(t *testing.T) {
		builder := stablehlo.New(t.Name())
		fn := builder.NewFunction("main")
		x := must(fn.Iota(S.Make(D.F32, 3, 2), 0))
		y := must(fn.Reshape(x, S.Make(D.F32, 2, 3)))
		fn.Return(y)
		program := must(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{0, 0, 1, 1, 2, 2}, []int{2, 3}},
		}, outputs)
	})

	t.Run("BroadcastInDim", func(t *testing.T) {
		builder := stablehlo.New(t.Name())
		fn := builder.NewFunction("main")
		x := must(fn.Iota(S.Make(D.F32, 3), 0))
		y := must(fn.BroadcastInDim(x, S.Make(D.F32, 2, 3), []int{1}))
		fn.Return(y)
		program := must(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{0, 1, 2, 0, 1, 2}, []int{2, 3}},
		}, outputs)
	})

	t.Run("BroadcastInDim<scalar>", func(t *testing.T) {
		builder := stablehlo.New(t.Name())
		fn := builder.NewFunction("main")
		x := must(fn.NewScalarConstant(float32(7.0)))
		y := must(fn.BroadcastInDim(x, S.Make(D.F32, 3), nil))
		fn.Return(y)
		program := must(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{7, 7, 7}, []int{3}},
		}, outputs)
	})

}

func TestBinaryOps(t *testing.T) {
	for pluginName, client := range pjrtClientsIterator(t) {
		t.Run(pluginName, func(t *testing.T) {
			testBinaryOps(t, client)
		})
	}
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
	t.Run("Atan2", func(t *testing.T) {
		testBinaryOp(t, "Atan2", (*stablehlo.Function).Atan2, D.Float32, []float32{3.0}, []float32{7.0},
			[]float32{float32(math.Atan2(3.0, 7.0))})
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

	t.Run("ShiftLeft", func(t *testing.T) {
		testBinaryOp(t, "ShiftLeft", (*stablehlo.Function).ShiftLeft, D.Uint32, []uint32{0b1}, []uint32{2}, []uint32{0b100})
	})

	t.Run("ShiftRightArithmetic", func(t *testing.T) {
		testBinaryOp(t, "ShiftRightArithmetic", (*stablehlo.Function).ShiftRightArithmetic, D.Int32, []int32{-8}, []int32{1}, []int32{-4})
	})

	t.Run("ShiftRightLogical", func(t *testing.T) {
		testBinaryOp(t, "ShiftRightLogical", (*stablehlo.Function).ShiftRightLogical, D.Uint32, []uint32{0b1100}, []uint32{2}, []uint32{0b11})
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

func TestCompare(t *testing.T) {
	for pluginName, client := range pjrtClientsIterator(t) {
		t.Run(pluginName, func(t *testing.T) {
			testCompare(t, client)
		})
	}
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

const pi32 = float32(math.Pi)

func TestUnaryOps(t *testing.T) {
	for pluginName, client := range pjrtClientsIterator(t) {
		t.Run(pluginName, func(t *testing.T) {
			testUnaryOps(t, client)
		})
	}
}

func testUnaryOps(t *testing.T, client *pjrt.Client) {
	testUnaryOp := func(t *testing.T, opName string,
		op func(fn *stablehlo.Function, x *stablehlo.Value) (*stablehlo.Value, error),
		dtype D.DType, input any, expected any) {
		builder := stablehlo.New(t.Name())
		shape := S.Make(dtype)
		inputV := stablehlo.NamedValue("input", shape)
		fn := builder.NewFunction("main", inputV)
		result := must(op(fn, inputV))
		fn.Return(result)
		program := must(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		a := must(client.BufferFromHost().FromFlatDataWithDimensions(input, []int{}).Done())
		output := compileAndExecute(t, client, program, a)
		requireBuffersEqual(t, []FlatAndDims{{expected, nil}}, output)
	}

	t.Run("Not_Bool", func(t *testing.T) {
		testUnaryOp(t, "Not", (*stablehlo.Function).Not, D.Bool, []bool{true}, []bool{false})
	})
	t.Run("Not_Uint8", func(t *testing.T) {
		testUnaryOp(t, "Not", (*stablehlo.Function).Not, D.Uint8, []uint8{128}, []uint8{127})
	})

	t.Run("Popcnt_Uint32", func(t *testing.T) {
		testUnaryOp(t, "Popcnt", (*stablehlo.Function).Popcnt, D.Uint32, []uint32{0b1011}, []uint32{3})
	})

	t.Run("CountLeadingZeros_Uint32", func(t *testing.T) {
		testUnaryOp(t, "CountLeadingZeros", (*stablehlo.Function).CountLeadingZeros, D.Uint32, []uint32{0b1}, []uint32{31})
	})

	t.Run("Erf_Float32", func(t *testing.T) {
		testUnaryOp(t, "Erf", (*stablehlo.Function).Erf, D.Float64, []float64{1.0}, []float64{
			float64(math.Erf(1))})
	})

	t.Run("Exponential_Float32", func(t *testing.T) {
		testUnaryOp(t, "Exponential", (*stablehlo.Function).Exponential, D.Float32, []float32{1.0},
			[]float32{float32(math.Exp(1))})
	})

	t.Run("ExponentialMinusOne_Float32", func(t *testing.T) {
		testUnaryOp(t, "ExponentialMinusOne", (*stablehlo.Function).ExponentialMinusOne, D.Float32, []float32{1.0},
			[]float32{float32(math.E - 1)})
	})

	t.Run("Log_Float32", func(t *testing.T) {
		testUnaryOp(t, "Log", (*stablehlo.Function).Log, D.Float32, []float32{2.7183}, []float32{1.0})
	})

	t.Run("LogPlusOne_Float32", func(t *testing.T) {
		testUnaryOp(t, "LogPlusOne", (*stablehlo.Function).LogPlusOne, D.Float32, []float32{1.7183}, []float32{1.0})
	})

	t.Run("Logistic_Float32", func(t *testing.T) {
		testUnaryOp(t, "Logistic", (*stablehlo.Function).Logistic, D.Float32, []float32{0.0}, []float32{0.5})
	})

	t.Run("Ceil_Float32", func(t *testing.T) {
		testUnaryOp(t, "Ceil", (*stablehlo.Function).Ceil, D.Float32, []float32{1.7}, []float32{2.0})
	})

	t.Run("Floor_Float32", func(t *testing.T) {
		testUnaryOp(t, "Floor", (*stablehlo.Function).Floor, D.Float32, []float32{1.7}, []float32{1.0})
	})

	t.Run("RoundNearestEven_Float32", func(t *testing.T) {
		testUnaryOp(t, "RoundNearestEven", (*stablehlo.Function).RoundNearestEven, D.Float32, []float32{2.5}, []float32{2.0})
	})
	t.Run("RoundNearestAfz_Float32", func(t *testing.T) {
		testUnaryOp(t, "RoundNearestAfz", (*stablehlo.Function).RoundNearestAfz, D.Float32, []float32{2.5}, []float32{3.0})
	})

	t.Run("Rsqrt_Float32", func(t *testing.T) {
		testUnaryOp(t, "Rsqrt", (*stablehlo.Function).Rsqrt, D.Float32, []float32{4.0}, []float32{0.5})
	})

	t.Run("Sqrt_Float32", func(t *testing.T) {
		testUnaryOp(t, "Sqrt", (*stablehlo.Function).Sqrt, D.Float32, []float32{4.0}, []float32{2.0})
	})

	t.Run("Cbrt_Float32", func(t *testing.T) {
		testUnaryOp(t, "Cbrt", (*stablehlo.Function).Cbrt, D.Float32, []float32{8.0}, []float32{2.0})
	})

	t.Run("Cosine_Float32", func(t *testing.T) {
		testUnaryOp(t, "Cosine", (*stablehlo.Function).Cosine, D.Float32, []float32{0.0}, []float32{1.0})
	})

	t.Run("Sine_Float32", func(t *testing.T) {
		testUnaryOp(t, "Sine", (*stablehlo.Function).Sine, D.Float32, []float32{0.0}, []float32{0.0})
	})

	t.Run("Tan_Float32", func(t *testing.T) {
		testUnaryOp(t, "Tan", (*stablehlo.Function).Tan, D.Float32, []float32{pi32 / 4}, []float32{1})
	})
	t.Run("Tanh_Float32", func(t *testing.T) {
		testUnaryOp(t, "Tanh", (*stablehlo.Function).Tanh, D.Float32, []float32{0.5},
			[]float32{float32(math.Tanh(0.5))})
	})

	t.Run("Abs_Float32", func(t *testing.T) {
		testUnaryOp(t, "Abs", (*stablehlo.Function).Abs, D.Float32, []float32{-3.0}, []float32{3.0})
	})

	t.Run("Negate_Float32", func(t *testing.T) {
		testUnaryOp(t, "Negate", (*stablehlo.Function).Negate, D.Float32, []float32{3.0}, []float32{-3.0})
	})

	t.Run("Sign_Float32", func(t *testing.T) {
		testUnaryOp(t, "Sign", (*stablehlo.Function).Sign, D.Float32, []float32{-3.0}, []float32{-1.0})
	})

	t.Run("Real_Complex64", func(t *testing.T) {
		testUnaryOp(t, "Real", (*stablehlo.Function).Real, D.Complex64, []complex64{complex(3.0, 4.0)}, []float32{3.0})
	})

	t.Run("Real_Complex128", func(t *testing.T) {
		testUnaryOp(t, "Real", (*stablehlo.Function).Real, D.Complex128, []complex128{complex(3.0, 4.0)}, []float64{3.0})
	})

	t.Run("Imag_Complex64", func(t *testing.T) {
		testUnaryOp(t, "Imag", (*stablehlo.Function).Imag, D.Complex64, []complex64{complex(3.0, 4.0)}, []float32{4.0})
	})

	t.Run("Imag_Complex128", func(t *testing.T) {
		testUnaryOp(t, "Imag", (*stablehlo.Function).Imag, D.Complex128, []complex128{complex(3.0, 4.0)}, []float64{4.0})
	})
}

func TestConstants(t *testing.T) {
	for pluginName, client := range pjrtClientsIterator(t) {
		t.Run(pluginName, func(t *testing.T) {
			testConstants(t, client)
		})
	}
}

func testConstants(t *testing.T, client *pjrt.Client) {
	testScalar := func(t *testing.T, scalar any) {
		builder := stablehlo.New(t.Name())
		fn := builder.Main()
		c, err := fn.NewScalarConstant(scalar)
		require.NoError(t, err)
		fn.Return(c)
		program := must(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		output := compileAndExecute(t, client, program)[0]
		gotFlat, gotDim, err := output.ToFlatDataAndDimensions()
		require.NoError(t, err)
		require.Len(t, gotDim, 0)
		gotScalar := reflect.ValueOf(gotFlat).Index(0).Interface()
		require.Equal(t, scalar, gotScalar)
	}

	t.Run("float32", func(t *testing.T) { testScalar(t, float32(3.0)) })
	t.Run("float64", func(t *testing.T) { testScalar(t, 1.234e-9) })
	t.Run("int64", func(t *testing.T) { testScalar(t, int64(-3)) })
	t.Run("uint8", func(t *testing.T) { testScalar(t, uint8(3)) })
	t.Run("bool-true", func(t *testing.T) { testScalar(t, true) })
	t.Run("bool-false", func(t *testing.T) { testScalar(t, false) })
	t.Run("complex64", func(t *testing.T) { testScalar(t, complex64(7-3i)) })
	t.Run("complex128", func(t *testing.T) { testScalar(t, complex64(-7+3i)) })

	testTensor := func(t *testing.T, flat any, dimensions ...int) {
		builder := stablehlo.New(t.Name())
		fn := builder.Main()
		c, err := fn.NewConstantFromFlat(flat, dimensions...)
		require.NoError(t, err)
		fn.Return(c)
		program := must(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		output := compileAndExecute(t, client, program)[0]
		gotFlat, gotDims, err := output.ToFlatDataAndDimensions()
		require.NoError(t, err)
		require.Equal(t, dimensions, gotDims)
		require.Equal(t, flat, gotFlat)
	}

	t.Run("1D-float32", func(t *testing.T) { testTensor(t, []float32{1, 2, 3, 5, 7}, 5) })
	t.Run("2D-complex64", func(t *testing.T) { testTensor(t, []complex64{1, 2, 3, 5i, 7i, 11i}, 2, 3) })
	t.Run("3D-bool", func(t *testing.T) { testTensor(t, []bool{false, true, false, true}, 2, 1, 2) })
}
