package gopjrt

import (
	"flag"
	"fmt"
	"iter"
	"math"
	"math/bits"
	"reflect"
	"strings"
	"testing"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/pjrt"
	. "github.com/gomlx/stablehlo"
	"github.com/gomlx/stablehlo/types"
	"github.com/gomlx/stablehlo/types/shapes"
	"github.com/stretchr/testify/require"
)

var flagPluginNames = flag.String("plugins", "cpu", "List (|-separated) of PJRT plugin names or full paths. E.g. \"cpu|cuda\"")

func must(err error) {
	if err != nil {
		panic(err)
	}
}

func must1[T any](value T, err error) T {
	if err != nil {
		panic(err)
	}
	return value
}

func must2[T1, T2 any](value1 T1, value2 T2, err error) (T1, T2) {
	if err != nil {
		panic(err)
	}
	return value1, value2
}

// withLines prefix each line of text with a "%04d: " of the line number.
func withLines(text []byte) string {
	var result string
	lines := strings.Split(string(text), "\n")
	for i, line := range lines {
		result += fmt.Sprintf("%04d: %s\n", i+1, line)
	}
	return result
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
		expectedShape, err := shapes.FromAnyValue(expected[i].Flat)
		require.NoErrorf(t, err, "failed to get shape for output #%d: %v", i, expected[i].Flat)
		dtype := expectedShape.DType
		fmt.Printf("\t - output #%d:\n\t   - Got: dims=%v, flat_values=%v\n", i, gotDims, gotFlat)
		fmt.Printf("\t   - Want(%s): dims=%v, flat_values=%v\n", dtype, expected[i].Dims, expected[i].Flat)
		require.NoErrorf(t, err, "failed to get buffer contents for output #%d, expected flat value %v", i, expected[i].Flat)
		require.Equalf(t, expected[i].Dims, gotDims, "output #%d dims don't match", i)
		switch dtype {
		case dtypes.Float64, dtypes.Float32:
			require.InDeltaSlicef(t, expected[i].Flat, gotFlat, 1e-4, "output #%d flat values don't match", i)
		default:
			require.Equalf(t, expected[i].Flat, gotFlat, "output #%d flat values don't match", i)
		}
	}
}

func TestOps(t *testing.T) {
	for pluginName, client := range pjrtClientsIterator(t) {
		t.Run(pluginName, func(t *testing.T) {
			testOps(t, client)
		})
	}
}

func testOps(t *testing.T, client *pjrt.Client) {
	t.Run("Return-multi-output", func(t *testing.T) {
		b := New(t.Name())
		fn := b.NewFunction("main")
		c1 := must1(fn.ConstantFromScalar(1.0))
		c2 := must1(fn.ConstantFromScalar(2.0))
		c3 := must1(fn.ConstantFromScalar(float32(math.Inf(-1))))
		c4 := must1(fn.ConstantFromScalar(math.Inf(1)))
		sum := must1(Add(c1, c2))
		must(fn.Return(c1, c2, sum, c3, c4))
		program := must1(b.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		output := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float64{1}, nil},
			{[]float64{2}, nil},
			{[]float64{3}, nil},
			{[]float32{float32(math.Inf(-1))}, nil},
			{[]float64{math.Inf(1)}, nil},
		}, output)
	})

	t.Run("Complex", func(t *testing.T) {
		builder := New(t.Name())
		shape := shapes.Make(dtypes.Float64)
		fn := builder.Main()
		lhsV, rhsV := fn.NamedInput("lhs", shape), fn.NamedInput("rhs", shape)
		must(fn.Return(must1(Complex(lhsV, rhsV))))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		a := must1(client.BufferFromHost().FromFlatDataWithDimensions([]float64{1.0}, nil).Done())
		b := must1(client.BufferFromHost().FromFlatDataWithDimensions([]float64{-1.0}, nil).Done())
		output := compileAndExecute(t, client, program, a, b)
		requireBuffersEqual(t, []FlatAndDims{{[]complex128{1 - 1i}, nil}}, output)
	})

	t.Run("Clamp", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		minV := fn.NamedInput("min", shapes.Make(dtypes.Float32))
		xV := fn.NamedInput("x", shapes.Make(dtypes.Float32, 3))
		maxV := fn.NamedInput("max", shapes.Make(dtypes.Float32))
		must(fn.Return(must1(Clamp(minV, xV, maxV))))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		minArg := must1(client.BufferFromHost().FromFlatDataWithDimensions([]float32{-1.0}, nil).Done())
		maxArg := must1(client.BufferFromHost().FromFlatDataWithDimensions([]float32{1.0}, nil).Done())
		x := must1(client.BufferFromHost().FromFlatDataWithDimensions([]float32{0.1, -2.2, 3.3}, []int{3}).Done())
		output := compileAndExecute(t, client, program, minArg, x, maxArg)
		requireBuffersEqual(t, []FlatAndDims{{[]float32{0.1, -1, 1}, []int{3}}}, output)
	})

	t.Run("Iota", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		must(fn.Return(
			must1(fn.Iota(shapes.Make(dtypes.F32, 2, 2), 0)),
			must1(fn.Iota(shapes.Make(dtypes.F32, 2, 2), 1)),
			must1(fn.Iota(shapes.Make(dtypes.F32, 4), 0)),
		))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{0, 0, 1, 1}, []int{2, 2}},
			{[]float32{0, 1, 0, 1}, []int{2, 2}},
			{[]float32{0, 1, 2, 3}, []int{4}},
		}, outputs)
	})

	t.Run("IsFinite", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		input := fn.NamedInput("x", shapes.Make(dtypes.F64, 6))
		must(fn.Return(must1(IsFinite(input))))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		v := must1(client.BufferFromHost().FromFlatDataWithDimensions([]float64{0, -1, 1, math.Inf(1), math.Inf(-1), math.NaN()}, []int{6}).Done())
		outputs := compileAndExecute(t, client, program, v)
		requireBuffersEqual(t, []FlatAndDims{
			{[]bool{true, true, true, false, false, false}, []int{6}},
		}, outputs)
	})

	t.Run("Reshape", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		x := must1(fn.Iota(shapes.Make(dtypes.F32, 3, 2), 0))
		y := must1(Reshape(x, shapes.Make(dtypes.F32, 2, 3)))
		must(fn.Return(y))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{0, 0, 1, 1, 2, 2}, []int{2, 3}},
		}, outputs)
	})

	t.Run("BroadcastInDim", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		x := must1(fn.Iota(shapes.Make(dtypes.F32, 3), 0))
		y := must1(BroadcastInDim(x, shapes.Make(dtypes.F32, 2, 3), []int{1}))
		must(fn.Return(y))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{0, 1, 2, 0, 1, 2}, []int{2, 3}},
		}, outputs)
	})

	t.Run("BroadcastInDim<scalar>", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		x := must1(fn.ConstantFromScalar(float32(7.0)))
		y := must1(BroadcastInDim(x, shapes.Make(dtypes.F32, 3), nil))
		must(fn.Return(y))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{7, 7, 7}, []int{3}},
		}, outputs)
	})

	t.Run("Gather", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		x := must1(fn.Iota(shapes.Make(dtypes.F32, 3*5), 0))
		x = must1(Reshape(x, shapes.Make(dtypes.F32, 3, 5)))
		indices := must1(fn.ConstantFromFlatAndDimensions([]int{2, 0}, 2, 1))
		offsetOutputAxes := []int{1}
		collapsedSliceAxes := []int{0}
		var operandBatchingAxes, startIndicesBatchingAxes []int
		startIndexMap := []int{0}
		sliceSizes := []int{1, 5}
		y := must1(Gather(x, indices, 1,
			offsetOutputAxes, collapsedSliceAxes, operandBatchingAxes,
			startIndicesBatchingAxes, startIndexMap,
			sliceSizes, false))
		must(fn.Return(y))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{ /* row=0: */ 10, 11, 12, 13, 14 /* row=1: */, 0, 1, 2, 3, 4}, []int{2, 5}},
		}, outputs)
	})

	//	Slice(x={0, 1, 2, 3, 4}, starts={2}, limits={4}, strides=nil) -> {2, 3}
	//	Slice(x={0, 1, 2, 3, 4}, starts={2}, limits={5}, strides={2}) -> {2, 4}
	t.Run("Slice", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		x := must1(fn.Iota(shapes.Make(dtypes.F32, 5), 0))
		y0 := must1(Slice(x, []int{2}, []int{4}, nil))
		y1 := must1(Slice(x, []int{2}, []int{5}, []int{2}))
		must(fn.Return(y0, y1))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{2, 3}, []int{2}},
			{[]float32{2, 4}, []int{2}},
		}, outputs)
	})

	t.Run("Concatenate", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		x := must1(fn.Iota(shapes.Make(dtypes.F32, 2, 3), 1))
		y := must1(fn.Iota(shapes.Make(dtypes.F32, 2, 1), 0))
		z := must1(Concatenate(1, x, y))
		must(fn.Return(z))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{0, 1, 2, 0, 0, 1, 2, 1}, []int{2, 4}},
		}, outputs)
	})

	t.Run("Reduce", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		x := must1(fn.Iota(shapes.Make(dtypes.F32, 2*3), 0))
		x = must1(Reshape(x, shapes.Make(dtypes.F32, 2, 3)))
		zero := must1(fn.ConstantFromScalar(float32(0)))
		reductionFn := fn.Closure()
		lhs := reductionFn.NamedInput("lhs", shapes.Make(dtypes.F32))
		rhs := reductionFn.NamedInput("rhs", shapes.Make(dtypes.F32))
		must(reductionFn.Return(must1(Add(lhs, rhs))))
		r0 := must1(Reduce(x, zero, reductionFn, 1))
		r1 := must1(Reduce(x, zero, reductionFn, 0))
		must(fn.Return(r0, r1))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{3, 12}, []int{2}},
			{[]float32{3, 5, 7}, []int{3}},
		}, outputs)
	})

	t.Run("MultiReduce", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		x := must1(fn.Iota(shapes.Make(dtypes.F32, 2*3), 0))
		x = must1(Reshape(x, shapes.Make(dtypes.F32, 2, 3)))
		y := must1(fn.Iota(shapes.Make(dtypes.Int32, 2*3), 0))
		y = must1(Reshape(y, shapes.Make(dtypes.Int32, 2, 3)))
		zeroF32 := must1(fn.ConstantFromScalar(float32(0)))
		zeroI32 := must1(fn.ConstantFromScalar(int32(0)))
		reductionFn := fn.Closure()
		lhs0 := reductionFn.NamedInput("lhs0", shapes.Make(dtypes.F32))
		lhs1 := reductionFn.NamedInput("lhs1", shapes.Make(dtypes.Int32))
		rhs0 := reductionFn.NamedInput("rhs0", shapes.Make(dtypes.F32))
		rhs1 := reductionFn.NamedInput("rhs1", shapes.Make(dtypes.Int32))
		must(reductionFn.Return(
			must1(Add(lhs0, rhs0)),
			must1(Add(lhs1, rhs1))))
		results := must1(MultiReduce(
			[]*Value{x, y},
			[]*Value{zeroF32, zeroI32}, reductionFn, 1))
		must(fn.Return(results[0], results[1]))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{3, 12}, []int{2}},
			{[]int32{3, 12}, []int{2}},
		}, outputs)
	})

	t.Run("Select", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		pred0 := must1(fn.ConstantFromFlatAndDimensions([]bool{true, false, true}, 3))
		pred1 := must1(fn.ConstantFromScalar(false))
		onTrue := must1(fn.Iota(shapes.Make(dtypes.F32, 3), 0))
		onFalse := must1(Negate(onTrue))
		result0 := must1(Select(pred0, onTrue, onFalse))
		result1 := must1(Select(pred1, onTrue, onFalse))
		must(fn.Return(result0, result1))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{0, -1, 2}, []int{3}},
			{[]float32{0, -1, -2}, []int{3}},
		}, outputs)
	})

	t.Run("BitcastConvert", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		c0 := must1(fn.ConstantFromFlatAndDimensions([]uint16{0xbeef, 0xdead}, 1, 2))
		c1 := must1(fn.ConstantFromScalar(uint32(0xdeadbeef)))
		c2 := must1(fn.ConstantFromScalar(uint32(0x7F800000)))
		must(fn.Return(
			must1(BitcastConvert(c0, dtypes.Uint32)),
			must1(BitcastConvert(c1, dtypes.Uint16)),
			must1(BitcastConvert(c2, dtypes.F32)),
		))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]uint32{0xdeadbeef}, []int{1}},
			{[]uint16{0xbeef, 0xdead}, []int{2}},
			{[]float32{float32(math.Inf(1))}, nil},
		}, outputs)
	})

	t.Run("Transpose", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		x := must1(fn.Iota(shapes.Make(dtypes.F32, 2*3), 0))
		x = must1(Reshape(x, shapes.Make(dtypes.F32, 2, 3)))
		must(fn.Return(must1(Transpose(x, 1, 0))))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{0, 3, 1, 4, 2, 5}, []int{3, 2}},
		}, outputs)
	})

	for _, algo := range []types.RngBitGeneratorAlgorithm{types.RngDefault, types.RngThreeFry, types.RngPhilox} {
		t.Run(fmt.Sprintf("RngBitGenerator-%s", algo), func(t *testing.T) {
			builder := New(t.Name())
			fn := builder.Main()
			state := must1(fn.ConstantFromFlatAndDimensions([]uint64{42, 1}, 2))
			const numSamples = 10_000
			_, noiseV := must2(RngBitGenerator(state, shapes.Make(dtypes.Uint64, numSamples), algo))
			must(fn.Return(noiseV))
			program := must1(builder.Build())
			fmt.Printf("%s program:\n%s", t.Name(), program)
			outputs := compileAndExecute(t, client, program)
			flat, dims, err := outputs[0].ToFlatDataAndDimensions()
			require.NoError(t, err)
			require.Equal(t, []int{numSamples}, dims)
			noise := flat.([]uint64)
			// Count bits in each uint64
			var totalBits int
			for _, n := range noise {
				totalBits += bits.OnesCount64(n)
			}
			// We expect roughly 32 bits per number +/- 2 standard deviations
			expectedBits := 32 * numSamples
			fmt.Printf("\tgot %d bits set, expected %d\n", totalBits, expectedBits)
			margin := 2 * numSamples
			require.Greater(t, totalBits, expectedBits-margin)
			require.Less(t, totalBits, expectedBits+margin)
		})
	}

	t.Run("Scatter", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		zeroF32 := must1(fn.ConstantFromScalar(float32(0)))
		zeroI32 := must1(fn.ConstantFromScalar(int32(0)))
		inputsF32 := must1(BroadcastInDim(zeroF32, shapes.Make(dtypes.F32, 2, 1, 2, 3), nil))
		inputsI32 := must1(BroadcastInDim(zeroI32, shapes.Make(dtypes.Int32, 2, 1, 2, 3), nil))
		scatterIndices := must1(fn.ConstantFromFlatAndDimensions(
			[]int32{1, 1, 0, 0, 0, 2, 0, 0}, 2, 2, 2))

		updatesF32 := must1(fn.Iota(shapes.Make(dtypes.F32, 2*2*1), 0))
		updatesF32 = must1(Reshape(updatesF32, shapes.Make(dtypes.F32, 2, 2, 1)))
		hundredF32 := must1(BroadcastInDim(
			must1(fn.ConstantFromScalar(float32(100))), updatesF32.Shape(), nil))
		updatesF32 = must1(Add(updatesF32, hundredF32))

		updatesI32 := must1(fn.Iota(shapes.Make(dtypes.Int32, 2*2*1), 0))
		updatesI32 = must1(Reshape(updatesI32, shapes.Make(dtypes.Int32, 2, 2, 1)))
		thousandI32 := must1(BroadcastInDim(
			must1(fn.ConstantFromScalar(int32(1000))), updatesI32.Shape(), nil))
		updatesI32 = must1(Add(updatesI32, thousandI32))

		updateWindowAxes := []int{2}
		insertedWindowAxes := []int{1, 2}
		inputBatchingAxes := []int{0}
		scatterIndicesBatchingAxes := []int{0}
		indexedInputAxes := []int{2, 3}
		indexVectorAxis := 2
		updateFn := fn.Closure()
		lhsF32 := updateFn.NamedInput("lhsF32", shapes.Make(dtypes.F32))
		lhsI32 := updateFn.NamedInput("lhsI32", shapes.Make(dtypes.Int32))
		rhsF32 := updateFn.NamedInput("rhsF32", shapes.Make(dtypes.F32))
		rhsI32 := updateFn.NamedInput("rhsI32", shapes.Make(dtypes.Int32))
		must(updateFn.Return(
			must1(Add(lhsF32, rhsF32)),
			must1(Add(lhsI32, rhsI32)),
		))
		results := must1(MultiScatter(
			[]*Value{inputsF32, inputsI32}, scatterIndices, []*Value{updatesF32, updatesI32},
			updateWindowAxes, insertedWindowAxes,
			inputBatchingAxes, scatterIndicesBatchingAxes,
			indexedInputAxes, indexVectorAxis,
			false, false,
			updateFn))
		must(fn.Return(results[0], results[1]))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{101, 0, 0, 0, 100, 0, 103, 0, 102, 0, 0, 0}, []int{2, 1, 2, 3}},
			{[]int32{1001, 0, 0, 0, 1000, 0, 1003, 0, 1002, 0, 0, 0}, []int{2, 1, 2, 3}},
		}, outputs)
	})

	t.Run("Convert", func(t *testing.T) {
		builder := New(t.Name())
		fn := builder.Main()
		x0 := must1(fn.Iota(shapes.Make(dtypes.F32, 3), 0))
		x1 := must1(fn.ConstantFromFlatAndDimensions([]bool{true, false, true}, 3))
		must(fn.Return(
			must1(Convert(x0, dtypes.Bool)),
			must1(Convert(x1, dtypes.Int32))))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		outputs := compileAndExecute(t, client, program)
		requireBuffersEqual(t, []FlatAndDims{
			{[]bool{false, true, true}, []int{3}},
			{[]int32{1, 0, 1}, []int{3}},
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
		op func(lhs, rhs *Value) (*Value, error),
		dtype dtypes.DType, lhs, rhs any, expected any) {
		builder := New(t.Name())
		shape := shapes.Make(dtype)
		fn := builder.Main()
		lhsV, rhsV := fn.NamedInput("lhs", shape), fn.NamedInput("rhs", shape)
		result := must1(op(lhsV, rhsV))
		must(fn.Return(result))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		a := must1(client.BufferFromHost().FromFlatDataWithDimensions(lhs, []int{}).Done())
		b := must1(client.BufferFromHost().FromFlatDataWithDimensions(rhs, []int{}).Done())
		output := compileAndExecute(t, client, program, a, b)
		requireBuffersEqual(t, []FlatAndDims{{expected, nil}}, output)
	}

	t.Run("Add", func(t *testing.T) {
		testBinaryOp(t, "Add", Add, dtypes.Float32, []float32{3.0}, []float32{7.0}, []float32{10.0})
	})
	t.Run("Atan2", func(t *testing.T) {
		testBinaryOp(t, "Atan2", Atan2, dtypes.Float32, []float32{3.0}, []float32{7.0},
			[]float32{float32(math.Atan2(3.0, 7.0))})
	})

	t.Run("Subtract", func(t *testing.T) {
		testBinaryOp(t, "Subtract", Subtract, dtypes.Float32, []float32{7.0}, []float32{3.0}, []float32{4.0})
	})

	t.Run("Multiply", func(t *testing.T) {
		testBinaryOp(t, "Multiply", Multiply, dtypes.Float32, []float32{3.0}, []float32{4.0}, []float32{12.0})
	})

	t.Run("Divide", func(t *testing.T) {
		testBinaryOp(t, "Divide", Divide, dtypes.Float32, []float32{12.0}, []float32{3.0}, []float32{4.0})
	})

	t.Run("Power", func(t *testing.T) {
		testBinaryOp(t, "Power", Power, dtypes.Float32, []float32{2.0}, []float32{3.0}, []float32{8.0})
	})

	t.Run("And_Uint32", func(t *testing.T) {
		testBinaryOp(t, "And", And, dtypes.Uint32, []uint32{0b1100}, []uint32{0b1010}, []uint32{0b1000})
	})

	t.Run("Or_Uint32", func(t *testing.T) {
		testBinaryOp(t, "Or", Or, dtypes.Uint32, []uint32{0b1100}, []uint32{0b1010}, []uint32{0b1110})
	})

	t.Run("Xor_Uint32", func(t *testing.T) {
		testBinaryOp(t, "Xor", Xor, dtypes.Uint32, []uint32{0b1100}, []uint32{0b1010}, []uint32{0b0110})
	})

	t.Run("And_Bool", func(t *testing.T) {
		testBinaryOp(t, "And", And, dtypes.Bool, []bool{true}, []bool{false}, []bool{false})
	})

	t.Run("Or_Bool", func(t *testing.T) {
		testBinaryOp(t, "Or", Or, dtypes.Bool, []bool{true}, []bool{false}, []bool{true})
	})

	t.Run("Xor_Bool", func(t *testing.T) {
		testBinaryOp(t, "Xor", Xor, dtypes.Bool, []bool{true}, []bool{false}, []bool{true})
	})

	t.Run("ShiftLeft", func(t *testing.T) {
		testBinaryOp(t, "ShiftLeft", ShiftLeft, dtypes.Uint32, []uint32{0b1}, []uint32{2}, []uint32{0b100})
	})

	t.Run("ShiftRightArithmetic", func(t *testing.T) {
		testBinaryOp(t, "ShiftRightArithmetic", ShiftRightArithmetic, dtypes.Int32, []int32{-8}, []int32{1}, []int32{-4})
	})

	t.Run("ShiftRightLogical", func(t *testing.T) {
		testBinaryOp(t, "ShiftRightLogical", ShiftRightLogical, dtypes.Uint32, []uint32{0b1100}, []uint32{2}, []uint32{0b11})
	})

	t.Run("Reminder", func(t *testing.T) {
		testBinaryOp(t, "Remainder", Remainder, dtypes.Float32, []float32{7.0}, []float32{4.0}, []float32{3.0})
	})

	t.Run("Maximum", func(t *testing.T) {
		testBinaryOp(t, "Maximum", Maximum, dtypes.Float32, []float32{3.0}, []float32{7.0}, []float32{7.0})
	})

	t.Run("Minimum", func(t *testing.T) {
		testBinaryOp(t, "Minimum", Minimum, dtypes.Float32, []float32{3.0}, []float32{7.0}, []float32{3.0})
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
		dtype dtypes.DType, lhs, rhs any, expected any) {
		builder := New(t.Name())
		shape := shapes.Make(dtype)
		fn := builder.Main()
		lhsV, rhsV := fn.NamedInput("lhs", shape), fn.NamedInput("rhs", shape)
		result := must1(Compare(lhsV, rhsV, direction, compareType))
		must(fn.Return(result))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		a := must1(client.BufferFromHost().FromFlatDataWithDimensions(lhs, []int{}).Done())
		b := must1(client.BufferFromHost().FromFlatDataWithDimensions(rhs, []int{}).Done())
		output := compileAndExecute(t, client, program, a, b)
		requireBuffersEqual(t, []FlatAndDims{{expected, nil}}, output)
	}

	t.Run("Float_EQ", func(t *testing.T) {
		runTest(t, "Compare", types.CompareEQ, types.CompareFloat,
			dtypes.Float32, []float32{3.0}, []float32{3.0}, []bool{true})
	})

	t.Run("Signed_GT", func(t *testing.T) {
		runTest(t, "Compare", types.CompareGT, types.CompareSigned,
			dtypes.Int32, []int32{7}, []int32{3}, []bool{true})
	})

	t.Run("Unsigned_LT", func(t *testing.T) {
		runTest(t, "Compare", types.CompareLT, types.CompareUnsigned,
			dtypes.Uint32, []uint32{3}, []uint32{7}, []bool{true})
	})

	t.Run("TotalOrder_GE", func(t *testing.T) {
		runTest(t, "Compare", types.CompareGE, types.CompareTotalOrder,
			dtypes.Float32, []float32{3.0}, []float32{3.0}, []bool{true})
	})

	t.Run("Float_NE", func(t *testing.T) {
		runTest(t, "Compare", types.CompareNE, types.CompareFloat,
			dtypes.Float32, []float32{3.0}, []float32{7.0}, []bool{true})
	})

	t.Run("Signed_LE", func(t *testing.T) {
		runTest(t, "Compare", types.CompareLE, types.CompareSigned,
			dtypes.Int32, []int32{3}, []int32{7}, []bool{true})
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
		op func(x *Value) (*Value, error),
		dtype dtypes.DType, input any, expected any) {
		builder := New(t.Name())
		shape := shapes.Make(dtype)
		fn := builder.Main()
		arg := fn.Input(shape)
		result := must1(op(arg))
		must(fn.Return(result))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		a := must1(client.BufferFromHost().FromFlatDataWithDimensions(input, []int{}).Done())
		output := compileAndExecute(t, client, program, a)
		requireBuffersEqual(t, []FlatAndDims{{expected, nil}}, output)
	}

	t.Run("Not_Bool", func(t *testing.T) {
		testUnaryOp(t, "Not", Not, dtypes.Bool, []bool{true}, []bool{false})
	})
	t.Run("Not_Uint8", func(t *testing.T) {
		testUnaryOp(t, "Not", Not, dtypes.Uint8, []uint8{128}, []uint8{127})
	})

	t.Run("Popcnt_Uint32", func(t *testing.T) {
		testUnaryOp(t, "Popcnt", Popcnt, dtypes.Uint32, []uint32{0b1011}, []uint32{3})
	})

	t.Run("CountLeadingZeros_Uint32", func(t *testing.T) {
		testUnaryOp(t, "CountLeadingZeros", CountLeadingZeros, dtypes.Uint32, []uint32{0b1}, []uint32{31})
	})

	t.Run("Erf_Float32", func(t *testing.T) {
		testUnaryOp(t, "Erf", Erf, dtypes.Float64, []float64{1.0}, []float64{
			math.Erf(1)})
	})

	t.Run("Exponential_Float32", func(t *testing.T) {
		testUnaryOp(t, "Exponential", Exponential, dtypes.Float32, []float32{1.0},
			[]float32{float32(math.Exp(1))})
	})

	t.Run("ExponentialMinusOne_Float32", func(t *testing.T) {
		testUnaryOp(t, "ExponentialMinusOne", ExponentialMinusOne, dtypes.Float32, []float32{1.0},
			[]float32{float32(math.E - 1)})
	})

	t.Run("Log_Float32", func(t *testing.T) {
		testUnaryOp(t, "Log", Log, dtypes.Float32, []float32{2.7183}, []float32{1.0})
	})

	t.Run("LogPlusOne_Float32", func(t *testing.T) {
		testUnaryOp(t, "LogPlusOne", LogPlusOne, dtypes.Float32, []float32{1.7183}, []float32{1.0})
	})

	t.Run("Logistic_Float32", func(t *testing.T) {
		testUnaryOp(t, "Logistic", Logistic, dtypes.Float32, []float32{0.0}, []float32{0.5})
	})

	t.Run("Ceil_Float32", func(t *testing.T) {
		testUnaryOp(t, "Ceil", Ceil, dtypes.Float32, []float32{1.7}, []float32{2.0})
	})

	t.Run("Floor_Float32", func(t *testing.T) {
		testUnaryOp(t, "Floor", Floor, dtypes.Float32, []float32{1.7}, []float32{1.0})
	})

	t.Run("RoundNearestEven_Float32", func(t *testing.T) {
		testUnaryOp(t, "RoundNearestEven", RoundNearestEven, dtypes.Float32, []float32{2.5}, []float32{2.0})
	})
	t.Run("RoundNearestAfz_Float32", func(t *testing.T) {
		testUnaryOp(t, "RoundNearestAfz", RoundNearestAfz, dtypes.Float32, []float32{2.5}, []float32{3.0})
	})

	t.Run("Rsqrt_Float32", func(t *testing.T) {
		testUnaryOp(t, "Rsqrt", Rsqrt, dtypes.Float32, []float32{4.0}, []float32{0.5})
	})

	t.Run("Sqrt_Float32", func(t *testing.T) {
		testUnaryOp(t, "Sqrt", Sqrt, dtypes.Float32, []float32{4.0}, []float32{2.0})
	})

	t.Run("Cbrt_Float32", func(t *testing.T) {
		testUnaryOp(t, "Cbrt", Cbrt, dtypes.Float32, []float32{8.0}, []float32{2.0})
	})

	t.Run("Cosine_Float32", func(t *testing.T) {
		testUnaryOp(t, "Cosine", Cosine, dtypes.Float32, []float32{0.0}, []float32{1.0})
	})

	t.Run("Sine_Float32", func(t *testing.T) {
		testUnaryOp(t, "Sine", Sine, dtypes.Float32, []float32{0.0}, []float32{0.0})
	})

	t.Run("Tan_Float32", func(t *testing.T) {
		testUnaryOp(t, "Tan", Tan, dtypes.Float32, []float32{pi32 / 4}, []float32{1})
	})
	t.Run("Tanh_Float32", func(t *testing.T) {
		testUnaryOp(t, "Tanh", Tanh, dtypes.Float32, []float32{0.5},
			[]float32{float32(math.Tanh(0.5))})
	})

	t.Run("Abs_Float32", func(t *testing.T) {
		testUnaryOp(t, "Abs", Abs, dtypes.Float32, []float32{-3.0}, []float32{3.0})
	})

	t.Run("Negate_Float32", func(t *testing.T) {
		testUnaryOp(t, "Negate", Negate, dtypes.Float32, []float32{3.0}, []float32{-3.0})
	})

	t.Run("Sign_Float32", func(t *testing.T) {
		testUnaryOp(t, "Sign", Sign, dtypes.Float32, []float32{-3.0}, []float32{-1.0})
	})

	t.Run("Real_Complex64", func(t *testing.T) {
		testUnaryOp(t, "Real", Real, dtypes.Complex64, []complex64{complex(3.0, 4.0)}, []float32{3.0})
	})

	t.Run("Real_Complex128", func(t *testing.T) {
		testUnaryOp(t, "Real", Real, dtypes.Complex128, []complex128{complex(3.0, 4.0)}, []float64{3.0})
	})

	t.Run("Imag_Complex64", func(t *testing.T) {
		testUnaryOp(t, "Imag", Imag, dtypes.Complex64, []complex64{complex(3.0, 4.0)}, []float32{4.0})
	})

	t.Run("Imag_Complex128", func(t *testing.T) {
		testUnaryOp(t, "Imag", Imag, dtypes.Complex128, []complex128{complex(3.0, 4.0)}, []float64{4.0})
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
		builder := New(t.Name())
		fn := builder.Main()
		c, err := fn.ConstantFromScalar(scalar)
		require.NoError(t, err)
		must(fn.Return(c))
		program := must1(builder.Build())
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
		builder := New(t.Name())
		fn := builder.Main()
		c, err := fn.ConstantFromFlatAndDimensions(flat, dimensions...)
		require.NoError(t, err)
		must(fn.Return(c))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		output := compileAndExecute(t, client, program)[0]
		gotFlat, gotDims, err := output.ToFlatDataAndDimensions()
		require.NoError(t, err)
		require.Equal(t, dimensions, gotDims)
		require.Equal(t, flat, gotFlat)
	}

	t.Run("0D-int8", func(t *testing.T) { testTensor(t, []int8{-3}) })
	t.Run("1D-float32", func(t *testing.T) { testTensor(t, []float32{1, 2, 3, 5, 7}, 5) })
	t.Run("2D-complex64", func(t *testing.T) { testTensor(t, []complex64{1, 2, 3, 5i, 7i, 11i}, 2, 3) })
	t.Run("3D-bool", func(t *testing.T) { testTensor(t, []bool{false, true, false, true}, 2, 1, 2) })
}
