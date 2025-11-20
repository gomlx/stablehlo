package gopjrt

import (
	"fmt"
	"testing"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/pjrt"
	"github.com/gomlx/stablehlo"
	"github.com/gomlx/stablehlo/types/shapes"
	"github.com/gomlx/stablehlo/types/shardy"
	"github.com/stretchr/testify/require"
)

func TestShardy(t *testing.T) {
	iterateClientsAndTest(t, testShardy)
}

// compileAndExecute program with PJRT. All inputs are donated.
func shardyCompileAndExecute(t *testing.T, client *pjrt.Client, program []byte,
	mesh *shardy.DeviceMesh, inputs ...*pjrt.Buffer) []*pjrt.Buffer {
	loadedExec, err := client.Compile().
		WithStableHLO(program).
		WithShardy(mesh.NumDevices()).
		WithDeviceAssignment(mesh.LogicalDeviceAssignment()).
		Done()
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

func testShardy(t *testing.T, client *pjrt.Client) {
	// We will test it with 2 devices.
	const numReplicas = 2
	numDevices := client.NumDevices()
	if numDevices < numReplicas {
		t.Skipf("Skipping test: not enough devices: %d < %d", numDevices, numReplicas)
		return
	}

	t.Run("input-data-sharding", func(t *testing.T) {
		mesh := must1(shardy.NewDeviceMesh("data_mesh", []int{2}, []string{"data"}))
		builder := stablehlo.New(t.Name()).WithShardy(mesh)
		fn := builder.Main()
		x := must1(fn.NamedInputWithSharding("arg0", shapes.Make(dtypes.F32, 2, 3),
			builder.NewShardingSpec().AddShardedAxis("data")))
		reductionFn := fn.Closure()
		lhs := must1(reductionFn.NamedInput("lhs", shapes.Make(dtypes.F32)))
		rhs := must1(reductionFn.NamedInput("rhs", shapes.Make(dtypes.F32)))
		must(reductionFn.Return(must1(stablehlo.Add(lhs, rhs))))
		zero := must1(fn.ConstantFromScalar(float32(0)))
		output := must1(stablehlo.Reduce(x, zero, reductionFn, 0, 1))
		must(fn.Return(output))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		x0 := must1(client.BufferFromHost().ToDeviceNum(0).FromFlatDataWithDimensions(
			[]float32{0, 1, 2}, []int{1, 3}).Done())
		x1 := must1(client.BufferFromHost().ToDeviceNum(1).FromFlatDataWithDimensions(
			[]float32{0, 0.1, 0.2}, []int{1, 3}).Done())
		outputs := shardyCompileAndExecute(t, client, program, mesh, x0, x1)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{3.3}, nil},
			{[]float32{3.3}, nil},
		}, outputs)
	})

	t.Run("output-data-sharding", func(t *testing.T) {
		mesh := must1(shardy.NewDeviceMesh("data_mesh", []int{2}, []string{"data"}))
		builder := stablehlo.New(t.Name()).WithShardy(mesh)
		fn := builder.NewFunction("main")
		x := must1(fn.NamedInputWithSharding("arg0", shapes.Make(dtypes.F32, 2, 3),
			builder.NewShardingSpec().AddShardedAxis("data")))
		reductionFn := fn.Closure()
		lhs := must1(reductionFn.NamedInput("lhs", shapes.Make(dtypes.F32)))
		rhs := must1(reductionFn.NamedInput("rhs", shapes.Make(dtypes.F32)))
		must(reductionFn.Return(must1(stablehlo.Add(lhs, rhs))))
		zero := must1(fn.ConstantFromScalar(float32(0)))
		output := must1(stablehlo.Reduce(x, zero, reductionFn, 1))
		must(fn.Return(output))
		program := must1(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		x0 := must1(client.BufferFromHost().ToDeviceNum(0).FromFlatDataWithDimensions(
			[]float32{0, 1, 2}, []int{1, 3}).Done())
		x1 := must1(client.BufferFromHost().ToDeviceNum(1).FromFlatDataWithDimensions(
			[]float32{0, 0.1, 0.2}, []int{1, 3}).Done())
		outputs := shardyCompileAndExecute(t, client, program, mesh, x0, x1)
		requireBuffersEqual(t, []FlatAndDims{
			{[]float32{3}, []int{1}},
			{[]float32{0.3}, []int{1}},
		}, outputs)
	})

}
