package gopjrt

import (
	"fmt"
	"testing"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/pjrt"
	. "github.com/gomlx/stablehlo"
	"github.com/gomlx/stablehlo/types/shapes"
	"github.com/stretchr/testify/require"
)

func TestCollectiveOps(t *testing.T) {
	iterateClientsAndTest(t, testCollectiveOps)
}

func testCollectiveOps(t *testing.T, client *pjrt.Client) {
	if client.NumDevices() < 2 {
		t.Skipf("Skipping collective ops test: requires at least 2 devices, but client %q only has %d",
			client.Plugin().Name(), client.NumDevices())
	}

	// We will test it with 2 devices.
	const numReplicas = 2
	replicaGroups := [][]int{{0, 1}}
	devices := client.AddressableDevices()

	t.Run("CollectiveBroadcast", func(t *testing.T) {
		b := New(t.Name()).WithNumReplicas(numReplicas)
		fn := b.Main()

		// SPMD program: takes one argument per replica.
		arg0 := fn.NamedInput("arg0", shapes.Make(dtypes.F32, 2))
		_ = fn.NamedInput("arg1", shapes.Make(dtypes.F32, 2)) // arg1 is for replica 1

		// Broadcast %arg0 (from replica 0) to all replicas.
		// The input to this op on replica 1 (%arg1) will be ignored.
		broadcasted := must1(CollectiveBroadcast(arg0, replicaGroups))
		must(fn.Return(broadcasted))
		program := must1(b.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))

		// Prepare inputs: one buffer for each replica.
		// Replica 0 has the data to be broadcasted.
		input0 := must1(client.BufferFromHost().FromFlatDataWithDimensions(
			[]float32{1.0, 2.0}, []int{2}).ToDeviceNum(0).Done())
		// Replica 1 has different data, which will be overwritten.
		input1 := must1(client.BufferFromHost().FromFlatDataWithDimensions(
			[]float32{7.0, 13.0}, []int{2}).ToDeviceNum(1).Done())

		// Execute expects a flat list of inputs, one for each argument of main(),
		// mapped to devices in order.
		e, err := client.Compile().WithStableHLO(program).Done()
		require.NoErrorf(t, err, "failed to compile program: \n%s", program)
		outputBuffers, err := e.Execute(input0, input1).DonateAll().Done()
		require.NoErrorf(t, err, "failed to execute program: \n%s", program)

		// Check outputs: all replicas should have the data from replica 0.
		want := []FlatAndDims{
			{[]float32{1.0, 2.0}, []int{2}}, // Output on replica 0
			{[]float32{1.0, 2.0}, []int{2}}, // Output on replica 1
		}
		requireBuffersEqual(t, want, outputBuffers)
	})

	t.Run("CollectiveAllReduce", func(t *testing.T) {
		b := New(t.Name()).WithNumReplicas(numReplicas)

		// Define SUM computation
		sumComputation := b.NewFunction("sum")
		sumComputation.Parent = b.Main() // Mark as closure
		{
			lhs := sumComputation.NamedInput("lhs", shapes.Make(dtypes.F32))
			rhs := sumComputation.NamedInput("rhs", shapes.Make(dtypes.F32))
			sum := must1(Add(lhs, rhs))
			must(sumComputation.Return(sum))
		}

		// Define main SPMD program
		fn := b.Main()
		arg0 := fn.NamedInput("arg0", shapes.Make(dtypes.F32, 2)) // Input for replica 0
		arg1 := fn.NamedInput("arg1", shapes.Make(dtypes.F32, 2)) // Input for replica 1

		// AllReduce %arg0 and %arg1.
		// In SPMD, the HLO is the same, but %arg0 is mapped to the local operand.
		// The stablehlo.Builder API is a bit abstract. We pass %arg0, and the
		// compiler will map it to %arg0 for replica 0 and %arg1 for replica 1.
		// Note: A more "pure" SPMD HLO would just have one %arg0, but this
		// builder structure implies one input per replica.
		reduced := must1(CollectiveAllReduce(arg0, replicaGroups, sumComputation))
		must(fn.Return(reduced))
		program := must1(b.Build())
		fmt.Printf("%s program:\n%s", t.Name(), withLines(program))

		// Prepare inputs: one buffer for each replica.
		input0 := must1(client.BufferFromHost().FromFlatDataWithDimensions(
			[]float32{1.0, 10.0}, []int{2}).ToDeviceNum(0).Done())
		input1 := must1(client.BufferFromHost().FromFlatDataWithDimensions(
			[]float32{2.0, 20.0}, []int{2}).ToDeviceNum(1).Done())

		// Execute expects a flat list of inputs, one for each argument of main(),
		// mapped to devices in order.
		e, err := client.Compile().WithStableHLO(program).Done()
		require.NoErrorf(t, err, "failed to compile program: \n%s", program)
		outputBuffers, err := e.Execute(input0, input1).DonateAll().Done()
		require.NoErrorf(t, err, "failed to execute program: \n%s", program)

		// Check outputs: all replicas should have the sum.
		want := []FlatAndDims{
			{[]float32{3.0, 30.0}, []int{2}}, // Output on replica 0
			{[]float32{3.0, 30.0}, []int{2}}, // Output on replica 1
		}
		requireBuffersEqual(t, want, outputBuffers)
	})
}
