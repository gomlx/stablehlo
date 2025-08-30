package gopjrt

import (
	"flag"
	"fmt"
	"strings"
	"testing"

	D "github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/pjrt"
	"github.com/gomlx/stablehlo"
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
			testRunWithClient(t, client)
		})
		require.NoError(t, client.Destroy())
	}
}

func testRunWithClient(t *testing.T, client *pjrt.Client) {
	const deviceNum = 0
	t.Run("no inputs", func(t *testing.T) {
		b := stablehlo.New(t.Name())
		fn := b.NewFunction("main")
		c1 := must(fn.NewConstant(1.0))
		c2 := must(fn.NewConstant(2.0))
		sum := must(fn.Add(c1, c2))
		fn.Return(sum)
		program := must(b.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		loadedExec := must(client.Compile().WithStableHLO(program).Done())
		outputBuffers := must(loadedExec.Execute().OnDevicesByNum(deviceNum).Done())
		require.Len(t, outputBuffers, 1)
		values, dims, err := outputBuffers[0].ToFlatDataAndDimensions()
		require.NoError(t, err)
		fmt.Printf("\t - output: dims=%v, values=%v\n", dims, values)
		require.Len(t, dims, 0)
		require.Equal(t, []float64{3.0}, values)
	})

	t.Run("with inputs", func(t *testing.T) {
		builder := stablehlo.New(t.Name())
		shape := S.Make(D.Float32)
		lhs, rhs := stablehlo.NamedValue("lhs", shape), stablehlo.NamedValue("rhs", shape)
		fn := builder.NewFunction("main", lhs, rhs)
		sum := must(fn.Add(lhs, rhs))
		fn.Return(must(fn.Negate(sum)))
		program := must(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
		a := must(client.BufferFromHost().FromFlatDataWithDimensions([]float32{3.0}, []int{}).Done())
		b := must(client.BufferFromHost().FromFlatDataWithDimensions([]float32{7.0}, []int{}).Done())
		loadedExec := must(client.Compile().WithStableHLO(program).Done())
		outputBuffers := must(loadedExec.Execute(a, b).DonateAll().OnDevicesByNum(deviceNum).Done())
		require.Len(t, outputBuffers, 1)
		values, dims, err := outputBuffers[0].ToFlatDataAndDimensions()
		require.NoError(t, err)
		fmt.Printf("\t - output: dims=%v, values=%v\n", dims, values)
		require.Len(t, dims, 0)
		require.Equal(t, []float32{-10.0}, values)
	})
}
