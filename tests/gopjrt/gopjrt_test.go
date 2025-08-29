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

var flagPluginNames = flag.String("plugin", "cpu|cuda", "List (|-separated) of PRJT plugin names or full paths")

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
		t.Run(pluginName, func(t *testing.T) {
			testRunWithPlugin(t, plugin)
		})
	}
}

func testRunWithPlugin(t *testing.T, plugin *pjrt.Plugin) {
	t.Run("no inputs", func(t *testing.T) {
		b := stablehlo.New(t.Name())
		fn := b.NewFunction("main")
		c1 := must(fn.NewConstant(1.0))
		c2 := must(fn.NewConstant(2.0))
		sum := must(fn.Add(c1, c2))
		fn.Return(sum)
		program := must(b.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
	})

	t.Run("with inputs", func(t *testing.T) {
		builder := stablehlo.New(t.Name())
		shape := S.Make(D.Float64)
		lhs, rhs := stablehlo.NamedValue("lhs", shape), stablehlo.NamedValue("rhs", shape)
		fn := builder.NewFunction("main", lhs, rhs)
		sum := must(fn.Add(lhs, rhs))
		fn.Return(sum)
		program := must(builder.Build())
		fmt.Printf("%s program:\n%s", t.Name(), program)
	})
}
