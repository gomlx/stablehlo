package stablehlo

import (
	"bytes"
	"fmt"
	"io"
	"slices"

	"github.com/gomlx/stablehlo/types"
	"github.com/pkg/errors"
)

// Builder is used to construct a StableHLO program.
// See details in New.
type Builder struct {
	name   string
	parent *Builder

	// functions holds all the functions created in the builder's scope.
	functions []*Function

	// inlineUniqueID is a counter used to generate unique names for inlined functions values.
	inlineUniqueID int

	// NumReplicas is the number of replicas for data parallelism.
	NumReplicas int
	// NumPartitions is the number of partitions for model parallelism.
	NumPartitions int

	// nextChannelID is the next ID to be assigned in channel handles.
	// It is just a Unique ID.
	nextChannelID int
}

// NormalizeIdentifier converts the name of an identifier (function name or function input parameter
// name) to a valid one: only letters, digits and underscores are allowed.
//
// Invalid characters are replaced with underscores.
// If the name starts with a digit, it is prefixed with an underscore.
//
// The name is normalized in place.
func NormalizeIdentifier(name string) string {
	result := make([]rune, 0, len(name)+1)
	if name[0] >= '0' && name[0] <= '9' {
		result = append(result, '_')
	}
	for _, r := range name {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' {
			result = append(result, r)
		} else {
			result = append(result, '_')
		}
	}
	return string(result)
}

// New creates a new Builder object holding a computation graph in construction.
//
// From a builder you can create functions.
// For each function you create operations (ops) one by one, until you defined the desired computation.
//
// You have to define the "main" function for your StableHLO program: you can use Builder.Main to do so, or
// Builder.NewFunction("main",...), it's the same.
//
// Once you are all set, call Builder.Build and it will return the StableHLO program as a []byte that can
// be used with PJRT.
//
// See github.com/gomlx/gopjrt for a Go API to PJRT.
func New(name string) *Builder {
	return &Builder{
		name: name,
	}
}

// WithNumReplicas sets the number of replicas (for data parallelism).
// This is added as an attribute to the StableHLO module.
func (b *Builder) WithNumReplicas(n int) *Builder {
	b.NumReplicas = n
	return b
}

// WithNumPartitions sets the number of partitions (for model parallelism).
// This is added as an attribute to the StableHLO module.
func (b *Builder) WithNumPartitions(n int) *Builder {
	b.NumPartitions = n
	return b
}

// elementWriter represents elements of ToStableHLO that know how to write themselves.
type elementWriter interface {
	Write(w io.Writer, indentation string) error
}

// NewFunction creates a new function and adds it to the program.
// The function outputs will be determined by the last statement in the function body.
//
// The function name must be unique in the program.
//
// The inputs are the values that the function will receive as arguments.
// The values are not added to the program, they are just used as inputs.
//
// You can also add new inputs later by calling Function.Input.
//
// The function body is defined by calling ops on the function object.
//
// See Function.
func (b *Builder) NewFunction(name string, inputs ...*Value) *Function {
	fn := &Function{
		Builder: b,
		Name:    name,
		Inputs:  inputs,
		values:  slices.Clone(inputs),
	}
	b.functions = append(b.functions, fn)
	return fn
}

const MainFunctionName = "main"

// Main creates the main function of the program.
// It is an alias to Builder.NewFunction("main", inputs...).
//
// The main function is the entry point of the program, and it's the only function that can be called from outside the program.
//
// Every program must have a main function.
//
// Like with NewFunction, you can add new inputs later by calling Function.Input.
func (b *Builder) Main(inputs ...*Value) *Function {
	return b.NewFunction(MainFunctionName, inputs...)
}

const IndentationStep = "  "

// getModuleAttributes returns the attributes for the StableHLO module (StableHLO code) generated.
func (b *Builder) getModuleAttributes() []string {
	var attributes []string
	if b.NumReplicas > 0 {
		attributes = append(attributes, fmt.Sprintf("stablehlo.num_replicas = %d", b.NumReplicas))
	}
	if b.NumPartitions > 0 {
		attributes = append(attributes, fmt.Sprintf(" stablehlo.num_partitions = %d", b.NumPartitions))
	}
	return attributes
}

// Write the StableHLO program (a readable string) to the given writer.
//
// It will write incomplete programs (without a main function or empty statements) without an error
// to help debugging.
//
// See Builder.Build to check and output the program.
func (b *Builder) Write(writer io.Writer) error {
	var err error
	w := func(format string, args ...any) {
		if err != nil {
			// No op if an error was encountered earlier
			return
		}
		_, err = fmt.Fprintf(writer, format, args...)
	}
	we := func(e elementWriter, indentation string) {
		if err != nil {
			// No op if an error was encountered earlier
			return
		}
		err = e.Write(writer, indentation)
	}

	// Write module header
	w("module @%s", NormalizeIdentifier(b.name))
	attrs := b.getModuleAttributes()
	if len(attrs) > 0 {
		w(" attributes {")
		for i, attr := range attrs {
			if i > 0 {
				w(", ")
			}
			w("%s", attr)
		}
		w(" }")
	}
	w(" {\n")

	// Write non-inline functions:
	var count int
	for _, fn := range b.functions {
		if fn.Parent != nil {
			continue
		}
		if count > 0 {
			w("\n\n")
		}
		we(fn, IndentationStep) // Indent functions inside module
		count++
	}
	w("\n}\n") // Close module block
	return err
}

// Build checks the validity and builds the StableHLO program.
//
// If you want the output of an incomplete program (without the checking), use Builder.Write instead.
func (b *Builder) Build() ([]byte, error) {
	hasMain := false
	for _, fn := range b.functions {
		if fn.Name == "main" {
			hasMain = true
		}
		if len(fn.Statements) == 0 {
			return nil, fmt.Errorf("function %q has no statements", fn.Name)
		}
	}
	if !hasMain {
		return nil, errors.New("program must have a main function")
	}

	var buf bytes.Buffer
	err := b.Write(&buf)
	if err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// getChannelHandle generates the channel_handle attribute string.
// It uses the config if provided (for MPMD), or the builder's internal
// counter if not (for SPMD).
func (b *Builder) getChannelHandle(config *types.CollectiveConfig) literalStr {
	var id int
	var typ int64

	if config != nil {
		typ = int64(config.ChannelType) // Use specified type
		if config.ChannelID != nil {
			// Manual ID provided (MPMD case)
			id = *config.ChannelID
		} else {
			// Automatic ID (SPMD case)
			id = b.nextChannelID
			b.nextChannelID++
		}
	} else {
		// Defaults for the simple SPMD case.
		typ = int64(types.CrossReplica)
		id = b.nextChannelID
		b.nextChannelID++
	}

	return literalStrF("#stablehlo.channel_handle<handle = %d, type = %d>", id, typ)
}
