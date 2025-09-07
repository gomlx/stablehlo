package stablehlo

import (
	"bytes"
	"fmt"
	"io"
	"slices"

	"github.com/pkg/errors"
)

// Builder is used to construct a ToStableHLO program.
// See New.
type Builder struct {
	name   string
	parent *Builder

	// functions holds all the functions created in the builder's scope.
	functions []*Function
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
// You can also add new inputs later by calling Function.NewInput.
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
// Like with NewFunction, you can add new inputs later by calling Function.NewInput.
func (b *Builder) Main(inputs ...*Value) *Function {
	return b.NewFunction(MainFunctionName, inputs...)
}

// NewInlineFunction creates an unnamed inline function that can be used as an argument to operations like
// Reduce, ReduceWindow, ScatterAndUpdate, etc.
//
// After created, the InlineFunction should not be changed. But it can be used multiple times.
//
// The inputs are the values that the function will receive as arguments.
// You can also add new inputs later by calling Function.NewInput.
//
// The function body is defined by calling ops on the function object.
func (b *Builder) NewInlineFunction(inputs ...*Value) *Function {
	inlineFn := b.NewFunction("", inputs...)
	inlineFn.IsInline = true
	return inlineFn
}

const IndentationStep = "  "

// Write the StableHLO program (a readable string) to the given writer.
//
// It will write incomplete programs (without a main function or empty statements) without an error
// to help debugging.
//
// See Builder.Build to check and output the program.
func (b *Builder) Write(writer io.Writer) error {
	indentation := ""
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
	for i, fn := range b.functions {
		if i > 0 {
			w("\n\n")
		}
		we(fn, indentation)
	}
	w("\n")
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
