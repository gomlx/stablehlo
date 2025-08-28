package stablehlo

import (
	"fmt"
	"io"
	"strings"

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

// New creates a new Builder object holding a computation graph in construction.
//
// From a builder you can create functtions.
// For each function you create operations (ops) one by one, until you defined the desired computation.
//
// You have to define the "main" function for your StableHLO program.
//
// Once you are all set, call Build and it will return the StableHLO program as a string, that can
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
	Write(w io.Writer) error
}

// NewFunction creates a new function and adds it to the program.
// The function outputs will be determined by the last statement in the function body.
func (b *Builder) NewFunction(name string, inputs []*Value) *Function {
	fn := &Function{
		Name:   name,
		Inputs: inputs,
	}
	b.functions = append(b.functions, fn)
	return fn
}

// Write the StableHLO program (a readable string) to the given writer.
//
// It will write incomplete programs (without a main function or empty statements) without an error,
// to help debuggging.
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
	we := func(e elementWriter) {
		if err != nil {
			// No op if an error was encountered earlier
			return
		}
		err = e.Write(writer)
	}
	for i, fn := range b.functions {
		if i > 0 {
			w("\n\n")
		}
		we(fn)
	}
	w("\n")
	return err
}

// Build checks the validity and builds the StableHLO program.
//
// If you want the output of an incomplete program (without the checing), use Builder.Write instead.
func (b *Builder) Build() (string, error) {
	hasMain := false
	for _, fn := range b.functions {
		if fn.Name == "main" {
			hasMain = true
		}
		if len(fn.Statements) == 0 {
			return "", fmt.Errorf("function %q has no statements", fn.Name)
		}
	}
	if !hasMain {
		return "", errors.New("program must have a main function")
	}

	var sb strings.Builder
	err := b.Write(&sb)
	if err != nil {
		return "", err
	}
	return sb.String(), nil
}
