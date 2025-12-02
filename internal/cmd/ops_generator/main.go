package main

import "log"

func main() {
	GenerateBinaryOps()
	GenerateUnaryOps()
}

func must(err error) {
	if err != nil {
		log.Fatalf("Failed: %+v", err)
	}
}

func must1[T any](value T, err error) T {
	must(err)
	return value
}
