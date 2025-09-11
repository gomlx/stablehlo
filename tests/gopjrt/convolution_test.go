package gopjrt

import (
	"testing"

	"github.com/gomlx/gopjrt/pjrt"
)

func TestConvolution(t *testing.T) {
	iterateClientsAndTest(t, testConvolution)
}

func testConvolution(t *testing.T, client *pjrt.Client) {
	
}
