package utils

import (
	"fmt"

	"github.com/gomlx/gopjrt/dtypes"
)

func DTypeToStableHLO(dtype dtypes.DType) string {
	switch dtype {
	case dtypes.F64:
		return "f64"
	case dtypes.F32:
		return "f32"
	case dtypes.F16:
		return "f16"
	case dtypes.BFloat16:
		return "bf16"
	case dtypes.S64:
		return "i64"
	case dtypes.S32:
		return "i32"
	case dtypes.S16:
		return "i16"
	case dtypes.S8:
		return "i8"
	case dtypes.U64:
		return "ui64"
	case dtypes.U32:
		return "ui32"
	case dtypes.U16:
		return "ui16"
	case dtypes.U8:
		return "ui8"
	case dtypes.Bool:
		return "i1"
	case dtypes.Complex64:
		return "complex<f32>"
	case dtypes.Complex128:
		return "complex<f64>"
	default:
		return fmt.Sprintf("unknown_dtype<%s>", dtype.String())
	}
}
