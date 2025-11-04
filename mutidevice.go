package stablehlo

import (
	"fmt"
	"strings"

	"github.com/gomlx/stablehlo/internal/optypes"
	"github.com/gomlx/stablehlo/shapeinference"
	"github.com/gomlx/stablehlo/types"
	"github.com/pkg/errors"
)

// formatReplicaGroups converts a 2D Go slice into the StableHLO dense tensor literal format.
// Example: [[0, 1], [2, 3]] -> "dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>"
func formatReplicaGroups(groups [][]int) literalStr {
	if len(groups) == 0 {
		return "dense<[]> : tensor<0x0xi64>"
	}

	var sb strings.Builder
	sb.WriteString("dense<[")
	for i, group := range groups {
		if i > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString("[")
		for j, replica := range group {
			if j > 0 {
				sb.WriteString(", ")
			}
			sb.WriteString(fmt.Sprintf("%d", replica))
		}
		sb.WriteString("]")
	}
	sb.WriteString("]>")
	sb.WriteString(fmt.Sprintf(" : tensor<%dx%dxi64>", len(groups), len(groups[0])))
	return literalStr(sb.String())
}

// CollectiveBroadcast broadcasts the operand from a source replica to all other replicas.
//
//   - operand: The tensor to be broadcasted. In an SPMD setup, this op will be called on all
//     replicas, but only the operand from the source device (typically the first device in
//     the replica_group) will be used.
//   - replicaGroups: A 2D array defining the communicating device groups. For standard data
//     parallelism, this is typically a single group with all the replica numbers --
//     notice it's not the device numbers by the replica numbers (there is an indirection).
//     Except if the config sets UseGlobalDeviceIDs, in which case they are interpreted as device
//     numbers. E.g., `[[0, 1, 2, 3]]`.
//   - config: Optional configuration of the channels to be used. This is not needed for SPMD programs.
func CollectiveBroadcast(operand *Value, replicaGroups [][]int, config ...*types.CollectiveConfig) (*Value, error) {
	op := optypes.CollectiveBroadcast
	fn := operand.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}

	outputShape, err := shapeinference.CollectiveBroadcast(operand.shape, replicaGroups)
	if err != nil {
		return nil, err
	}

	var cfg *types.CollectiveConfig
	if len(config) > 1 {
		return nil, errors.Errorf("only one config can be provided, got %d", len(config))
	} else if len(config) == 1 {
		cfg = config[0]
	}

	if cfg != nil && (cfg.UseGlobalDeviceIDs || cfg.ChannelType == types.CrossPartition) {
		return nil, errors.Errorf("UseGlobalDeviceIDs or CrossPartition type is not supported for CollectiveBroadcast")
	}

	stmt := fn.addOp(op, outputShape, operand)
	stmt.Attributes = map[string]any{
		"replica_groups": formatReplicaGroups(replicaGroups),
		"channel_handle": fn.Builder.getChannelHandle(cfg),
	}
	return stmt.Outputs[0], nil
}

// AllReduce performs a distributed reduce operation across replicas.
// It is a distributed version of Reduce.
//
//   - operand: The tensor from the *local* replica to be reduced.
//   - replicaGroups: A 2D array defining the communicating device groups, e.g., `[[0, 1, 2, 3]]`.
//   - computation: A closure function that defines the reduction operation (e.g., SUM). It must
//     take two scalar inputs of the operand's dtype and return one scalar output of the same dtype.
//   - replicaGroups: A 2D array defining the communicating device groups. For standard data
//     parallelism, this is typically a single group with all the replica numbers --
//     notice it's not the device numbers by the replica numbers (there is an indirection).
//     Except if the config sets UseGlobalDeviceIDs, in which case they are interpreted as device
//     numbers. E.g., `[[0, 1, 2, 3]]`.
//   - config: Optional configuration of the channels to be used. This is not needed for SPMD programs.
func AllReduce(operand *Value, replicaGroups [][]int, computation *Function, config ...*types.CollectiveConfig) (*Value, error) {
	op := optypes.AllReduce
	fn := operand.fn
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q",
			op, fn.Name)
	}
	if computation.Parent != fn {
		return nil, errors.Errorf("cannot add operation %s because computation is not a StableHLO closure of %s",
			op, fn.Name)
	}

	outputShape, err := shapeinference.AllReduce(
		operand.shape,
		valuesToShapes(computation.Inputs),
		computation.Outputs,
		replicaGroups)
	if err != nil {
		return nil, err
	}

	var cfg *types.CollectiveConfig
	if len(config) > 1 {
		return nil, errors.Errorf("only one config can be provided, got %d", len(config))
	} else if len(config) == 1 {
		cfg = config[0]
	}

	stmt := fn.addOp(op, outputShape, operand)
	stmt.Attributes = map[string]any{
		"replica_groups": formatReplicaGroups(replicaGroups),
		"channel_handle": fn.Builder.getChannelHandle(cfg),
	}
	if cfg != nil && cfg.UseGlobalDeviceIDs {
		stmt.Attributes["use_global_device_ids"] = true
	}
	stmt.AddFunctionParameter("computation", computation)
	return stmt.Outputs[0], nil
}
