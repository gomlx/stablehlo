package shardy

import (
	"github.com/pkg/errors"
)

// ShardSpec (also known as PartitionSpec in JAX) defines how a logical tensor is to be sharded (partitioned) across
// a DeviceMesh. This is used by Shardy, and is based on its documentation in [1].
//
// The definition is per axis of the logical tensor -- and not per axis of the Mesh, common confusion.
// If not all axes of the Tensor are defined, the tail axes are considered simply to be replicated across the whole
// mesh.
//
// Each tensor axis can be replicated or sharded across one or more mesh axes.
//
// Example:
//
//	mesh := NewDeviceMesh("my_mesh", []int{2, 2}, []string{"data", "model"})
//
//	// Input's "batch" axis is sharded across the "data" axis of the mesh.
//	inputSharding := MakeShardSpec(mesh.Name()).AddShardedAxis("data")
//
//	// First axis is replicated, second is shared across "model" devices
//	variableSharding := MakeShardSpec(mesh.Name()).AddReplicated().AddShardedAxis("model")
//
//	// Second axis is sharded across both "data" and "model" devices.
//	 largeWeights := MakeShardSpec(mesh.Name()).AddReplicated().AddShardedAxis("data", "model")
//
// There are two advanced features supported but not tested (pls if you need let us know how it goes, or if you find
// any issues):
//
//  1. The tensor can also be sharded across mesh "sub-axes" -- seed detailed documentation in [1]
//  2. If using ShardSpec for hints, instead of mesh axes one can give an "open" (in StableHLO marked as "?")
//     axis, with the semantics that XLA Shardy can choose any mesh axis (or axes) to shard the tensor. See [1].
//
// [1] https://github.com/openxla/shardy/blob/main/docs/sharding_representation.md
type ShardSpec struct {
	Mesh *DeviceMesh
	Axes []TensorAxisSpec
}

// TensorAxisSpec specifies how a tensor axis is to be sharded (or replicated).
// See details in ShardSpec.
//
// Usually, one would create this using ShardSpec.AddAxis or ShardSpec.AddReplicated
type TensorAxisSpec struct {
	MeshAxes []MeshAxisSpec
	Opened   bool // If opened to further sharding.
}

type MeshAxisSpec struct {
	AxisName string

	// PreSize, Size are only set if defining a sub-axis of the mesh.
	PreSize, Size int
}

// NewShardSpec creates a new ShardSpec.
func NewShardSpec(mesh *DeviceMesh) *ShardSpec {
	return &ShardSpec{mesh, make([]TensorAxisSpec, 0)}
}

// AddShardedAxis adds a new sharded axis to the ShardSpec using one or more mesh axes.
//
// It returns itself, so calls can be chained.
func (s *ShardSpec) AddShardedAxis(meshAxisName string, moreMeshAxesNames ...string) *ShardSpec {
	axisSpec := TensorAxisSpec{MeshAxes: []MeshAxisSpec{{AxisName: meshAxisName}}}
	for _, meshAxisName := range moreMeshAxesNames {
		axisSpec.MeshAxes = append(axisSpec.MeshAxes, MeshAxisSpec{AxisName: meshAxisName})
	}
	s.Axes = append(s.Axes, axisSpec)
	return s
}

// AddReplicated adds a new replicated axis to the ShardSpec.
//
// It returns itself, so calls can be chained.
func (s *ShardSpec) AddReplicated() *ShardSpec {
	s.Axes = append(s.Axes, TensorAxisSpec{})
	return s
}

// Rank returns the number of axes this ShardSpec describes.
//
// Notice this may be smaller than the rank of the tensor using it: if a tensor axis is not defined in ShardSpec,
// it is assumed to be replicated.
func (s *ShardSpec) Rank() int {
	return len(s.Axes)
}

// IsReplicated returns true if the tensor is fully replicated
// (i.e., not sharded along any axis and not marked as "open").
func (s *ShardSpec) IsReplicated() bool {
	for _, axisSpec := range s.Axes {
		if axisSpec.MeshAxes != nil || axisSpec.Opened {
			return false
		}
	}
	return true
}

// Validate checks that the ShardSpec is valid for the given mesh.
func (s *ShardSpec) Validate() error {
	for i, axisSpec := range s.Axes {
		for _, meshAxisSpec := range axisSpec.MeshAxes {
			axisName := meshAxisSpec.AxisName
			if axisName == "" {
				return errors.Errorf("ShardSpec axis %d refers to empty mesh axis name", i)
			}
			if _, ok := s.Mesh.nameToAxis[axisName]; !ok {
				return errors.Errorf("ShardSpec axis #%d refers to unknown mesh axis %q",
					i, axisName)
			}
		}
	}
	return nil
}
