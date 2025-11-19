package shardy

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestShardSpec_ToStableHLO(t *testing.T) {
	mesh, err := NewDeviceMesh("test_mesh", []int{4, 2}, []string{"z", "a"})
	require.NoError(t, err)
	testCases := []struct {
		name     string
		spec     *ShardSpec
		expected string
	}{
		{
			name:     "Replicated",
			spec:     NewShardSpec(mesh).AddReplicated(),
			expected: "sharding<@test_mesh, [{}], replicated={a, z}>",
		},
		{
			name:     "Sharded",
			spec:     NewShardSpec(mesh).AddShardedAxis("z"),
			expected: "sharding<@test_mesh, [{z}], replicated={a}>",
		},
		{
			name:     "Sharded with multiple axes",
			spec:     NewShardSpec(mesh).AddShardedAxis("z", "a"),
			expected: "sharding<@test_mesh, [{z, a}]>",
		},
		{
			name: "Sharded with sub-axis",
			spec: &ShardSpec{
				Mesh: mesh,
				Axes: []TensorAxisSpec{
					{MeshAxes: []MeshAxisSpec{{AxisName: "a", PreSize: 1, Size: 2}}},
				},
			},
			expected: "sharding<@test_mesh, [{a:(1)2}], replicated={z}>",
		},
		{
			name:     "Opened",
			spec:     &ShardSpec{Mesh: mesh, Axes: []TensorAxisSpec{{Opened: true}}},
			expected: "sharding<@test_mesh, [{?}], replicated={a, z}>",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			require.Equal(t, tc.expected, tc.spec.ToStableHLO())
		})
	}
}

func TestShardSpec_Validate(t *testing.T) {
	mesh, err := NewDeviceMesh("test_mesh", []int{2, 8}, []string{"z", "a"})
	require.NoError(t, err)
	testCases := []struct {
		name        string
		spec        *ShardSpec
		expectError bool
	}{
		{
			name:        "Valid sharding",
			spec:        NewShardSpec(mesh).AddShardedAxis("z"),
			expectError: false,
		},
		{
			name:        "Unknown mesh axis",
			spec:        NewShardSpec(mesh).AddShardedAxis("x"),
			expectError: true,
		},
		{
			name: "Valid sub-axis",
			spec: &ShardSpec{
				Mesh: mesh,
				Axes: []TensorAxisSpec{
					{MeshAxes: []MeshAxisSpec{{AxisName: "a", PreSize: 2, Size: 4}}},
				},
			},
			expectError: false,
		},
		{
			name: "Invalid sub-axis (PreSize)",
			spec: &ShardSpec{
				Mesh: mesh,
				Axes: []TensorAxisSpec{
					{MeshAxes: []MeshAxisSpec{{AxisName: "a", PreSize: 0, Size: 4}}},
				},
			},
			expectError: true,
		},
		{
			name: "Invalid sub-axis (Size)",
			spec: &ShardSpec{
				Mesh: mesh,
				Axes: []TensorAxisSpec{
					{MeshAxes: []MeshAxisSpec{{AxisName: "a", PreSize: 2, Size: 5}}},
				},
			},
			expectError: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.spec.Validate()
			if tc.expectError {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}
		})
	}
}
