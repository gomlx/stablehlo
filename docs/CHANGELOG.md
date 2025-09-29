# v0.0.5: 2025/09/29

- Updated dependency to new Gopjrt v0.8.2 -- issues with the CUDA PJRT backward compatibility (lack of).

# v0.0.4: 2025/09/28

- Added support for comparison of bool values, and added corresponding tests.

# v0.0.3: 2025/09/27

- Fixed wrong checking for during shapeinference.Gather 

# v0.0.2: 2025/09/21 - Small fixes, making sure stablehlo passes all GoMLX graph tests.

- Fixed rendering of odd floats.


# v0.0.1: 2025/09/21 - Same XlaBuilder ops coverage

- Initial release.
- Coverage of all XlaBuilder ops, so it can be used by [GoMLX](https://github.com/gomlx/gomlx) as a drop-in 
  replacement for the XLA backends.