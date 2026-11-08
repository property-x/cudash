@cudarc @planned
Feature: Lending a cudash buffer to cudarc calls
  Planned API from plan/api.md. `Interop::lend` and `lend_mut` give a cudarc call temporary
  access to a cudash slice. Borrow rules and the wait on Vulkan work are checked for us.

  Background:
    Given an NVIDIA GPU paired with a Vulkan device
    And a cudash slice of 100 f32 elements, allocated on the cudash stream

  Scenario: A lent slice works as a cudarc copy destination
    When I lend the slice mutably and copy host values into it with cudarc's memcpy_htod
    Then after the lend ends, cudash reads the same values back

  Scenario: A lent slice works as a cudarc kernel argument
    When I lend the slice mutably and pass it to a cudarc PTX kernel with builder.arg
    Then after the lend ends, cudash reads the kernel's output

  Scenario: Lending waits for pending cudash work on the slice
    Given cudash has queued a fill of the slice that has not finished
    When I lend the slice to cudarc
    Then cudarc sees the filled values

  Scenario: Lending a slice that was not allocated for export is an error
    Given a slice allocated in a way that cannot be exported
    When I lend it
    Then the result is an error

  Scenario: Using the slice from cudash while it is lent does not compile
    When the code uses the slice from cudash inside a lend scope
    Then the compiler rejects the program
    # Compile-fail test. Needs a trybuild-style check.

  Scenario: Lending twice in a row reuses the first import
    When I lend the slice, end the lend, and lend it again
    Then the second lend does not import the memory a second time
    # Whether to cache imports is undecided. This scenario states the goal, and measurement
    # decides it. See plan/api.md, "Import cost".
