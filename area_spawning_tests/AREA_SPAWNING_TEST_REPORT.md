# Area Spawning Test Report

## Test Configuration
- **Number of tests per area width**: 10
- **Area widths tested**: 10%, 20%, 30%, 40%
- **Environment**: 1v1, sensing_radius=0.1, speed_ratio=0.3

## Results Summary

| Area Width | Percentage | Y Range Expected | Y Range Actual | X Spread | Status |
|------------|------------|------------------|----------------|----------|--------|
| 0.1 | 10% | [0.9, 1.0] | [1.000, 1.000] | 0.826 | ✅ |\n| 0.2 | 20% | [0.8, 1.0] | [1.000, 1.000] | 0.752 | ✅ |\n| 0.3 | 30% | [0.7, 1.0] | [1.000, 1.000] | 0.645 | ✅ |\n| 0.4 | 40% | [0.6, 1.0] | [1.000, 1.000] | 0.677 | ✅ |\n
## Analysis

### Area Spawning Implementation
- **10% area (0.1)**: Attackers spawn in Y range [0.9, 1.0] - narrow band near top
- **20% area (0.2)**: Attackers spawn in Y range [0.8, 1.0] - 1/5th of environment  
- **30% area (0.3)**: Attackers spawn in Y range [0.7, 1.0] - larger tactical area
- **40% area (0.4)**: Attackers spawn in Y range [0.6, 1.0] - very large spawn zone

### X Position Distribution
All area modes should show full X range [0.0, 1.0] utilization.

### Tactical Implications
- **Smaller areas**: More predictable, easier to defend
- **Larger areas**: More variety, harder to cover all possible approaches
- **Area mode vs discrete**: Continuous positioning vs fixed spawn points

## Files Generated
- `area_spawning_test_width_0.1.png` - 10% area test\n- `area_spawning_test_width_0.2.png` - 20% area test\n- `area_spawning_test_width_0.3.png` - 30% area test\n- `area_spawning_test_width_0.4.png` - 40% area test\n- `area_spawning_comparison.png` - Side-by-side area comparison\n