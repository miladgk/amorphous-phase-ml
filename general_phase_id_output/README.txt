PhaseID/SmoothedPhaseID User Guide
===================================

Phase labels (0 = Cu50Zr50, 1 = Cu46Zr54, 2 = Cu64Zr36)

IMPORTANT: The Phase A (Cu50Zr50) vs Phase B (Cu46Zr54) boundary
represents a CONTINUOUS COMPOSITIONAL GRADIENT, not a discrete
structural phase boundary. Direct local-Cu-fraction regression
confirmed: the Cu fraction changes smoothly from ~0.40 to ~0.45
over ~40A across this boundary (R²=0.938, MAE=0.024).

This means discrete PhaseID labels in this region are inherently
ambiguous. The LocalCuFraction column provides the continuously
predicted Cu fraction (0.0-1.0 scale from the regression model)
which better represents the actual structural variation.

The Phase C (Cu64Zr36) boundary IS a genuine sharp structural
transition (Cu fraction changes by 0.144 over 12A). PhaseID/
SmoothedPhaseID are reliable for this separation.

For visualization in OVITO:
  - Color by LocalCuFraction for smooth gradients in A/B region
  - Color by SmoothedPhaseID for sharp classification boundaries
