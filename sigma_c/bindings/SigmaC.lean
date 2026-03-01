/-
  Sigma-C Lean 4 Binding
  =======================
  Copyright (c) 2025 ForgottenForge.xyz

  Lean 4 implementation of critical susceptibility analysis.
  Computes sigma_c (critical point) from epsilon-observable data
  by finding the peak of the numerical derivative (susceptibility).

  SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
-/

namespace SigmaC

/-- Result of a criticality analysis. -/
structure Result where
  sigmaC   : Float   -- Critical point (epsilon value at peak susceptibility)
  kappa    : Float   -- Peak sharpness (chi_max / chi_mean)
  chiMax   : Float   -- Maximum susceptibility value
  peakIdx  : Nat     -- Index of the peak in the chi array
  deriving Repr, Inhabited

/-- Compute the absolute value of a Float. -/
private def floatAbs (x : Float) : Float :=
  if x < 0.0 then -x else x

/-- Compute finite differences: diff[i] = (obs[i+1] - obs[i]) / (eps[i+1] - eps[i]). -/
private def finiteDiff (epsilon : Array Float) (observable : Array Float) : Array Float :=
  let n := min epsilon.size observable.size
  if n < 2 then #[]
  else
    let mut result : Array Float := #[]
    for i in [:n - 1] do
      let dEps := epsilon[i + 1]! - epsilon[i]!
      let dObs := observable[i + 1]! - observable[i]!
      let chi := if floatAbs dEps > 1e-15 then dObs / dEps else 0.0
      result := result.push (floatAbs chi)
    result

/-- Find the index of the maximum element in an array of Floats. -/
private def argmax (arr : Array Float) : Nat :=
  if arr.isEmpty then 0
  else
    let mut bestIdx : Nat := 0
    let mut bestVal : Float := arr[0]!
    for i in [1:arr.size] do
      if arr[i]! > bestVal then
        bestIdx := i
        bestVal := arr[i]!
    bestIdx

/-- Compute the mean of a Float array. -/
private def mean (arr : Array Float) : Float :=
  if arr.isEmpty then 0.0
  else
    let sum := arr.foldl (· + ·) 0.0
    sum / arr.size.toFloat

/--
  Compute critical susceptibility from epsilon-observable data.

  The susceptibility chi = |dO/d(epsilon)| is computed via finite differences.
  The critical point sigma_c is the epsilon value where chi reaches its maximum.
  The sharpness kappa = chi_max / chi_mean quantifies how pronounced the peak is.

  ## Example
  ```lean
  let eps := #[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
  let obs := #[1.0, 0.95, 0.8, 0.4, 0.1, 0.05]
  let result := SigmaC.computeSigmaC eps obs
  -- result.sigmaC is near 0.3 (steepest drop)
  ```
-/
def computeSigmaC (epsilon : Array Float) (observable : Array Float) : Result :=
  let chi := finiteDiff epsilon observable
  if chi.isEmpty then
    { sigmaC := 0.0, kappa := 0.0, chiMax := 0.0, peakIdx := 0 }
  else
    let peakIdx := argmax chi
    let chiMax := chi[peakIdx]!
    let chiMean := mean chi
    let kappa := if chiMean > 1e-15 then chiMax / chiMean else 0.0
    let sigmaC := if peakIdx < epsilon.size then epsilon[peakIdx]! else 0.0
    { sigmaC := sigmaC, kappa := kappa, chiMax := chiMax, peakIdx := peakIdx }

/-- Streaming criticality calculator with windowed updates. -/
structure StreamingCalc where
  windowSize       : Nat
  epsilonBuffer    : Array Float
  observableBuffer : Array Float
  currentSigmaC    : Float
  deriving Repr, Inhabited

/-- Create a new streaming calculator with the given window size. -/
def StreamingCalc.new (windowSize : Nat := 100) : StreamingCalc :=
  { windowSize := windowSize,
    epsilonBuffer := #[],
    observableBuffer := #[],
    currentSigmaC := 0.0 }

/--
  Push a new data point and recompute sigma_c if enough data is available.
  Returns updated calculator and current sigma_c estimate.
-/
def StreamingCalc.update (s : StreamingCalc) (eps : Float) (obs : Float) : StreamingCalc × Float :=
  let epsBuf := s.epsilonBuffer.push eps
  let obsBuf := s.observableBuffer.push obs
  -- Trim to window size
  let epsBuf := if epsBuf.size > s.windowSize then epsBuf.extract 1 epsBuf.size else epsBuf
  let obsBuf := if obsBuf.size > s.windowSize then obsBuf.extract 1 obsBuf.size else obsBuf
  -- Recompute if enough data
  if epsBuf.size >= 10 then
    let result := computeSigmaC epsBuf obsBuf
    let s' := { s with epsilonBuffer := epsBuf, observableBuffer := obsBuf,
                        currentSigmaC := result.sigmaC }
    (s', result.sigmaC)
  else
    let s' := { s with epsilonBuffer := epsBuf, observableBuffer := obsBuf }
    (s', s.currentSigmaC)

/--
  Rank candidate observables by their susceptibility sharpness.
  Each column of the data matrix is a candidate observable, all sharing
  the same epsilon axis.

  Returns an array of (columnIndex, kappa) pairs sorted by descending kappa.
-/
def rankObservables (epsilon : Array Float) (candidates : Array (Array Float)) : Array (Nat × Float) :=
  let mut scores : Array (Nat × Float) := #[]
  for i in [:candidates.size] do
    let result := computeSigmaC epsilon candidates[i]!
    scores := scores.push (i, result.kappa)
  -- Sort by descending kappa (insertion sort for simplicity)
  let mut sorted := scores
  for i in [1:sorted.size] do
    let mut j := i
    while j > 0 && (sorted[j - 1]!).2 < (sorted[j]!).2 do
      let tmp := sorted[j]!
      sorted := sorted.set! j (sorted[j - 1]!)
      sorted := sorted.set! (j - 1) tmp
      j := j - 1
  sorted

end SigmaC
