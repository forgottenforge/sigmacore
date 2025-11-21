(* Sigma-C Mathematica Package *)
(* ============================= *)
(* Copyright (c) 2025 ForgottenForge.xyz *)

BeginPackage["SigmaC`"]

CriticalPoint::usage = "CriticalPoint[data] computes the critical point from {epsilon, observable} data."
ComputeSigmaC::usage = "ComputeSigmaC[epsilon, observable] computes criticality metrics."
ListPlotCritical::usage = "ListPlotCritical[data] plots data with critical point highlighted."
StreamingSigmaC::usage = "StreamingSigmaC[windowSize] creates a streaming calculator."

Begin["`Private`"]

(* Core computation *)
ComputeSigmaC[epsilon_List, observable_List] := Module[
    {chi, peakIdx, sigmaC, chiMax, kappa},
    
    (* Compute susceptibility *)
    chi = Differences[observable] / Differences[epsilon];
    
    (* Find peak *)
    peakIdx = First[Ordering[Abs[chi], -1]];
    sigmaC = epsilon[[peakIdx]];
    chiMax = Abs[chi[[peakIdx]]];
    
    (* Compute kappa *)
    kappa = If[Length[chi] > 2,
        Abs[Differences[chi][[peakIdx]]] / chiMax,
        1.0
    ];
    
    <|
        "SigmaC" -> sigmaC,
        "Kappa" -> kappa,
        "ChiMax" -> chiMax,
        "PeakLocation" -> sigmaC
    |>
]

(* Convenience function *)
CriticalPoint[data_List] := Module[
    {epsilon, observable},
    epsilon = data[[All, 1]];
    observable = data[[All, 2]];
    ComputeSigmaC[epsilon, observable]
]

(* Visualization *)
ListPlotCritical[data_List, opts___] := Module[
    {result, sigmaC, plot},
    
    result = CriticalPoint[data];
    sigmaC = result["SigmaC"];
    
    plot = ListPlot[data,
        opts,
        Epilog -> {
            Red, PointSize[0.02],
            Point[{sigmaC, Interpolation[data][sigmaC]}],
            Text[Style[StringForm["σ_c = ``", NumberForm[sigmaC, {4, 3}]], 14],
                {sigmaC, Interpolation[data][sigmaC]}, {0, -2}]
        },
        PlotLabel -> "Criticality Analysis",
        AxesLabel -> {"ε", "Observable"}
    ];
    
    plot
]

(* Streaming calculator *)
StreamingSigmaC[windowSize_: 100] := Module[
    {buffer = {}, currentSigmaC = 0.0},
    
    Function[{epsilon, observable},
        AppendTo[buffer, {epsilon, observable}];
        
        (* Keep only recent window *)
        If[Length[buffer] > windowSize,
            buffer = Take[buffer, -windowSize]
        ];
        
        (* Recompute *)
        If[Length[buffer] >= 10,
            result = CriticalPoint[buffer];
            currentSigmaC = result["SigmaC"]
        ];
        
        currentSigmaC
    ]
]

End[]
EndPackage[]

(* Example usage:
   data = Table[{x, Sin[10 x]}, {x, 0, 0.5, 0.01}];
   result = CriticalPoint[data];
   ListPlotCritical[data]
*)
