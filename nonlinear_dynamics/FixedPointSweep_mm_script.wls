#!/usr/bin/env wolframscript
(* ::Package:: *)

var = Rest[$ScriptCommandLine]
dataDir = var[[1]]
fname = var[[2]]
DeltaMin = ToExpression[var[[3]]];
DeltaMax = ToExpression[var[[4]]];
nDelta = ToExpression[var[[5]]];
ss = ToExpression[var[[6]]];
gamma = ToExpression[var[[7]]];
mu = ToExpression[var[[8]]];
rr = ToExpression[var[[9]]];
zeta = ToExpression[var[[10]]];
tauth = ToExpression[var[[11]]];
rho = ToExpression[var[[12]]];
taufc = ToExpression[var[[13]]];
chi =  ToExpression[var[[14]]];
alpha = ToExpression[var[[15]]];
psi = ToExpression[var[[16]]];
tauxfc = ToExpression[var[[17]]];
eta = ToExpression[var[[18]]];
nk = ToExpression[var[[19]]];
DeltaRange = Subdivide[DeltaMin,DeltaMax,nDelta];
subs :={s->ss,\[Gamma]->gamma,\[Mu]->mu,r->rr,\[Zeta]->zeta,Subscript[\[Tau], th]->tauth,\[Rho]->rho,Subscript[\[Tau],c]->taufc,\[Chi]->chi,\[Alpha]->alpha,\[Psi]->psi,Subscript[\[Tau], xfc]->tauxfc,\[Eta]->eta};
n =\[Chi] Subscript[\[Tau], c](Pa^2+Pb^2) //.subs;
T = \[Zeta] Subscript[\[Tau], th](\[Rho] r(Pa^2+Pb^2) + n (Pa+Pb)/\[Mu]) //.subs;
Ba =(-(1/2)+ \[CapitalDelta] I-\[Gamma]/2 I)+(I-r)Pa+(-I-1/\[Mu]) n +T I // ComplexExpand //.subs;
Bb =(-(1/2)+ \[CapitalDelta] I+\[Gamma]/2 I)+(I-r)Pb+(-I-1/\[Mu]) n + T I // ComplexExpand //.subs;
RHSa = Assuming[Pa>=0&& Pb>=0 &&s>=0 &&\[Gamma]>0 &&\[Mu]>0 &&r>0 &&Subscript[\[Tau], th]>0 &&\[Zeta]>0 &&\[Eta]>0 &&\[Chi]>0 &&Subscript[\[Tau], c]>0 && \[CapitalDelta] \[Element]Reals,Simplify[(Re[Ba]^2+Im[Ba]^2)Pa - \[Eta] s^2]]/.subs; 
RHSb = Assuming[Pa>=0&& Pb>=0 &&s>=0 &&\[Gamma]>0 &&\[Mu]>0 &&r>0 &&Subscript[\[Tau], th]>0 &&\[Zeta]>0 &&\[Eta]>0 &&\[Chi]>0 &&Subscript[\[Tau], c]>0 && \[CapitalDelta] \[Element]Reals,Simplify[(Re[Bb]^2+Im[Bb]^2)Pb - \[Eta] s^2]]/.subs; 
sysInit := RHSa==0. && RHSb==0. && Pa>=0. && Pb>=0.;
sys = sysInit /.subs;
LaunchKernels[nk];
pts = ParallelTable[NSolve[sys/.{\[CapitalDelta]->d},{Pa,Pb}],{d,DeltaRange}]; 
SetDirectory[dataDir];
Export[fname,pts,"CSV"];

