#!/usr/bin/env wolframscript
(* ::Package:: *)

var = Rest[$ScriptCommandLine]
dataDir = var[[1]]
fname = var[[2]]
DeltaMin = ToExpression[var[[3]]]
DeltaMax = ToExpression[var[[4]]]
dDelta = ToExpression[var[[5]]]
ss = ToExpression[var[[6]]]
gamma = ToExpression[var[[7]]]
mu = ToExpression[var[[8]]]
rr = ToExpression[var[[9]]]
zeta = ToExpression[var[[10]]]
tauth = ToExpression[var[[11]]]
eta = ToExpression[var[[12]]]
taufc = ToExpression[var[[13]]]
chi = ToExpression[var[[14]]]

Subscript[n, ss]:=2\[Chi] Subscript[\[Tau], fc](Pa^2+Pb^2);
Subscript[T, ss]:= \[Zeta] Subscript[\[Tau], th](\[Eta] r(Pa^2+Pb^2)+Subscript[n, ss]/\[Mu] (Pa+Pb));
Ba:=(-(1/2)+ \[CapitalDelta] I-\[Gamma]/2 I)+(I-r)Pa+(-I-1/\[Mu])Subscript[n, ss]+Subscript[T, ss] I // ComplexExpand;
Bb:=(-(1/2)+ \[CapitalDelta] I+\[Gamma]/2 I)+(I-r)Pb+(-I-1/\[Mu])Subscript[n, ss]+Subscript[T, ss] I // ComplexExpand;
RHSa = Assuming[Pa>=0&& Pb>=0 &&s>=0 &&\[Gamma]>0 &&\[Mu]>0 &&r>0 &&Subscript[\[Tau], th]>0 &&\[Zeta]>0 &&\[Eta]>0 &&\[Chi]>0 &&Subscript[\[Tau], fc]>0 && \[CapitalDelta] \[Element]Reals,Simplify[(Re[Ba]^2+Im[Ba]^2)Pa-s^2]]; 
RHSb = Assuming[Pa>=0&& Pb>=0 &&s>=0 &&\[Gamma]>0 &&\[Mu]>0 &&r>0 &&Subscript[\[Tau], th]>0 &&\[Zeta]>0 &&\[Eta]>0 &&\[Chi]>0 &&Subscript[\[Tau], fc]>0 && \[CapitalDelta] \[Element]Reals,Simplify[(Re[Bb]^2+Im[Bb]^2)Pb-s^2]]; 

FixedPoints[Delta_,ss_,gamma_,mu_,rr_,zeta_,tauth_,eta_,taufc_,chi_] := (
	subs :={\[CapitalDelta]->Delta,s->ss,\[Gamma]->gamma,\[Mu]->mu,r->rr,\[Zeta]->zeta,Subscript[\[Tau], th]->tauth,\[Eta]->eta,Subscript[\[Tau], fc]->taufc,\[Chi]->chi};
	sysInit := RHSa==0 && RHSb==0 && Pa>=0 && Pb>=0;
	sys := sysInit /.subs;
	roots=NSolve[Reduce[Rationalize[sys],{Pa,Pb}],{Pa,Pb}]
	)
	
FPsweep[fname_,DeltaMin_,DeltaMax_,dDelta_,ss_,gamma_,mu_,rr_,zeta_,tauth_,eta_,taufc_,chi_] := (
	DeltaRange := Rationalize[Range[DeltaMin,DeltaMax,dDelta]];
	data:=FixedPoints[#,ss,gamma,mu,rr,zeta,tauth,eta,taufc,chi]&/@ DeltaRange;
	Export[fname,data,"CSV"];
	)
SetDirectory[dataDir];
FPsweep[fname,DeltaMin,DeltaMax,dDelta,ss,gamma,mu,rr,zeta,tauth,eta,taufc,chi];
