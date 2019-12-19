#!/usr/bin/env wolframscript
(* ::Package:: *)

var = Rest[$ScriptCommandLine]
dataDir = var[[1]]
fname = var[[2]]
DeltaMin = ToExpression[var[[3]]]
DeltaMax = ToExpression[var[[4]]]
nDelta = ToExpression[var[[5]]]
ss = Rationalize[ToExpression[var[[6]]],.01]
gamma = Rationalize[ToExpression[var[[7]]],0.1]
mu = Rationalize[ToExpression[var[[8]]],0.5]
rr = Rationalize[ToExpression[var[[9]]],0.02]
zeta = Rationalize[ToExpression[var[[10]]],0.01]
tauth = Rationalize[ToExpression[var[[11]]],0.5]
rho = Rationalize[ToExpression[var[[12]]],0.1]
taufc = Rationalize[ToExpression[var[[13]]],0.005]
chi = Rationalize[ToExpression[var[[14]]],0.1]
alpha = Rationalize[ToExpression[var[[15]]],0.1]
psi = Rationalize[ToExpression[var[[16]]],0.1]
tauxfc = Rationalize[ToExpression[var[[17]]],0.01]
eta = Rationalize[ToExpression[var[[18]]],0.02]
wp = ToExpression[var[[19]]]

Subscript[ntot, ss]:=\[Chi] Subscript[\[Tau], fc](Pa^2+Pb^2) + \[Alpha] Subscript[\[Tau], fc](Pa+Pb);
Subscript[ndiff, ss]:=(\[Chi](Pa^2-Pb^2) + \[Alpha](Pa-Pb))/( 1/Subscript[\[Tau], fc] +  2/Subscript[\[Tau], xfc]);
Subscript[na, ss]:= Subscript[ntot, ss] + Subscript[ndiff, ss];
Subscript[nb, ss]:= Subscript[ntot, ss] - Subscript[ndiff, ss];
Subscript[T, ss]:= \[Zeta] Subscript[\[Tau], th](\[Rho] r(Pa^2+Pb^2)+ \[Psi] \[Alpha] r (Pa+Pb)/(2\[Chi]) + ( Subscript[na, ss]Pa + Subscript[nb, ss]Pb )/\[Mu]);
Ba:=(-(1/2)+ \[CapitalDelta] I-\[Gamma]/2 I)+(I-r)Pa+(-I-1/\[Mu])Subscript[na, ss]+Subscript[T, ss] I // ComplexExpand;
Bb:=(-(1/2)+ \[CapitalDelta] I+\[Gamma]/2 I)+(I-r)Pb+(-I-1/\[Mu])Subscript[nb, ss]+Subscript[T, ss] I // ComplexExpand;
RHSa = Assuming[Pa>=0&& Pb>=0 &&s>=0 &&\[Gamma]>0 &&\[Mu]>0 &&r>0 &&Subscript[\[Tau], th]>0 &&\[Zeta]>0 &&\[Eta]>0 &&\[Chi]>0 &&Subscript[\[Tau], fc]>0 && \[CapitalDelta] \[Element]Reals,Simplify[(Re[Ba]^2+Im[Ba]^2)Pa - \[Eta] s^2]]; 
RHSb = Assuming[Pa>=0&& Pb>=0 &&s>=0 &&\[Gamma]>0 &&\[Mu]>0 &&r>0 &&Subscript[\[Tau], th]>0 &&\[Zeta]>0 &&\[Eta]>0 &&\[Chi]>0 &&Subscript[\[Tau], fc]>0 && \[CapitalDelta] \[Element]Reals,Simplify[(Re[Bb]^2+Im[Bb]^2)Pb - \[Eta] s^2]]; 

FixedPoints[Delta_,ss_,gamma_,mu_,rr_,zeta_,tauth_,rho_,taufc_,chi_,alpha_,psi_,tauxfc_,eta_,wp_] := (
	subs :={\[CapitalDelta]->Delta,s->ss,\[Gamma]->gamma,\[Mu]->mu,r->rr,\[Zeta]->zeta,Subscript[\[Tau], th]->tauth,\[Rho]->rho,Subscript[\[Tau], fc]->taufc,\[Chi]->chi,\[Alpha]->alpha,\[Psi]->psi,Subscript[\[Tau], xfc]->tauxfc,\[Eta]->eta};
	sysInit := RHSa==0 && RHSb==0 && Pa>=0 && Pb>=0;
	sys := sysInit /.subs;
	roots=NSolve[Reduce[Rationalize[sys],{Pa,Pb}],{Pa,Pb},WorkingPrecision->wp]
	)
	
FPsweep[fname_,DeltaMin_,DeltaMax_,nDelta_,ss_,gamma_,mu_,rr_,zeta_,tauth_,rho_,taufc_,chi_,alpha_,psi_,tauxfc_,eta_,wp_] := (
	DeltaRange := Rationalize[Subdivide[DeltaMin,DeltaMax,nDelta]];
	data:=FixedPoints[#,ss,gamma,mu,rr,zeta,tauth,rho,taufc,chi,alpha,psi,tauxfc,eta,wp]&/@ DeltaRange;
	Export[fname,data,"CSV"];
	)
SetDirectory[dataDir];
FPsweep[fname,DeltaMin,DeltaMax,nDelta,ss,gamma,mu,rr,zeta,tauth,rho,taufc,chi,alpha,psi,tauxfc,eta,wp];
