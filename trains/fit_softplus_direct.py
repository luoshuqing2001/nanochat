"""Constrained value/derivative fit; coefficients use t=(abs(s)-lo)/(hi-lo)."""
import numpy as np
from numpy.polynomial import polynomial as p

def fit(lo,hi,degree):
 h=hi-lo;t=np.linspace(0,1,20001);a=lo+h*t
 f=np.logaddexp(0,-a);df=-h/(1+np.exp(a))
 f0,f1=f[0],f[-1];d0,d1=df[0],df[-1]
 herm=np.array([f0,d0,3*(f1-f0)-2*d0-d1,2*(f0-f1)+d0+d1])
 base=np.array([0,0,1,-2,1.])
 mat=np.array([p.polyval(t,p.polymul(base,[0]*i+[1])) for i in range(degree-3)]).T
 dm=np.array([p.polyval(t,p.polyder(p.polymul(base,[0]*i+[1]))) for i in range(degree-3)]).T
 weights=1/f
 c=np.linalg.lstsq(np.concatenate([mat*weights[:,None],dm*weights[:,None]]),np.concatenate([(f-p.polyval(t,herm))*weights,(df-p.polyval(t,p.polyder(herm)))*weights]),rcond=None)[0]
 coeff=p.polyadd(herm,p.polymul(base,c));v=p.polyval(t,coeff);d=p.polyval(t,p.polyder(coeff))/h
 print(lo,hi,degree,'relvalue',max(abs(v-f)/f),'relgrad',max(abs(d-df/h)/(-df/h)))
 return coeff.tolist()
def fit_log(degree):
 y=np.linspace(0,1,20001);f=np.ones_like(y);f[1:]=np.log1p(y[1:])/y[1:]
 base=np.array([1.,np.log(2)-1])
 mat=np.array([y**(i+1)*(1-y) for i in range(degree-1)]).T
 c=np.linalg.lstsq(mat/f[:,None],(f-p.polyval(y,base))/f,rcond=None)[0]
 coeff=p.polyadd(base,p.polymul([0,1,-1],c))
 print('log degree',degree,'max relative error',max(abs(p.polyval(y,coeff)-f)/f))
 return coeff.tolist()

if __name__ == "__main__":
 import argparse,json
 parser=argparse.ArgumentParser()
 parser.add_argument('--output',default='/tmp/softplus_direct_coeffs.json')
 args=parser.parse_args()
 cs=dict(direct=[fit(a,b,7) for a,b in ((0,4),(4,8))],log={d:fit_log(d) for d in (3,4)})
 with open(args.output,'w') as f:json.dump(cs,f,indent=2)
