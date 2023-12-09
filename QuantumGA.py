#!/usr/bin/env python

"""
-------------------------------------------------------------------------------

 Name:        QuantumGA

 Purpose:     Define some multivector classes for studying Quantum Mechanics 

 Author:      Fred Feucht

 Created:     1/25/2023
 
 Copyright:   (c) Fred Feucht 2023        
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

-------------------------------------------------------
"""

from numpy import array, ones, zeros, ma, mean, var, array_equal, tensordot
from numpy import sqrt, e, pi, cos, sinh, cosh, log, inf, sign
from numpy import arccos, arctan, arccosh, arctanh, arctan2
from numpy import dot, matmul, random, set_printoptions
from clifford import Cl, pretty, MultiVector, eps

# initialize the G3 algebra

layout, blades = Cl(3)

# Global variables 

_places = 4             # decimal places
_minnum = 1e-12         # minimum non-zero number
_linelen = 120          # display line length
_interact = True        # interactive mode
_nonstd = False         # use special power series expansion
_ecnt = 25              # exponential iteration count
_separator = '*'        # blade display character  


def errmsg(msg):
    if _interact != True: raise ValueError(msg)
    print("\033[31m\n"+msg); return

# utility function to control the floating point format  

def places(places, minnum=0):
    global _places
    global _minnum
    _places = places
    _minnum = minnum
    if _minnum == 0: 
        _minnum = 1e-12
        if places > 12:
            _minnum = 1/(10**places)
    pretty(places)
    eps(_minnum)
    set_printoptions(precision=places, suppress=True)
"""    set_float = '{0:' + '.' + str(places) + 'f}'
    set_printoptions(formatter={'float_kind': lambda x: set_float.format(x)},
                        precision=places,
                        suppress=True)
"""
# utility function to control the display width  

def linelen(linelen):
    global _linelen
    _linelen = linelen
    set_printoptions(linewidth=_linelen)

# utility function to set the power series iteration count

def ecount(cnt):
    global _ecnt
    _ecnt = cnt

# define single-qubit multivector class

class mvec(MultiVector):
    '''
    def __init__(self, layout, value, dtype=None):
        super().__init__(layout, value=value, dtype=dtype)
    '''
    @staticmethod
    def fromArray(array):
        """ construct a multivector from an 8-element complex array """
        if array.shape != (8,):
            errmsg("Array must be contain 8 elements"); return
        mv = mvec(layout,zeros(8))
        for i in range(8):
            mv.value[i] += (array[i]).real.copy()
            jj = (array[i]).imag.copy()
            if jj != 0.0:
                if i in [2,4,6,7]: jj = -jj
                mv.value[7-i] += jj
        return mv
    def fromComp(comp, mult=1.0):
        """ construct a multivecotr from a complex number """
        aa = array([comp.real,0,0,0,0,0,0,comp.imag])
        return mvec.fromArray(aa)
    def __str__(self):
        """ local multivector formatting routine """
        str = ''
        for i in range(8):
            if abs(self.value[i]) > _minnum:
                if self.value[i] > 0.0:
                    if len(str) != 0: str += ' + '
                    str += '%s' % (round(self.value[i], _places))
                    if i > 0: str += '%s%s' % (_separator, layout.names[i])
                else:
                    if len(str) == 0: str = '-'
                    else: str += ' - '
                    str += '%s' % (-round(self.value[i], _places))
                    if i > 0: str += '%s%s' % (_separator, layout.names[i])
        if str == '' : str = '0.0'
        return str
    def __mul__(self, other):
        """ multiply a multivector by something """
        if type(other) == type(_dyad):
            return dyad(self, _one) * other
        if type(other) == type(_triad):
            return dyad(self, _one) * other
        rr = super().__mul__(other)
        return mvec.fromArray(rr.value)
    def __rmul__(self, other):
        """ multiply something by a multivector """
        rr = super().__rmul__(other)
        return mvec.fromArray(rr.value)
    def __repr__(self):
        """ create a displayable string for this multivector """
        return self.__str__()
    def __rpow__(self, other):
        """ raise a real number to a multivector power """
        if other == e: return self.exp(_nonstd)
        return super().__rpow__(other)
    def __mod__(self, other):
        """ override ampersand for commutator product of bivectors """
        return comm(self(2), other(2)/2)
    def sym(self, other):
        """ calculate the symetric product between multivectors """
        return acomm(self, other)/2
    def asym(self, other):
        """ calculate the  anti-symetric tilde product """
        return comm(self, other)/2
    def tsym(self, other):
        """ calculate the symetric tilde product """
        return tacomm(self, other)/2
    def tasym(self, other):
        """ calculate the  anti-symetric product """
        return tcomm(self, other)/2
    def emag(self):
        """ calculate the Euclidean magnitude """
        return abs(self)
    def cmag(self):
        """ calculate the complex magnitude of a multivector """
        return sqrt((self**2).scalar.comp())*_one
    def bmag(self):
        """ calculate the bar-magnitude of a multivector """
        return sqrt((self.bprod()).scalar.comp())*_one
    def mag(self):
        """ define default magnitude function """
        return self.cmag()
    def norm(self):
        """ calculate the Euclidean norm """
        return self.emag()
    def exp(self, nonstd=False):
        """ approximate the power series sum of a multivector """
        term = _one
        if nonstd == True:
            ss = ~self*self
            pp = ss.prop(ss**2)
            if pp != 0: term = ss/pp
        tot = term
        for i in range(_ecnt):
            term *= self/(i+1)
            tot += term
        return tot
    def prop(self, other):
        """ determine if two multivectors are proportional """
        if type(other) != type(self):
            errmsg("Argument must be a multivector"); return 0
        aa = ma.masked_equal(self.value, 0.0)
        bb = ma.masked_equal(other.value, 0.0)
        if array_equal(aa.mask, bb.mask) == True:
            pp = bb / aa
            if pp.mask.all() == False and abs(var(pp)) <= _minnum:
                return mean(pp)
        return 0
    def inv(self):
        mm = self.bar() * self
        if mm(0) == mm:
            return self.bar() / float(mm(0))
        return super().inv()
    def dot(self, other):
        """ calculate the scalar product between multivectors """
        return dot((~self).value, other.value)*_one
    def iprod(self, other=0):
        """ calculate the Dirac inner product of multivectors """
        if other == 0: other = self
        if type(other) != type(self):
            errmsg("Argument must be a multivector"); return
        return self.neg_high() * other
    def oprod(self, other=0):
        """ calculate the Dirac outer product of multivectors """
        if other == 0: other = self
        if type(other) != type(self):
            errmsg("Argument must be a multivector"); return
        return self * other.neg_high()
    ip = iprod
    op = oprod
    def tprod(self, other=0):
        """ calculate the tilde product of multivectors """
        if other == 0: other = self
        if type(other) != type(self):
            errmsg("Argument must be a multivector"); return
        return self * other.neg_high()
    def bprod(self, other=0):
        """ calculate the bar product of multivectors """
        if other == 0: other = self
        if type(other) != type(self):
            errmsg("Argument must be a multivector"); return
        return self * other.neg_vec()
    def hprod(self, other=0):
        """ calculate the hat product of multivectors """
        if other == 0: other = self
        if type(other) != type(self):
            errmsg("Argument must be a multivector"); return
        return self * other.neg_odd()
    def tdiv(self, other=0):
        """ calculate the tilde ratio of multivectors """
        if other == 0: other = self
        if type(other) != type(self):
            errmsg("Argument must be a multivector"); return
        return self / other.neg_high()
    def bdiv(self, other=0):
        """ calculate the bar ratio of multivectors """
        if other == 0: other = self
        if type(other) != type(self):
            errmsg("Argument must be a multivector"); return
        return self / other.neg_vec()
    def hdiv(self, other=0):
        """ calculate the hat ratio of multivectors """
        if other == 0: other = self
        if type(other) != type(self):
            errmsg("Argument must be a multivector"); return
        return self / other.neg_odd()
    def expect(self, oper):
        """ calculate the expectation value for an operator """
        return (~self * oper * self)(0)
    def tinspect(self, oper):
        """ calculate the tilde inspection value for an operator """
        return self * oper * self.tilde()
    def binspect(self, oper):
        """ calculate the bar inspection value for an operator """
        return self * oper * self.bar()
    def hinspect(self, oper):
        """ calculate the hat inspection value for an operator """
        return self * oper * self.hat()
    def neg_high(self):
        """  negate high blades (reversion) """
        return mvec.fromArray(self.value * _neg_high)
    def neg_odd(self):
        """  negate odd blades """
        return mvec.fromArray(self.value * _neg_odd)
    def neg_vec(self):
        """  negate vector and bivector blades """
        return mvec.fromArray(self.value * _neg_vec)
    def trans(self):
        """  transpose in the specular basis """
        return mvec.fromArray(self.value * _neg_trans)
    def star(self):
        """  complex conjugation """
        return mvec.fromArray(self.value * _neg_trans * _neg_high)
    def flip(self):
        """  reverse bivectors only """
        return mvec.fromArray(self.value * _neg_bi)
    def tilde(self):
        return self.neg_high()
    def bar(self):
        """ Clifford conjugate """
        return self.neg_vec()
    def hat(self):
        """ grade involution """
        return self.neg_odd()
#    def star(self):
#        return (self-2*self(3))
    def proj(self, vec):
        """ projection of a multivector in a given direction """
        return 2*self.sym(vec)/vec
    def rej(self, vec):
        """ rejection of a multivector in a given direction """
        return 2*self.asym(vec)/vec
    def vproj(self, vec):
        """ projection of a vector in a given direction """
        return (self|vec)/vec
    def vrej(self, vec):
        """ rejection of a vector in a given direction """
        return (self^vec)/vec
    def para(self, vec):
        """ find a multivector subspace parallel to a given vector """
        return (self + vec*self*vec)/2      
    def perp(self, vec):
        """ find a multivector subspace perpendicular to a given vector """
        return (self - vec*self*vec)/2      
    def tpara(self, vec):
        """ find a multivector subspace parallel to a given realized vector """
        return (self + vec*self*vec.tilde())/2      
    def tperp(self, vec):
        """ find a multivector subspace perpendicula to a given realized vector """
        return (self - vec*self*vec.tilde())/2      
    def bpara(self, vec):
        """ find a multivector subspace parallel to a given invariant """
        return (self + vec*self*vec.bar())/2      
    def bperp(self, vec):
        """ find a multivector subspace perpendicula to a given invariant """
        return (self - vec*self*vec.bar())/2      
    def dual(self):
        """  dualize vector and bivector blades """
        return mvec.fromArray(matmul(_da, self.value))
    def flop(self):
        """  dualize e2 and e13 blades """
        return mvec.fromArray(matmul(_sa, self.value))
    def unflop(self):
        """  undualize e2 and e13 blades """
        return mvec.fromArray(matmul(_ua, self.value))
    def deComp(self):
        """ decompose a complex exponential """
        if self(1) + self(2) != 0:
            errmsg("Object must be complex"); return
        a = self.value[0]
        b = self.value[7]
        if a == 0: return (-self(3)*_e123, pi/2*_one, _e123)
        c = arctan2(b, a)
        # if sign(a) == -1: c += sign(b)*pi
        return (c*_e123)
    def deQuat(self):
        """ decompose a quaternion """
        if self(1) + self(3) != 0:
            errmsg("Object must be a quaternion"); return
        a = self.value[0]
        b = abs(self(2))
        if a == 0: return (abs(self(2)), pi/2*_one, self(2).enormal())
        c = arctan(b/abs(a))
        d = _e12
        if b != 0.0: d = self(2).enormal()
        e = self(0)/cos(c)
        return (e, c, d)
    def dePara(self):
        """ decompose a time-like relativistic paravector """
        if self(2) + self(3) != 0:
            errmsg("Object must be a paravector"); return
        a = self.value[0]
        b = abs(self(1))
        w = b/a
        if a == 0 or abs(w) > 1.0:
            errmsg("Object must be hyperbolic"); return
        if w == 1.0:
            return(a, inf, self(1).enormal())
        c = arctanh(w)
        d = _e1
        if b != 0.0: d = self(1).enormal()
        e = self(0)/cosh(c)
        return (float(e), float(c), d)
    def deSara(self):
        """ decompose a space-like relativistic paravector """
        if self(2) + self(3) != 0:
            errmsg("Object must be a paravector"); return
        a = self.value[0]
        b = abs(self(1))
        if a == 0 or (a/b) >= 1.0:
            errmsg("Object must be anti-hyperbolic"); return
        c = arctanh(a/b)
        d = _e1
        if b != 0.0: d = self(1).enormal()
        e = self(0)/sinh(c)
        return (float(e), float(c), d)
    def deTrans(self):
        """ decompose Lorentz tranformation """
        ww = _zero
        if abs(self(2)) > _minnum:
            ww = self(2).enormal() * arctan(abs(self(2)/self(0)))
        zz = self*e**(-ww)
        if abs(zz(3)) > _minnum:
            "print(zz.value)"
            errmsg("Object is not Lorentzian"); return
        xx = (zz*zz.bar()).zero
        if xx <= _minnum:
            "print(xx)"
            errmsg("Object is not Lorentzian"); return
        rr = sqrt(xx)
        vv = _zero
        if abs(zz(1)) > _minnum:
            vv = zz(1).enormal() * arctanh(abs(zz(1)/zz(0)))
        return (rr, vv, ww)
    def deState(self):
        """ decompose a relativistic paravector """
        if self(2) + self(3) != 0:
            errmsg("Object must be a paravector"); return
        a = abs(self(0))
        b = abs(self(1))
        if a == 0 or (b/a) >= 1.0:
            errmsg("Object must be hyperbolic"); return
        c = _one*arctanh(b/a)
        d = _e1
        if b != 0.0: d = self(1).enormal()
        e = _one*self(0)/cosh(abs(c))
        return (e, c, d)
    def deScrew(self):
        """ decompose a complex boost """
        pv0 = self(0)
        pv1 = self(1)
        pv2 = self(2)
        pv3 = self(3)
        if abs(pv0[0]) < _minnum:
            errmsg("Invalid Object"); return
        if abs(pv1)+abs(pv2)+abs(pv3) < _minnum:
            """ screw is a scalar """
            return (pv0, _zero, _zero)
        if abs(pv1)+abs(pv2) < _minnum:
            """ screw is a complex scalar """
            w = (pv0+pv3)*(pv0-pv3)
            m = sqrt(w[0])*_one
            w = (-_e123*pv3)/pv0
            j = arctan(float(w))
            return (m, j*_e123, _zero)
        w = (pv0+pv2)*(pv0-pv2)-(pv1+pv3)*(pv1-pv3)
        if abs(w(1) + w(2) + w(3)) > _minnum:
            "print(w)"
            errmsg("Object is not a screw")
            return
        m = sqrt(abs(w[0]))*_one
        w1 = pv1
        w2 = -_e123*pv2
        w = w1
        if abs(w) < _minnum: 
            w = w2
        else:
            if abs(pv2) > _minnum:
                if float(abs(abs(w1.enormal()|w2.enormal()) - 1.0)) > _minnum:
                    errmsg("Object is not a linear screw"); return
        v = w.enormal()
        w = pv1/pv0
        if abs(w) > 1.0:    
            errmsg("Object is not hyperbolic"); return
        l = arctanh(float((w/v)(0)))
        w = (-_e123*pv2)/pv0
        j = arctan(float((w/v)(0)))
        z = l*_one + j*_e123
        if abs(z) < _minnum: v = _zero 
        return (m, z, v)
    def deJoint(self):
        """ decompose a joint """
        if abs(self.zero + self.pseudo) > _minnum:
            errmsg("Object is not a joint"); return
        l1 = 0.0
        l2 = 0.0
        v1 = None
        v2 = None
        if abs(self(1)) > _minnum:
            v1 = self(1).enormal()
            l1 = abs(self(1))
        if abs(self(2)) > _minnum:
            v2 = self(2).enormal()/_e123
            l2 = abs(self(2))
        return (l1, v1, l2, v2)        
    def deCvec(self):
        """ decompose a complex vector """
        if abs(self.sc) > _minnum:
            errmsg("Object is not a complex vector"); return
        m = self.mag()
        f = self.normal()
        a = f(1).normal()
        c = f.getPlane()
        l = f.getRapid()
        return (m, a, c, l)
    def getPlane(self):
        """ return the plane of a complex vector """
        if abs(self(1)) < _minnum:
            errmsg("no vector component"); return
        if abs(self(2)) < _minnum:
            errmsg("no bivector component"); return
        return (self(1)*self(2)).vnormal()
    def getRapid(self):
        """ extract the internal rapidity from a complex vector """
        if abs(self(1)) < _minnum:
            errmsg("no vector component"); return
        if abs(self(2)) < _minnum:
            errmsg("no bivector component"); return
        return arctanh((((self(2)/self(1))).mag()).zero)*_one
    def getWedge(self):
        """ return the pseudo-wedge of a complex vector """
        if abs(self(1)) < _minnum:
            errmsg("no vector component"); return
        if abs(self(2)) < _minnum:
            errmsg("no bivector component"); return
        return self(1)^self(2)/_e123
    def getMo(self):
        """ extract the momentum 4-vector from a complex vector """
        if abs(self(1)) < _minnum:
            errmsg("no vector component"); return
        return self/self(1).normal()
    def getFlect(self, other):
        """ find a reflector from one vector to another """
        if abs(self.scalar) > _minnum:
            errmsg("object is not a vector"); return
        if abs(other.scalar) > _minnum:
            errmsg("target is not a vector"); return
        return (self-other).normal()    
    def getRotor(self, psi, min=_minnum):
        """ find the complex rotor to another (complex) vector """
        w = (psi/self).log()
        return e**(w/2)
    def getBoost(self, psi, min=_minnum):
        """ find the complex boost to another vector exponential """
        if abs(self.bprod()-1.0) > min:
                errmsg("Base object is not invariant"); return
        if abs(psi.bprod()-1.0) > min:
                errmsg("Boost operator is not invariant"); return
        w = (psi+self.bar()).vs()
        v = w.normal()
        r = arctanh(w.mag().comp())*_one
        return e**(v*r)
    def log(self):
        """ find the natural logarithm of a multivector """
        if abs(self.scalar) < _minnum:
            m = self.mag()
            if abs(m) < _minnum: 
                errmsg("Cannot take the log if a null vector"); return
            n = self.normal()
            return _e123*pi/2 - _e123*pi*n/2 + log(m)
        a = self.bprod().tprod()
        if self.zero > _minnum:
            b = a/self.zero**4
            if abs(a) < _minnum or abs(b.zero) < _minnum:
                print("Multivector is light-like")
                b = log(2*self.zero+1)/2
                return b*(1+self.vector.normal())
        s = self.bprod().scalar.slog()/2
        v = (e**-s*self).slog()
        if abs(self.zero) > _minnum:
            if sign((e**(s+v)).zero) != sign(self.zero):
                return s + v + pi*_e123
        return s + v
    def slog(self):
        """ find the natural logarithm for a vector or scalar subspace """
        if abs(self.vector.mag()) > _minnum:
            t = arccosh(self.scalar.comp())
            j = self.vector.normal()
            if self.zero < 0.0 and abs(self(1)) > _minnum: 
                j = j.bar() 
        else:
            t = log(self.scalar.comp())
            j = _one
        return j*t
    def sfact(self, low=True):
        """ factor multivector in the e1-up basis """
        if((abs(self[0]-self[1]) > _minnum) or 
           (abs(self[2]+self[4]) > _minnum) or
           (abs(self[3]+self[5]) > _minnum) or
           (abs(self[7]-self[6]) > _minnum)):
                errmsg("Object cannot be factored in the e1-up basis")
                return
        if(low == False): 
            return 2*(self[0] + self[2]*_e2 + self[3]*_e3 + self[7]*_e123)
        return 2*(self[1]*_e1 + self[4]*_e12 + self[5]*_e13 + self[6]*_e23)
    def tfact(self, low=True):
        """ factor multivector in the e2-up basis """
        if((abs(self[0]-self[2]) > _minnum) or 
           (abs(self[1]-self[4]) > _minnum) or
           (abs(self[3]+self[6]) > _minnum) or
           (abs(self[7]+self[5]) > _minnum)):
                errmsg("Object cannot be factored in the e2-up basis")
                return
        if(low == False): 
            return 2*(self[0] + self[1]*_e1 + self[3]*_e3 + self[7]*_e123)
        return 2*(self[2]*_e2 + self[4]*_e12 + self[6]*_e23 + self[5]*_e13)
    def ufact(self, low='', quiet=False):
        """ factor multivector in the e3-up basis """
        if((abs(self[0]-self[3]) > _minnum) or 
           (abs(self[1]-self[5]) > _minnum) or
           (abs(self[2]-self[6]) > _minnum) or
           (abs(self[4]-self[7]) > _minnum)):
                if quiet == False:
                    errmsg("Object cannot be factored in the e3-up basis")
                return 
        if(low == False): 
            return 2*(self[0]*_one + self[1]*_e1 + self[2]*_e2 + self[4]*_e12)
        if(low == True):
            return 2*(self[0]*_one + self[1]*_e1 + self[6]*_e23 + self[7]*_e123)
            # return 2*(self[3]*_e3 + self[5]*_e13 + self[6]*_e23 + self[7]*_e123)
        return 2*(self[0]*_one + self[5]*_e13 + self[6]*_e23 + self[4]*_e12)
    def nfact(self, low=''):
        """ factor multivector in the e3-up basis """
        if((abs(self[0]+self[3]) > _minnum) or 
           (abs(self[1]+self[5]) > _minnum) or
           (abs(self[2]+self[6]) > _minnum) or
           (abs(self[4]+self[7]) > _minnum)):
                errmsg("Object cannot be factored in the e3-not_up basis")
                return
        if(low == False): 
            return 2*(self[0] + self[1]*_e1 + self[2]*_e2 + self[4]*_e12)
        if(low == True):
            return 2*(self[3] + self[5]*_e13 + self[6]*_e23 + self[7]*_e123)
        return 2*(self[0] + self[1]*_e1 + self[6]*_e23 + self[7]*_e123)
    def cfact(self):
        """ extract complex 4-vector coordinates of a multivector """
        a = self.value[0] + self.value[7]*_e123        
        b = self.value[1] + self.value[6]*_e123        
        c = self.value[2] - self.value[5]*_e123        
        d = self.value[3] + self.value[4]*_e123
        return (a, b, c, d)
    def updn(self, evec=None, fvec=None):
        """ find the spectral coordinates of a multivector  """
        if evec == None: evec = _e3
        if fvec == None: 
            fvec = _e1
            if abs(evec(1)) > _minnum and abs(evec(2)) > _minnum:
                fvec = evec.normal().getPlane()
        ket = (1+evec.normal())
        vec = fvec.normal()
        a = (self*ket).sc
        b = (self*vec*ket).sc
        c = (self*ket*vec).sc
        d = (self*vec*ket*vec).sc
        return (a, b, c, d)
    def toArray(self, basis):
        """ convert multivector to 2x2 complex array """
        a = zeros((2,2),dtype=complex)
        t = basis.T
        for i in range(2):
            for j in range(2):
                a[i,j] = (2*self*t[i,j]).scalar.comp()
        return a
    def scde(self, evec=None, fvec=None):
        """ find the projected components of a ket vector """
        if evec == None: evec = _e3
        if fvec == None: 
            fvec = _e1
            if abs(evec(1)) > _minnum and abs(evec(2)) > _minnum:
                fvec = evec.normal().getPlane()
        uvec = evec.normal()
        cvec = fvec.normal()
        dvec = _e123*cvec*uvec
        a = (self).sc
        b = (self*cvec).sc
        c = (self*dvec).sc
        d = (self*uvec).sc
        return (a, b, c, d)        
    def T(self, evec=None, fvec=None):
        """ construct the transpose of a ket vector  """
        if evec == None: evec = _e3
        if fvec == None: 
            fvec = _e1
            if abs(evec(1)) > _minnum and abs(evec(2)) > _minnum:
                fvec = evec.normal().getPlane()
        dvec = _e123*fvec*evec
        return (dvec*self.bar()*dvec)
    def tpose(self, evec=None, fvec=None):
        """ construct the transpose of a ket vector """
        if evec == None: evec = _e3
        if fvec == None: 
            fvec = _e1
            if abs(evec(1)) > _minnum and abs(evec(2)) > _minnum:
                fvec = evec.normal().getPlane()
        uvec = evec.normal()
        cvec = fvec.normal()
        dvec = _e123*cvec*uvec
        a = (self).sc
        b = (self*uvec).sc*uvec
        c = (self*cvec).sc*cvec
        d = (self*dvec).sc*dvec        
        return a+b+c-d
    def bro(self, evec=None, fvec=None):
        """ construct the conjugate transpose of a ket vector """
        if evec == None: evec = _e3
        if fvec == None: 
            fvec = _e1
            if abs(evec(1)) > _minnum and abs(evec(2)) > _minnum:
                fvec = evec.normal().getPlane()
        ket = (1+evec.normal())
        vec = fvec.normal()
        a = ~(self*ket).sc*ket/2
        b = ~(self*vec*ket).sc*vec*ket/2
        c = ~(self*ket*vec).sc*ket*vec/2
        d = ~(self*vec*ket*vec).sc*vec*ket*vec/2        
        return a+b+c+d
    def bra(self, evec=None, fvec=None):
        """ construct the conjugate transpose of a ket vector """
        if evec == None: evec = _e3
        if fvec == None: 
            fvec = _e1
            if abs(evec(1)) > _minnum and abs(evec(2)) > _minnum:
                fvec = evec.normal().getPlane()
        uvec = evec.normal()
        cvec = fvec.normal()
        dvec = _e123*cvec*uvec
        a = ~(self).sc
        b = ~(self*uvec).sc*uvec
        c = ~(self*cvec).sc*cvec
        d = ~(self*dvec).sc*dvec        
        return a+b+c+d
    def toMVarray(self):
        """ convert multivector to 2x2 multivector array """
        Left = ((_one+_e3)/2, (_one-_e3)/2)
        Right = ((_one+_e3)/2,(_one-_e3)/2)
        Array = zeros((2,2),dtype=mvec)
        for i in range(2):
            for j in range(2):
                Array[i,j] = ((Left[i]*self*Right[j]))
        return Array
    def splitp(self, vec):
        """ separate out parallel and perpendicular components """
        A = vec*self.lo
        B = vec*self.hi
        a = vec*A.lo
        b = vec*A.hi
        c = vec*B.lo
        d = vec*B.hi
        return (a+d, b+c)
    def splitl(self):
        """ split a transform into real and imaginary components """
        l = self.log()
        if l.scalar.emag() > _minnum: 
            errmsg("Object is not a Lorentz transform")
            return 
        m = l.mag()
        f = l.normal()
        return (e**(f*m(0)), e**(f*m(3)))        
    def half(self):
        """ split a screw in two """
        ss = self.deScrew()
        return (sqrt(abs(ss[0]))*e**(ss[1]*ss[2]/2))
    def vs(self):
        """ divide the vector blades by the scalar blades """
        return self.vector/self.scalar
    def hilo(self):
        """ divide the high grade blades by the low grade blades """
        return self.hi/self.lo
    def deprop(self):
        """ convert a relativistic paravector to classical values """
        return (self/self.scalar)
    def comp(self):
        """ translate scalar multivector to complex number """
        if self.isComp() == False:
            errmsg("Object is not complex scalar")
            return
        return self[0] + self[7]*1j
    def isComp(self):
        """ multivector is complex if it has no vector components """
        if abs(self.vector) > _minnum: return False
        return True
    def sqrt(self):
        """ find the square root of a scalar multivector """
        if self.isComp() == True:
            return sqrt(self.comp())*_one
        errmsg("Object is not scalar")
        return
    def cos(self):
        """ find the cosine of a multivector """
        return (e**(_e123*self) + e**-(_e123*self))/2
    def sin(self):
        """ find the sine of a multivector """
        return (e**(_e123*self) - e**-(_e123*self))/(2*_e123)
    def tan(self):
        """ find the tangent of a multivector """
        return self.sin()/self.cos()
    def cosh(self):
        """ find the hyperbolic cosine of a multivector """
        return (e**self + e**-self)/2
    def sinh(self):
        """ find the hyperbolic sine of a multivector """
        return (e**self - e**-self)/2
    def tanh(self):
        """ find the hyperbolic tangent of a multivector """
        return (e**self - e**-self)/(e**self + e**-self)
    def normal(self):
        """ root-square normalize a multivector """
        if abs(self.mag()) < _minnum:
            errmsg("Object is a null vector")
            return
        return self/self.mag()
    def enormal(self):
        """ tilde product normalize a multivector """
        return self/abs(self)
    def snormal(self):
        """ scalar normalize a multivector """
        if abs(self.sc) < _minnum:
            errmsg("Object has no scalar components"); return
        return self/self.sc
    def vnormal(self):
        """ return the normalize vector components """
        return self.vector.normal()
    @property
    def dt(self):
        """ return scalar magnitude """
        return self.oprod()(0)
    @property
    def dx(self):
        """ return e1 magnitude """
        return self.oprod().x*_one
    @property
    def dy(self):
        """ return e2 magnitude """
        return self.oprod().y*_one
    @property
    def dz(self):
        """ return e3 magnitude """
        return self.oprod().z*_one
    @property
    def low(self):
        """ return grade 0 and grade 1 blades """
        return self(0) + self(1)
    lo = low
    re = low
    @property
    def high(self):
        """ return grade 2 and grade 3 blades """
        return self(2) + self(3)
    hi = high
    im = high
    @property
    def vector(self):
        """ return grade 1 and grade 2 blades """
        return self(1) + self(2)
    ve = vector
    vec = vector
    @property
    def scalar(self):
        """ return grade 0 and grade 3 blades """
        return self(0) + self(3)
    sc = scalar
    @property
    def zero(self):
        """ return grade 0 blade as a float """
        return self.value[0]
    @property
    def pseudo(self):
        """ return grade 3 blade as a float """
        return self.value[7]
    @property
    def e1(self):
        """ return e1 blade """
        return self.value[1]*_e1
    @property
    def e2(self):
        """ return ey blade """
        return self.value[2]*_e2
    @property
    def e3(self):
        """ return e3 blade """
        return self.value[3]*_e3
    @property
    def e12(self):
        """ return e12 blade only """
        return self[4]*_e12
    @property
    def e13(self):
        """ return e13 blade only """
        return self[5]*_e13
    @property
    def e23(self):
        """ return e23 blade only """
        return self[6]*_e23
    @property
    def e123(self):
        """ return e123 blade only """
        return self[7]*_e123
    @property
    def x(self):
        """ return e1 blade """
        return self.value[1]
    @property
    def y(self):
        """ return ey blade """
        return self.value[2]
    @property
    def z(self):
        """ return e3 blade """
        return self.value[3]
    @property
    def xy(self):
        """ return e12 blade only """
        return self[4]
    @property
    def xz(self):
        """ return e13 blade only """
        return self[5]
    @property
    def yz(self):
        """ return e23 blade only """
        return self[6]
    @property
    def xyz(self):
        """ return e123 blade only """
        return self[7]
    @property
    def points(self):
        """ return 3D vector as a list of floats """
        if abs(self(0)) + abs(self(2)) + abs(self(3)) > _minnum:
            errmsg("Non-vector blades are present")
            return
        x = self.value[1]
        y = self.value[2]
        z = self.value[3]
        if abs(x) < _minnum: x = 0.0 
        if abs(y) < _minnum: y = 0.0 
        if abs(z) < _minnum: z = 0.0 
        return x, y, z
    @property
    def pts(self):
        """ return 3D vector as a list of floats """
        if abs(self(0)) + abs(self(2)) + abs(self(3)) > _minnum:
            errmsg("Non-vector blades are present")
            return
        x = self.value[1]
        y = self.value[2]
        z = self.value[3]
        if abs(x) < _minnum: x = 0.0 
        if abs(y) < _minnum: y = 0.0 
        if abs(z) < _minnum: z = 0.0 
        return "{0}*e1 + {1}*e2 + {2}*e3".format(x, y, z)

# define a double-multivector (separable) class

class pair(object):
    """ separable pair of multivectors """
    def __init__(self, mv1, mv2):
        if type(mv1) == float or type(mv1) == int: mv1 *= _one
        if type(mv2) == float or type(mv2) == int: mv2 *= _one
        self.left = mv1
        self.right = mv2
    def __repr__(self):
        """ create a displayable string for a pair """
        return "[(%s) o (%s)]" % (self.left, self.right)
    def __mul__(self, other):
        """ multiply pairwise """
        if type(other) != type(self):
            errmsg("Argument must be a pair")
        return pair(self.left * other.left, self.right * other.right)
    def __truediv__(self, other):
        """ divide pairwise """
        if type(other) != type(self):
            errmsg("Argument must be a pair")
        return pair(self.left / other.left, self.right / other.right)
    def __div__(self, other):
        """ divide pairwise """
        return self.__truediv__(other)
    def __eq__(self, other):
        """ pairwise compare """
        if type(other) != type(self):
            errmsg("Argument must be a pair")
        if (self - other) == dyad(_zero, _zero): return True
        return False
#    def __call__(self, value=1.0):
#        """ return scaled copy (multiply left and divide high) """
#        return pair(self.left*float(value), self.right/float(value))
    def zero(self):
        """ zero out a pair """
        self.left = self.right = _zero
    def scale(self, value):
        """ multiply left, divide right scaling """
        return pair(self.left*float(value), self.right/float(value))
#        return self(value)

# define double-multivector list class 
    
class pairList(list):
    """ list of left/right pairs """
    def __init__(self, pair):
        self.pair = pair
    def __repr__(self):
        """ create a displayable string for a pairlist """
        return ' +\n'.join('%s' % (i) for i in self.pair)

# define a general purpose two-qubit multivector class

class dyad(object):
    """ dyadic multivector """
    def __init__(self, mv1, mv2):
        if type(mv1) == float or type(mv1) == int: mv1 *= _one
        if type(mv2) == float or type(mv2) == int: mv2 *= _one
        self.value = mv1.value.reshape(8,1) * mv2.value
    @staticmethod
    def frompair(pair):
        """ construct a dyad object from a pair object """
        return dyad(pair.left, pair.right)
    @staticmethod
    def fromArray(array):
        """ construct a dyad object from an 8 x 8 array """
        if array.shape != (8,8):
            errmsg("Array must be contain 8x8 elements"); return
        d = dyad(_zero, _zero)
        for i in range(8):
            for j in range(8):
                d.value[i,j] += (array[i,j]).real.copy()
                jj = (array[i,j]).imag.copy()
                if jj != 0.0:
                    if i in [2,4,6,7]: jj = -jj
                    d.value[7-i,j] += jj
        return d
    def __repr__(self):
        """ create a displayable string for a dyad """
        list = self.tolist()
        if len(list) == 0: return '[(0.0) o (0.0)]'
        return ' +\n'.join('%s' % (i) for i in list)
    def __add__(self, other):
        """ add two dyads """
        if type(other) != type(self):
            errmsg("Argument must be a dyad")
        return dyad.fromArray(self.value + other.value)
    def __sub__(self, other):
        """ subtract two dyads """
        if type(other) != type(self):
            errmsg("Argument must be a dyad")
        return dyad.fromArray(self.value - other.value)
    def __mul__(self, other):
        """ multiply a dyad with something """
        if type(other) == type(_one): other = dyad(other, _one)
        if type(other) == type(self):
            l1 = self.tolist()
            l2 = other.tolist()
            newbit = dyad.fromArray(zeros((8,8)))
            for i in l1:
                for j in l2:
                    newbit.value += dyad.frompair(i * j).value
            return newbit 
        return dyad.fromArray(self.value * other)
    def __rmul__(self, other):
        """ multiply something with a dyad """
        return dyad.fromArray(self.value * other)
    def __truediv__(self, other):
        if type(other) == type(self):
            l1 = other.tolist()
            l2 = self.tolist()
            newbit = dyad.fromArray(zeros((8,8)))
            for i in l1:
                for j in l2:
                    newbit.value += dyad.frompair(i / j).value
        return dyad.fromArray(self.value / other)
    def __div__(self, other):
        return self.__truediv__(other)
    def __neg__(self):
        """ calculate the negative of a dyad """
        return dyad.fromArray(self.value * -1.0) 
    def __eq__(self, other):
        """ compare two dyads for equality """
        if type(other) != type(self): return False
        if(self.value.all() == other.value.all()): return True
        return False
    def __invert__(self):
        """ perform reversion in the Dirac algebra """
        return dyad.fromArray(self.value * _high_both)
    def __call__(self, grade):
        """ return specific grade dyadic blade """
        if grade == 0: return self.value[0][0]*dyad(_one,_one)
        if grade == 1: return self.value[1][1]*dyad(_e1,_e1)
        if grade == 2: return self.value[2][2]*dyad(_e2,_e2)
        if grade == 3: return self.value[3][3]*dyad(_e3,_e3)
        if grade == 7: return self.value[7][7]*dyad(_e123,_e123)
        errmsg("Unsuppored grade")
        return 0 
    def __rpow__(self, other):
        """ exponentiate a dyad """
        if other == e: return self.exp(_nonstd)
        errmsg("Only natural exponentials supported")
        return 0
    def __abs__(self):
        """ calculate dyad magnitude """
        return sqrt((self*~self).value[0][0])
    def __pow__(self, other):
        """ raise a dyad to an integer power """
        if not isinstance(other, (int)):
            errmsg("Exponent must be an integer")
        if other < 1:
            errmsg("Exponent must be a positive integer")
        dd = self
        for i in range(1, other):
            dd *= self
        return dd
    def exp(self, nonstd=False):
        """ approximate the power series sum of a dyad """
        term = dyad(_one,_one)
        if nonstd == True:
            ss = ~self*self
            pp = ss.prop(ss**2)
            if pp != 0: term = ss/pp
        tot = term
        for i in range(_ecnt):
            term *= self/(i+1.0)
            tot += term
        return tot
    def prop(self, other):
        """ determine if two dyads are proportional """
        if type(other) != type(self):
            errmsg("Argument must be a dyad"); return 0
        aa = ma.masked_equal(self.value, 0.0)
        bb = ma.masked_equal(other.value, 0.0)
        if array_equal(aa.mask, bb.mask):
            pp = bb / aa
            if pp.mask.all() == False and abs(var(pp)) <= _minnum:
                return mean(pp)
        return 0
    def tolist(self, left=True):
        """ generate a list of dyadic blades """
        list = []
        for i in range(8):
            for j in range(8):
                if left:
                    if abs(self.value[j][i]) >= _minnum:
                        list += [pair(self.value[j][i]*_blist[j], _blist[i])]
                else:
                    if abs(self.value[i][j]) >= _minnum:
                        list += [pair(_blist[i], self.value[i][j]*_blist[j])]
        return list
    def ii(self):
        """ convert double imaginary dyadic blades to real """
        d = dyad.fromArray(self.value)
        for i in range(4):
            for j in range(4):
                n = d.value[7-i][7-j]
                if n != 0:
                    d.value[7-i][7-j] = 0
                    if i == 2: n = -n
                    if j == 2: n = -n
                    d.value[i][j] -= n
        return d
    def dense(self, scale=1.0):
        """ collect dyadic blades into pairs """
        list = self.tolist()
        if len(list) == 0: return pairList([pair(_zero, _zero)])
        for i in range(len(list)):
            for j in range(i+1, len(list)):
                if list[i].right == list[j].right:
                    list[i].left += list[j].left
                    list[j].zero()
        for i in range(len(list)):
            for j in range(i+1, len(list)):
                cc = list[i].left.prop(list[j].left)
                if cc != 0:
                    list[i].right += cc*list[j].right
                    list[j].zero()
        if scale == 'r':    
            return pairList([i.scale(abs(i.right)) for i in list if abs(i.left) > _minnum])
        if scale == 'l':
            return pairList([i.scale(1.0/abs(i.left)) for i in list if abs(i.left) > _minnum])
        if scale == 'sr':    
            return pairList([i.scale(sqrt(2)*abs(i.right)) for i in list if abs(i.left) > _minnum])
        if scale == 'sl':
            return pairList([i.scale(0.5*sqrt(2)/abs(i.left)) for i in list if abs(i.left) > _minnum])
        return pairList([i.scale(scale) for i in list if abs(i.left) > _minnum])
    def fense(self, scale=1.0):
        """ collect dyadic blades into pairs """
        list = self.tolist(False)
        for i in range(len(list)):
            for j in range(i+1, len(list)):
                if list[i].left == list[j].left:
                    list[i].right += list[j].right
                    list[j].zero()
        for i in range(len(list)):
            for j in range(i+1, len(list)):
                cc = list[i].right.prop(list[j].right)
                if cc != 0:
                    list[i].left += cc*list[j].left
                    list[j].zero()
        if scale == 'r':    
            return pairList([i.scale(abs(i.right)) for i in list if abs(i.left) > _minnum])
        if scale == 'l':
            return pairList([i.scale(1.0/abs(i.left)) for i in list if abs(i.left) > _minnum])
        if scale == 'sr':    
            return pairList([i.scale(sqrt(2)*abs(i.right)) for i in list if abs(i.left) > _minnum])
        if scale == 'sl':
            return pairList([i.scale(0.5*sqrt(2)/abs(i.left)) for i in list if abs(i.left) > _minnum])
            return pairList([i.scale(1.0/abs(i.left)) for i in list if abs(i.left) > _minnum])
        return pairList([i.scale(scale) for i in list if abs(i.left) > _minnum])
    def aright(self, scale=1.0):
        """ collect dyadic blades into pairs """
        list = self.tolist()
        if len(list) == 0: return pairList([pair(_zero, _zero)])
        for i in range(len(list)):
            for j in range(i+1, len(list)):
                if list[i].right == list[j].right:
                    list[i].left += list[j].left
                    list[j].zero()
        return pairList([i.scale(scale) for i in list if abs(i.left) > _minnum])
    def aleft(self, scale=1.0):
        """ collect dyadic blades into pairs """
        list = self.tolist(False)
        for i in range(len(list)):
            for j in range(i+1, len(list)):
                if list[i].left == list[j].left:
                    list[i].right += list[j].right
                    list[j].zero()
        return pairList([i.scale(scale) for i in list if abs(i.left) > _minnum])
    def rev_right(self):
        """ reverse right-side of dyad """
        return dyad.fromArray(self.value * _high_right)
    def rev_left(self):
        """ reverse left-side of dyad """
        return dyad.fromArray(self.value * _high_left)
    def rev_both(self):
        """  reverse both sides of dyad """
        return dyad.fromArray(self.value * _high_both)
    def gin_right(self):
        """ grade involute right-side dyad """
        return dyad.fromArray(self.value * _odd_right)
    def gin_left(self):
        """ grade involute left-side dyad """
        return dyad.fromArray(self.value * _odd_left)
    def gin_both(self):
        """ grade involute both sides of a dyad """
        return dyad.fromArray(self.value * _odd_both)
    def con_right(self):
        """ clifford conjugate right-side dyad """
        return dyad.fromArray(self.value * _vec_right)
    def con_left(self):
        """ clifford conjugate left-side dyad """
        return dyad.fromArray(self.value * _vec_left)
    def con_both(self):
        """ clifford conjugate both sides of a dyad """
        return dyad.fromArray(self.value * _vec_both)
    def star_right(self):
        """ complex conjugate right side of a dyad """
        return dyad.fromArray(self.value * _trans_right * _high_right)
    def star_left(self):
        """ complex conjugate left side of a dyad """
        return dyad.fromArray(self.value * _trans_left * _high_left)
    def tilde(self):
        """ reverse both sides of dyads """
        return dyad.fromArray(self.value * _high_both)
    def trans(self):
        """ transpose both sides of a dyad """
        return dyad.fromArray(self.value * _trans_both)
    def star(self):
        """ complex conjugate both sides of a dyad """
        return dyad.fromArray(self.value * _trans_both * _high_both)
    def rev_dirac(self):
        """ reverse dirac dyad """
        return dyad.fromArray(self.value * _dirac_high)
    def gin(self):
        """ reverse left and grade involute right of dyad """
        return dyad.fromArray(self.value * _high_left * _odd_right)
    def con(self):
        """ reverse left and clifford conjugate right of dyad """
        return dyad.fromArray(self.value * _high_left * _vec_right)
    def enormal(self):
        """ convert to unit dyad """
        return self / abs(self)
    def norm2(self):
        """ convert to normal dyad """
        return self / self.value[0,0]
    def dot(self, other):
        """ calculate the scalar product of dyads """
        return tensordot((~self).value, other.value).item(0)*dyad(_one, _one)
    def iprod(self, other=0):
        """ calculate inner product of spin dyad(s) """
        if other == 0: other = self
        if type(other) != type(self):
            errmsg("Argument must be a dyad"); return
        return self.tilde() * other
    def oprod(self, other=0):
        """ calculate outer product of spin dyad(s) """
        if other == 0: other = self
        if type(other) != type(self):
            errmsg("Argument must be a dyad")
        return self * other.tilde()
    op = oprod
    ip = iprod
    def rprod(self, other=0):
        """ calculate outer product of relativistic dyad(s) """
        return self.oprod_con(other)
    def gprod(self, other=0):
        """ calculate outer product of relativistic dyad(s) """
        if other == 0: other = self
        if type(other) != type(self):
            errmsg("Argument must be a dyad")
        return self * other.gin_left()
    def cprod(self, other=0):
        """ calculate outer product of relativistic dyad(s) """
        if other == 0: other = self
        if type(other) != type(self):
            errmsg("Argument must be a dyad")
        return self * other.con()
    def iprod_rel(self, other=0):
        """ calculate inner product of relativistic dyad(s) """
        if other == 0: other = self
        if type(other) != type(self):
            errmsg("Argument must be a dyad"); return
        return self.adj() * other
    def oprod_rel(self, other=0):
        """ calculate outer product of relativistic dyad(s) """
        if other == 0: other = self
        if type(other) != type(self):
            errmsg("Argument must be a dyad"); return
        return self * other.adj()
    def iprod_gin(self, other=0):
        """ calculate inner product of relativistic dyad(s) """
        if other == 0: other = self
        if type(other) != type(self):
            errmsg("Argument must be a dyad"); return
        return self.gin() * other
    def oprod_gin(self, other=0):
        """ calculate outer product of relativistic dyad(s) """
        if other == 0: other = self
        if type(other) != type(self):
            errmsg("Argument must be a dyad"); return
        return self * other.gin()
    def iprod_con(self, other=0):
        """ calculate inner product of relativistic dyad(s) """
        if other == 0: other = self
        if type(other) != type(self):
            errmsg("Argument must be a dyad"); return
        return self.con() * other
    def oprod_con(self, other=0):
        """ calculate outer product of relativistic dyad(s) """
        if other == 0: other = self
        if type(other) != type(self):
            errmsg("Argument must be a dyad"); return
        return self * other.con()
    def expect(self, oper):
        """ calculate the expectation value for an operator """
        if type(oper) != type(self):
            errmsg("Operator must be a dyad"); return
        return ~self * oper * self
    def comm(self, other):
        """ calculate commutator product """
        if type(other) != type(self):
            errmsg("Argument must be a dyad"); return
        return (self*other - other*self)
    def acomm(self, other):
        """ calculate anti-commutator product """
        if type(other) != type(self):
            errmsg("Argument must be a dyad"); return
        return (self*other + other*self)
    def bar(self):
        """ spatialy reverse the right-hand dyad  """
        return self.con_right()
    def adj(self):
        """ calculate Dirac adjoint """
        return ~self * dyad(_e3,_one)
    def bar_w(self):
        """ calculate Weyl adjoint """
        return ~self * dyad(_e1,_one)
    def rev(self):
        """ reverse odd gamma or Dirac blades """
        return dyad(_e3,_one) * self * dyad(_e3,_one)
    def BAR(self):
        """ reverse Dirac and gamma blades """
        return ~self.rev()
    def compD(self):
        """ extract scalar and pseudoscalar blade as a complex dyad """ 
        sc = (self.value[0,0]-self.value[7,7]) * _one
        ps = (self.value[7,0]+self.value[0,7]) * _e123
        return sc + ps
    def comp(self):
        """ extract scalar and pseudoscalar blade as a complex multivector """ 
        sc = (self.value[0,0]-self.value[7,7]) * _one
        ps = (self.value[7,0]+self.value[0,7]) * _e123
        return (sc + ps).comp()
    def uufact(self, low=True):
        """ factor dyadic multivector in the up-up basis """
        list = self.dense()
        dd = dyad(_zero, _zero)
        for i in list.pair:
            lf = i.left.ufact(low, True)
            rf = i.right.ufact(low, True)
            if (lf == None or rf == None):
                errmsg("Object cannot be factored in the up-up basis")
                return
            dd += dyad(lf, rf)
        return dd
    def updn(self, evec=None, fvec=None):
        """ find the 2x2 spectral coordinates of a dyadic spinor """
        if evec == None: evec = dyad(_e3, _one)
        if fvec == None: fvec = dyad(_e1, _one)
        ket = dyad(_one,_one) + evec.normal()
        vec = fvec.normal()
        a = (self*ket).compD()
        b = (self*vec*ket).compD()
        c = (self*ket*vec*ket).compD()
        d = (self*vec*ket*vec).compD()
        return (a, b, c, d)
    def UPDN(self, evec=None, lvec=None, rvec=None):
        """ find the 2x2 spectral coordinates of a dyadic spinor """
        if evec == None: evec = dyad(_one + _e3, _one + _e3)
        if lvec == None: lvec = dyad(_e1, _one)
        if rvec == None: rvec = dyad(_one, _e1)
        a = (self*evec).compD()
        b = (self*evec*rvec).compD()
        c = (self*evec*lvec).compD()
        d = (self*evec*rvec*lvec).compD()
        return (a, b, c, d)
    def scde(self, evec=None, fvec=None):
        """ find the projected coordinates of a dyadic spinor """
        if evec == None: evec = dyad(_e3, _one)
        if fvec == None: fvec = dyad(_e1, _one)
        dvec = dyad(_one,_e123)*fvec*evec        
        s = (self).compD()
        c = (self*fvec).compD()
        d = (self*dvec).compD()
        e = (self*evec).compD()
        return (s, c, d, e)
    def toArray(self, basis):
        """ convert multivector to 4x4 complex array """
        a = zeros((4,4),dtype=complex)
        t = basis.T
        for i in range(4):
            for j in range(4):
                a[i,j] = (4*self*t[i,j]).comp()
        return a
    def bra2(self, evec=None, fvec=None):
        """ find the conjugate transpose in updn coordinates """
        if evec == None: evec = dyad(_e3, _one)
        if (evec**2).comp() != 1.0:
            evec = evec.normal()
        if fvec == None: fvec = dyad(_e1, _one)
        ket = (dyad(_one,_one)+evec)
        vec = fvec
        a = dyad(~((self*ket).compD()),_one)*ket/2
        b = dyad(~((self*vec*ket).compD()),_one)*vec*ket/2
        c = dyad(~((self*ket*vec).compD()),_one)*ket*vec/2
        d = dyad(~((self*vec*ket*vec).compD()),_one)*vec*ket*vec/2
        return a+b+c+d
    def bra(self, basis):
        """ find the conjugate transpose of a dyad """
        ar = self.toArray(basis)
        return tract(ar.conj().T, basis)
    def normal(self):
        """ normalize a dyadic biparavector """
        mag = self.mag()
        return self*dyad(_one/mag,_one)
    def snormal(self):
        """ scalar normalize a dyadic biparavector """
        mag = self.comp()
        return self*dyad((_one*1/mag),_one)
    def mag(self):
        """ calculate the magnitude of a dyadic biparavector """
        mag = sqrt((self**2).comp())
        # magc = sqrt(mag.value[0] + 1j*mag.value[7])
        return _one*mag.real + _e123*mag.imag
    def dt(self):
        """ calculate derivative with respect to time """
        return 4*abs(self.expect(dyad(_one,_one))(0))
    def dx(self):
        """ calculate derivative with respect to e1 """
        return 4*abs(self.expect(dyad(_e1,_e1))(0))
    def dy(self):
        """ calculate derivative with respect to e2 """
        return 4*abs(self.expect(dyad(_e1,_e2))(0))
    def dz(self):
        """ calculate derivative with respect to e3 """
        return 4*abs(self.expect(dyad(_e1,_e3))(0))
    def trace(self):
        """ calculate trace in Bloch space """
        return self.value[0,0]+self.value[1,1]+self.value[2,2]+self.value[3,3]
    def hileft(self):
        """ align high blades on left-hand side of dyad """
        oa = array(self.value)
        oa[:,[0,1,2,3]] = 0
        hi = dyad.fromArray(oa)
        return self - hi + hi*dyad(_e123,-_e123)
        return self - self.high + self.high*dyad(_e123,-_e123)
    hile = hileft
    hili = hileft
    def hiright(self):
        """ align high blades on right-hand side of dyad """
        oa = array(self.value)
        oa[[0,1,2,3],:] = 0
        hi = dyad.fromArray(oa)
        return self - hi + hi*dyad(_e123,-_e123)
    hiri = hiright
    def gammaV(self):
        s = self.value[3,0]*_one
        x = -self.value[5,1]*_e1
        y = -self.value[5,2]*_e2
        z = -self.value[5,3]*_e3
        return s + x + y + z
    @property
    def boost(self):
        """ extract boost dyadic blades """ 
        db = zeros((8,8))
        dl = (0,0),(1,1),(1,2),(1,3)
        for i in dl: db[i] = self.value[i]
        return dyad.fromArray(db)
    @property
    def gamma(self):
        """ extract Gamma dyadic blades """ 
        gb = zeros((8,8))
        gl = (0,0),(3,0),(5,1),(5,2),(5,3)
        for i in gl: gb[i] = self.value[i]
        return dyad.fromArray(gb)
    @property
    def dirac(self):
        """ extract Dirac dyadic blades """ 
        gb = zeros((8,8))
        gl = (0,0),(3,0),(1,1),(1,2),(1,3)
        for i in gl: gb[i] = self.value[i]
        return dyad.fromArray(gb)
    @property
    def dng(self):
        """ extract boost and Dirac dyadic blades """ 
        gb = zeros((8,8))
        gl = (0,0),(3,0),(1,1),(1,2),(1,3),(5,1),(5,2),(5,3)
        for i in gl: gb[i] = self.value[i]
        return dyad.fromArray(gb)
    @property
    def nbnd(self):
        """ extract not a boost or Dirac dyadic blade """
        return self-self.dng
    @property
    def weyl(self):
        """ extract Weyl dyadic blades """ 
        wb = zeros((8,8))
        wl = (0,0),(1,0),(5,1),(5,2),(5,3)
        for i in wl: wb[i] = self.value[i]
        return dyad.fromArray(wb)
    @property
    def motor(self):
        """ extract odd-on-right boost blades """ 
        bb = zeros((8,8))
        bl = (1,1),(1,2),(1,3)
        for i in bl: bb[i] = self.value[i]
        return dyad.fromArray(bb)
    @property
    def rotor(self):
        """ extract even-on-right unit blades """ 
        rb = zeros((8,8))
        rl = (0,0),(0,4),(0,5),(0,6)
        for i in rl: rb[i] = self.value[i]
        return dyad.fromArray(rb)
    @property
    def bell(self):
        """ extract bell state blades """ 
        ba = zeros((8,8))
        bs = (0,0),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7)
        for i in bs: ba[i] = self.value[i]
        return dyad.fromArray(ba)
    @property
    def conform(self):
        """ extract conformal representation blades """ 
        ba = zeros((8,8))
        bs = (0,0),(3,0),(5,0),(1,1),(1,2),(1,3)
        for i in bs: ba[i] = self.value[i]
        return dyad.fromArray(ba)
    @property
    def even(self):
        """ extract even-on-right blades """ 
        ea = array(self.value)
        ea[:,[1,2,3,7]] = 0
        return dyad.fromArray(ea)
    @property
    def odd(self):
        """ extract odd-on-right blades """ 
        oa = array(self.value)
        oa[:,[0,4,5,6]] = 0
        return dyad.fromArray(oa)
    @property
    def high(self):
        """ extract high-on-right blades """ 
        ea = array(self.value)
        ea[:,[0,1,2,3]] = 0
        return dyad.fromArray(ea)
    @property
    def low(self):
        """ extract low-on-right blades """ 
        oa = array(self.value)
        oa[:,[4,5,6,7]] = 0
        return dyad.fromArray(oa)
    @property
    def scalar(self):
        """ extract scalar-on-right blades """ 
        ea = array(self.value)
        ea[:,[1,2,3,4,5,6]] = 0
        return dyad.fromArray(ea)
    @property
    def vector(self):
        """ extract vector-on-right blades """ 
        oa = array(self.value)
        oa[:,[0,7]] = 0
        return dyad.fromArray(oa)
    @property
    def tvector(self):
        """ extract complex gamma-vector blades """ 
        ca = zeros((8,8))
        cl = (3,0),(4,0),(2,1),(2,2),(2,3),(5,1),(5,2),(5,3)
        for i in cl: ca[i] = self.value[i]
        return dyad.fromArray(ca)
    @property
    def cvector(self):
        """ extract complex gamma-vector blades """ 
        ca = zeros((8,8))
        cl = ((3,0),(4,0),
              (1,1),(1,2),(1,3),(2,1),(2,2),(2,3),
              (5,1),(5,2),(5,3),(6,1),(6,2),(6,3))
        for i in cl: ca[i] = self.value[i]
        return dyad.fromArray(ca)
    @property
    def zero(self):
        """ extract scalar-scalar blade as a float """ 
        return self.value[0,0]

# define a triple-multivector (separable) class

class triple(object):
    """ separable tri[ple of multivectors """
    def __init__(self, mv1, mv2, mv3):
        if type(mv1) == float or type(mv1) == int: mv1 *= _one
        if type(mv2) == float or type(mv2) == int: mv2 *= _one
        if type(mv3) == float or type(mv3) == int: mv3 *= _one
        self.left = mv1
        self.center = mv2
        self.right = mv3
    def __repr__(self):
        """ create a displayable string for a pair """
        return "[(%s) o (%s) o (%s)]" % (self.left, self.center, self.right)
    def __mul__(self, other):
        """ multiply triples """
        if type(other) != type(self):
            errmsg("Argument must be a triple"); return
        return triple(self.left*other.left,
                      self.center*other.center,
                      self.right*other.right)
    def __truediv__(self, other):
        """ divide triples """
        if type(other) != type(self):
            errmsg("Argument must be a triple"); return
        return pair(self.left / other.left,
                    self.center / other.center,
                    self.right / other.right)
    def __div__(self, other):
        """ divide triples """
        return self.__truediv__(other)
    def __eq__(self, other):
        """ compare triples """
        if type(other) != type(self):
            errmsg("Argument must be a triplw"); return
        if (self - other) == dyad(_zero, _zero): return True
        return False
    def zero(self):
        """ zero out a triple """
        self.left = self.right = _zero
    def scale(self, scale1=1.0, scale2=1.0):
        """ multiply left, divide right scaling """
        return triple(self.left*float(scale1*scale2),
                      self.center/float(scale1),
                      self.right/float(scale2))

# define utility triple-multivector list class

class tripleList(list):
    """ list of triples """
    def __init__(self, triples):
        self.triples = triples
    def __repr__(self):
        """ create a displayable string for a triplelist """
        return ' +\n'.join('%s' % (i) for i in self.triples)

# define a general purpose three-qubit multivector class
# (a work in progress)

class triad(object):
    """ triadic multivector """
    def __init__(self, mv1, mv2, mv3):
        if type(mv1) == float or type(mv1) == int: mv1 *= _one
        if type(mv2) == float or type(mv2) == int: mv2 *= _one
        if type(mv3) == float or type(mv3) == int: mv3 *= _one
        self.value = (mv1.value.reshape(8,1)*mv2.value).reshape(8,8,1)*mv3.value
    @staticmethod
    def fromtriple(triple):
        """ construct a triad object from a triple object """
        return triad(triple.left, triple.center, triple.right)
    @staticmethod
    def fromArray(array):
        if array.shape != (8,8,8):
            errmsg("Array must be contain 8x8x8 elements"); return
        t = triad(_zero, _zero, _zero)
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    rr = (array[i,j,k]).real.copy()
                    if rr != 0.0:
                        t.value[i,j,k] += rr
                    jj = (array[i,j,k]).imag.copy()
                    if jj != 0.0:
                        if i in [2,4,6,7]: jj = -jj
                        t.value[7-i,j,k] += jj
        return t
        """ construct a triad object from an 8 x 8 x 8 array """
        t = triad(_zero, _zero, _zero)
        if t.value.shape == array.shape: t.value = (array)
        return t
    def __repr__(self):
        """ format a triad string """
        list = self.tolist()
        if len(list) == 0: return '[(0.0) o (0.0) o (0.0)]'
        return ' +\n'.join('%s' % (i) for i in list)
    def rep(self):
        """ create a displayable string for a triad """
        l = self.tolist()
        s = str(l[0])
        r = _linelen - len(s) 
        for i in range(len(l)-1):
            w = str(l[i+1])
            r -= len(w)
            if r > 1:
                s += ' + ' + w
            else:
                s += ' +\n' + w
                r = _linelen - len(w)
        return s
    def __add__(self, other):
        """ add two triads """
        if type(other) != type(self):
            errmsg("Argument must be a triad"); return
        return triad.fromArray(self.value + other.value)
    def __sub__(self, other):
        """ subtract two triads """
        if type(other) != type(self):
            errmsg("Argument must be a triad"); return
        return triad.fromArray(self.value - other.value)
    def __mul__(self, other):
        """ multiply a triad with something """
        if type(other) == type(self):
            l1 = self.tolist()
            l2 = other.tolist()
            newbit = triad.fromArray(zeros((8,8,8)))
            for i in l1:
                for j in l2:
                    newbit.value += triad.fromtriple(i * j).value
            return newbit 
        return triad.fromArray(self.value * other)
    def __rmul__(self, other):
        """ multiply something with a triad """
        return triad.fromArray(self.value * other)
    def __truediv__(self, other):
        """ divide a triad by something """
        if type(other) == type(self):
            l1 = self.tolist()
            l2 = other.tolist()
            newbit = triad.fromArray(zeros((8,8,8)))
            for i in l1:
                for j in l2:
                    newbit.value += triad.fromtriple(i / j).value
            return newbit 
        return triad.fromArray(self.value / other)
    def __div__(self, other):
        return self.__truediv__(other)
    def __neg__(self):
        """ calculate the negative of a triad """
        return triad.fromArray(self.value * -1.0) 
    def __eq__(self, other):
        """ compare two triads for equality """
        if type(other) != type(self): return False
        if(self.value.all() == other.value.all()): return True
        return False
    def __invert__(self):
        """ calculate triad reverse """
        return triad.fromArray(self.value * _high_three)
    def __call__(self, grade):
        """ return specific grade triadic blade """
        if grade == 0: return self.value[0][0][0]*triad(_one,_one,_one)
        if grade == 3: return self.value[7][7][7]*triad(_e123,_e123,_e123)
        errmsg("Unsuppored grade")
        return 0 
    def __abs__(self):
        """ calculate triad magnitude """
        return sqrt((self*~self).value[0][0][0])
    def __pow__(self, other):
        """ raise a triad to an integer power """
        if not isinstance(other, (int)):
            errmsg("Exponent must be an integer"); return
        if other < 1:
            errmsg("Exponent must be a positive integer"); return
        tt = self
        for i in range(1, other):
            tt *= self
        return tt
    def __rpow__(self, other):
        """ exponentiate a triad """
        if other == e: return self.exp(_nonstd)
        errmsg("Only natural exponentials supported"); return
    def dot(self, other):
        """ calculate scalar product of triads """
        return tensordot((~self).value, other.value, axes=3).item(0)*triad(_one,_one,_one)
    def exp(self, nonstd=False):
        """ approximate the power series sum of a triad """
        term = triad(_one,_one,_one)
        if nonstd == True:
            ss = ~self*self
            pp = ss.prop(ss**2)
            if pp != 0: term = ss/pp
        tot = term
        for i in range(_ecnt):
            term *= self/(i+1)
            tot += term
        return tot
    def prop(self, other):
        """ determine if two triads are proportional """
        if type(other) != type(self):
            errmsg("Argument must be a triad"); return 0
        aa = ma.masked_equal(self.value, _minnum)
        bb = ma.masked_equal(other.value, _minnum)
        if array_equal(aa.mask, bb.mask):
            pp = bb / aa
            if pp.mask.all() == False and abs(var(pp)) <= _minnum:
                return mean(pp)
        return 0
    def tolist(self):
        """ generate a list of triadic blades """
        list = []
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    if abs(self.value[i][j][k]) >= _minnum:
                        list += [triple(self.value[i][j][k]*_blist[i],
                                        _blist[j],
                                        _blist[k])]
        return list
    def dense(self, scale1=1.0, scale2=1.0):
        """ collect triadic blades into triples """
        list = self.tolist()
        if len(list) == 0: return tripleList([triple(_zero, _zero, _zero)])
        for i in range(len(list)):
            for j in range(i+1, len(list)):
                if list[i].right == list[j].right and list[i].center == list[j].center:
                    list[i].left += list[j].left
                    list[j].zero()
        for i in range(len(list)):
            for j in range(i+1, len(list)):
                if list[i].center == list[j].center:
                    cc = list[i].left.prop(list[j].left)
                    if cc != 0:
                        list[i].right += cc*list[j].right
                        list[j].zero()
        for i in range(len(list)):
            for j in range(i+1, len(list)):
                if list[i].right == list[j].right:
                    cc = list[i].left.prop(list[j].left)
                    if cc != 0:
                        list[i].center += cc*list[j].center
                        list[j].zero()
        if scale1 == 'r':    
            return tripleList([i.scale(abs(i.right)) for i in list if abs(i.left) > _minnum])
        if scale1 == 'l':
            return tripleList([i.scale(1.0/abs(i.left)) for i in list if abs(i.left) > _minnum])
        return tripleList([i.scale(scale1, scale2) for i in list if abs(i.left) > _minnum])
    def tilde(self):
        """ reverse triad blades """
        return triad.fromArray(self.value * _high_three)
    def normal(self):
        """ convert to unit triad """
        return self / abs(self)
    def iprod(self, other=0):
        """ calculate inner product of triads """
        if other == 0: other = self
        if type(other) != type(self):
            errmsg("Argument must be a triad"); return
        return self.tilde() * other
    def oprod(self, other=0):
        """ calculate outer product of triads """
        if other == 0: other = self
        if type(other) != type(self):
            errmsg("Argument must be a triad"); return
        return self * other.tilde()
    def toArray(self, basis):
        """ convert multivector to 8x8 complex array """
        a = zeros((8,8),dtype=complex)
        t = basis.T
        for i in range(8):
            for j in range(8):
                a[i,j] = (8*self*t[i,j]).comp()
        return a
    def comp(self):
        """ extract scalar and pseudoscalar blade as a complex multivector """ 
        ar = self.value
        sc = _one * (ar[0,0,0]-ar[0,7,7]-ar[7,0,7]-ar[7,7,0])
        ps = _e123 * (ar[0,0,7]+ar[0,7,0]+ar[7,0,0]-ar[7,7,7])
        return (sc + ps).comp()
    def hileft(self):
        """ align high blades on left-hand side of dyad """
        oa = array(self.value)
        oa[:,:,[0,1,2,3]] = 0
        hi = triad.fromArray(oa)
        it = self - hi + hi*triad(_e123, _one,-_e123)
        oa = array(it.value)
        oa[:,[0,1,2,3],:] = 0
        hi = triad.fromArray(oa)
        return it - hi + hi*triad(-_e123, _e123, _one)
    hile = hileft
    hili = hileft
    def hiright(self):
        """ align high blades on right-hand side of dyad """
        oa = array(self.value)
        oa[[0,1,2,3],:,:] = 0
        hi = triad.fromArray(oa)
        it = self - hi + hi*triad(_e123, _one,-_e123)
        oa = array(it.value)
        oa[:,[0,1,2,3],:] = 0
        hi = triad.fromArray(oa)
        return it - hi + hi*triad(_one, -_e123, _e123)
    hiri = hiright

# define the internal blades

_zero = mvec(layout,[0.,0.,0.,0.,0.,0.,0.,0.])

_one = mvec(layout, (1.,0.,0.,0.,0.,0.,0.,0.))

_e1 = mvec(layout, (0.,1.,0.,0.,0.,0.,0.,0.))
_e2 = mvec(layout, (0.,0.,1.,0.,0.,0.,0.,0.))
_e3 = mvec(layout, (0.,0.,0.,1.,0.,0.,0.,0.))

_e12 = mvec(layout, (0.,0.,0.,0.,1.,0.,0.,0.))
_e13 = mvec(layout, (0.,0.,0.,0.,0.,1.,0.,0.))
_e23 = mvec(layout, (0.,0.,0.,0.,0.,0.,1.,0.))

_e123 = mvec(layout, (0.,0.,0.,0.,0.,0.,0.,1.))
'''
_zero = mvec(layout, array([0.,0.,0.,0.,0.,0.,0.,0.]))
_one = mvec(layout, array([1.,0.,0.,0.,0.,0.,0.,0.]))

_e1 = blades['e1']
_e2 = blades['e2']
_e3 = blades['e3']
_e12 = blades['e12']
_e23 = blades['e23']
_e13 = blades['e13']
_e123 = blades['e123']
'''

# define some helper arrays

_blist = [_one, _e1, _e2, _e3, _e12, _e13, _e23, _e123]

_neg_high  = array([1,  1,  1,  1, -1, -1, -1, -1])
_neg_odd   = array([1, -1, -1, -1,  1,  1,  1, -1])
_neg_vec   = array([1, -1, -1, -1, -1, -1, -1,  1])
_neg_dirac = array([1, -1, -1,  1, -1,  1,  1, -1])
_neg_trans = array([1,  1, -1,  1,  1, -1,  1,  1])
_neg_bi    = array([1,  1,  1,  1, -1, -1, -1,  1])

_dirac_right = ones(8).reshape(8,1) * _neg_dirac
_dirac_left = _neg_dirac.reshape(8,1) * ones(8)
_dirac_both = _neg_dirac.reshape(8,1) * _neg_dirac

_high_right = ones(8).reshape(8,1) * _neg_high
_high_left = _neg_high.reshape(8,1) * ones(8)
_high_both = _neg_high.reshape(8,1) * _neg_high

_odd_right = ones(8).reshape(8,1) * _neg_odd
_odd_left = _neg_odd.reshape(8,1) * ones(8)
_odd_both = _neg_odd.reshape(8,1) * _neg_odd

_vec_right = ones(8).reshape(8,1) * _neg_vec
_vec_left = _neg_vec.reshape(8,1) * ones(8)
_vec_both = _neg_vec.reshape(8,1) * _neg_vec

_dirac_high = _neg_dirac.reshape(8,1) * _neg_high
_dirac_odd = _neg_dirac.reshape(8,1) * _neg_odd
_dirac_vec = _neg_dirac.reshape(8,1) * _neg_vec

_trans_left = _neg_trans.reshape(8,1) * ones(8)
_trans_right = ones(8).reshape(8,1) * _neg_trans
_trans_both = _neg_trans.reshape(8,1) * _neg_trans


_high_three = (_neg_high.reshape(8,1) * _neg_high).reshape(8,8,1) * _neg_high

_da = array([[1, 0, 0, 0, 0, 0, 0,-1],
            [0, 1, 0, 0, 0, 0,-1, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 1,-1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]])
    
_sa = array([[1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0,-1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]])
    
_ua = array([[1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0,-1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]])  

_dyad  = dyad(_one, _one)
_triad = triad(_one, _one, _one) 
    
# define some helper functions

def tract(mat1, mat2):
    """ calculate the contraction of two matrices """
    return tensordot(mat1, mat2).item(0)

def boost1(left, right):
    """ create a boost gate in the +/- e1 basis """
    return e**(left)*(1+_e1)/2 + e**(right)*(1-_e1)/2

def boost2(left, right):
    """ create a boost gate in the +/- e2 basis """
    return e**(left)*(1+_e2)/2 + e**(right)*(1-_e2)/2

def boost3(left, right):
    """ create a boost gate in the +/- e3 basis """
    return e**(left)*(1+_e3)/2 + e**(right)*(1-_e3)/2

def phase1(left, right):
    """ create a phase gate in the +/- e1 basis """
    return e**(_e123*left)*(1+_e1)/2 + e**(_e123*right)*(1-_e1)/2

def phase2(left, right):
    """ create a phase gate in the +/- e2 basis """
    return e**(_e123*left)*(1+_e2)/2 + e**(_e123*right)*(1-_e2)/2

def phase3(up, dn):
    """ create a phase gate in the +/- e3 basis """
    return e**(_e123*up)*(1+_e3)/2 + e**(_e123*dn)*(1-_e3)/2

def PS1(left, right):
    """ create a simplified phase gate in the +/- e1 basis """
    return phase1(left*pi/12.0, right*pi/12.0)

def PS2(left, right):
    """ create a simplified phase gate in the +/- e2 basis """
    return phase2(left*pi/12.0, right*pi/12.0)

def PS3(up, dn):
    """ create a simplified phase gate in the +/- e3 basis """
    return phase3(up*pi/12.0, dn*pi/12.0)

def R13(alice, bob):
    """ create a pair of rotations in the e13 plane """
    return dyad(PS2(alice,-alice), PS2(bob,-bob))

def dip(mv1, mv2=None):
    """ alternate Dirac inner product """
    if mv2 == None: mv2 = mv1
    return acomm(mv1,mv2).sc

def iprod(mv1, mv2=0):
    """ calculate the Dirac outer product """
    if mv2 == 0: return ~mv1*mv1
    return ~mv1*mv2

def oprod(mv1, mv2=0):
    """ calculate the Dirac outer product """
    if mv2 == 0: return mv1*~mv1
    return mv1*~mv2

def bracket(mv1, mv2=None):
    """ alternate Dirac inner product """
    if mv2 == None: mv2 = mv1
    return acomm(mv1,mv2).sc

def rprod(mv1, mv2=0):
    """ calculate the relativistic Dirac inner product """
    if mv2 == 0: return mv1.bar()*mv1
    return mv1.bar()*mv2

def comm(mv1, mv2):
    """ simple commutator calculation """
    return (mv1*mv2 - mv2*mv1)

def acomm(mv1, mv2):
    """ simple anti-commutator calculation """
    return (mv1*mv2 + mv2*mv1)

def rcomm(mv1, mv2):
    """ relativistic anti-commutator calculation """
    return (mv1*mv2.bar() - mv2*mv1.bar())

def racomm(mv1, mv2):
    """ relativistic anti-commutator calculation """
    return (mv1*mv2.bar() + mv2*mv1.bar())

def tcomm(mv1, mv2):
    """ tilde commutator calculation """
    return (mv1*mv2 - mv2*mv1.tilde())

def tacomm(mv1, mv2):
    """ tilde anti-commutator calculation """
    return (mv1*mv2 + mv2*mv1.tilde())

def rapido(theta, si, ss):
    """ find output rapidity given a deflection angle and reference boost """
    ba = si.dePara()[1]
    sa = si/si.dePara()[0]
    aa = cosh(ba)-sinh(ba)*cos(theta)
    bb = float((sa*ss.bar())(0))
    cc = cosh(ba)+sinh(ba)*cos(theta)
    dd = bb**2 - aa*cc
    if dd < 0:
        print("No real solutions")
        print(aa, bb, cc)
        return 0
    return log((bb + sqrt(dd))/aa)

def rapid(si, ss, vv, plus=True):
    """ find the rapidity for a given input boost and output direction """
    aa = si.bar().dot(1+vv).zero
    bb = si.bar().dot(ss).zero
    cc = si.bar().dot(1-vv).zero
    dd = bb*bb - aa*cc
    if dd < 0:
        print("No real solutions")
        print(aa/2, bb, cc/2)
        return 0
    if (plus == False): return log((bb - sqrt(dd))/aa)
    return log((bb + sqrt(dd))/aa)
def deflect(bb, si, ss):
    """ find deflection angle given an output rapidity and reference boost """
    ba = si.dePara()[1]
    sa = si/si.dePara()[0]
    sb = ss/ss.dePara()[0]
    dd = float((sa*sb.bar())(0))
    arg = float((cosh(ba)*cosh(bb)-dd)/(sinh(ba)*sinh(bb)))
    if abs(arg) > 1.0: 
        print("No solution")
        return 0
    return arccos(arg)

def randComp(pauli=False):
    """ generate a random complex """
    re = 2*(random.rand()-1/2)
    im = 2*(random.rand()-1/2)
    if pauli == True: return (re*_one + im*_e12)
    return (re*_one + im*_e123)

def compPair():
    """ generate a normalized complex pair """
    a = randComp()
    b = randComp()
    m = sqrt(abs(~a*a+~b*b))
    return a/m, b/m

def compQuad():
    """ generate a set of four normalized complex multivectors """
    a = randComp()
    b = randComp()
    c = randComp()
    d = randComp()
    m = sqrt(abs(~a*a+~b*b+~c*c+~d*d))
    return (a/m, b/m, c/m, d/m)

def randBoost():
    """ generate a random boost """
    scale=2.0
    r = scale*(random.rand()-1/2)
    v1 = scale*(random.rand()-1/2)*_e1
    v2 = scale*(random.rand()-1/2)*_e2
    v3 = scale*(random.rand()-1/2)*_e3
    v = (v1+v2+v3).normal()
    return (e**(v*r)).low

def randPhasor():
    """ generate a random complex exponential """
    re = 2*(random.rand()-1/2)
    im = 2*(random.rand()-1/2)
    return (e**(re+_e123*im)).scalar

def randCrank():
    """ generate a random boost and phasor """
    scale=2.0
    r = scale*(random.rand()-1/2)
    v1 = scale*(random.rand()-1/2)*_e1
    v2 = scale*(random.rand()-1/2)*_e2
    v3 = scale*(random.rand()-1/2)*_e3
    v = (v1+v2+v3).normal()
    im = scale*(random.rand()-1/2)*_e123
    return (e**(v*r+im))
def randQuat():
    """ generate a random rotor """
    scale=2.0
    r = 2*pi*random.rand()
    m = scale*(random.rand()-1/2)
    b1 = (random.rand()-1/2)*_e12
    b2 = (random.rand()-1/2)*_e23
    b3 = (random.rand()-1/2)*_e13
    b = (b1+b2+b3).enormal()
    return (m*e**(b*r)).even

def randScrew():
    """ generate a random screw """
    scale=2.0
    z = scale*(random.rand()-1/2)*_one + scale*(random.rand()-1/2)*_e123
    v1 = scale*(random.rand()-1/2)*_e1
    v2 = scale*(random.rand()-1/2)*_e2
    v3 = scale*(random.rand()-1/2)*_e3
    v = (v1+v2+v3).enormal()
    return e**(v*z)

def randVec():
    """ generate a random MultiVector """
    scale=2.0
    v1   = scale*(random.rand()-1/2)*_e1
    v2   = scale*(random.rand()-1/2)*_e2
    v3   = scale*(random.rand()-1/2)*_e3
    return v1+v2+v3

def randMV():
    """ generate a random MultiVector """
    scale=2.0
    v0   = scale*(random.rand()-1/2)
    v1   = scale*(random.rand()-1/2)*_e1
    v2   = scale*(random.rand()-1/2)*_e2
    v3   = scale*(random.rand()-1/2)*_e3
    v12  = scale*(random.rand()-1/2)*_e12
    v23  = scale*(random.rand()-1/2)*_e23
    v13  = scale*(random.rand()-1/2)*_e13
    v123 = scale*(random.rand()-1/2)*_e123
    return v0+v1+v2+v3+v12+v23+v13+v123
