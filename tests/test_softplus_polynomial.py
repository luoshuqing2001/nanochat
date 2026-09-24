"""Dense numerical checks for the polynomial coefficients, independent of CUDA."""
import unittest
import numpy as np
from numpy.polynomial import polynomial as poly
from flash_attn_4.softplus import _DIRECT_COEFFS, _SHORT_LOG_COEFFS


class TestPolynomial(unittest.TestCase):
    def test_direct_value_and_derivative(self):
        for lo,cs in zip((0.,4.),_DIRECT_COEFFS):
            t=np.linspace(0,1,65537);a=lo+4*t
            ref=np.logaddexp(0,-a);grad=-1/(1+np.exp(a))
            val=poly.polyval(t,cs);der=poly.polyval(t,poly.polyder(cs))/4
            self.assertLess(float(np.max(np.abs(val-ref)/ref)),.00016)
            self.assertLess(float(np.max(np.abs(der-grad)/(-grad))),.0005)
            self.assertTrue(np.all(val>0))
            self.assertTrue(np.all(der<0))
            for j in (0,-1):
                self.assertAlmostEqual(val[j],ref[j],places=12)
                self.assertAlmostEqual(der[j],grad[j],places=12)

    def test_short_log_relative_error(self):
        y=np.linspace(0,1,65537);ref=np.ones_like(y)
        ref[1:]=np.log1p(y[1:])/y[1:]
        for degree,tol in ((3,.00065),(4,.000095)):
            c=_SHORT_LOG_COEFFS[degree];value=poly.polyval(y,c)
            self.assertLess(float(np.max(np.abs(value-ref)/ref)),tol)
            self.assertEqual(value[0],1.)
            self.assertAlmostEqual(value[-1],np.log(2),places=14)
            self.assertTrue(np.all(poly.polyval(y,poly.polyder([0,*c]))>0))

if __name__=='__main__':unittest.main()
