
import unittest

class TestBundleAdjustment(unittest.TestCase):

    def test_reparameterization(self):
        """Ensure the BA works when the camera pose is reparameterized.

        Notes
            Most of the literature assumes a camera is expressed as its own
            pose, not as a point transformation. That is, to transform a 3D
            world point X into a point P in the camera coordinate frame, one
            would need to compute
                P = R'(X - t).

            However, the conventions used in the BAL dataset assume a model
            which transforms points as
                P = RX + t.

            For solving the problem elegantly using analytical (read:
            hand-coded), we need to reparameterize the camera vectors. This test
            ensures the problem is still solved OK when this reparameterization
            is applied.

            To this end, we leverage numerical jacobians estimated using finite
            differences, instead of the analytical formulation, which is only
            available for the reparameterized (aka canonical), and not for the
            BAL dataset formulation.
        """
        self.fail("Not yet implemented.")
