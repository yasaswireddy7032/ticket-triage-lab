import unittest
from app import *

class TicketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.model=fit(examples())
    def test_empty_abstains(self):self.assertEqual(classify(self.model,'')['route'],'human_review')
    def test_unknown_abstains(self):self.assertEqual(classify(self.model,'zyxwvuts')['reason'],'no_known_terms')
    def test_confident_route(self):self.assertEqual(classify(self.model,'password reset link expired',.3)['route'],'access')
    def test_high_threshold(self):self.assertEqual(classify(self.model,'invoice',1)['route'],'human_review')

if __name__ == '__main__': unittest.main()
