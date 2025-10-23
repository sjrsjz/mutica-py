import unittest
from mutica_py import MuticaType, MuticaGC, MuticaEngine

class TestMuticaBindings(unittest.TestCase):
    def setUp(self):
        self.gc = MuticaGC()
        self.engine = MuticaEngine()

    def test_integer_type(self):
        integer_type = MuticaType.integer()
        self.assertIsInstance(integer_type, MuticaType)

    def test_char_type(self):
        char_type = MuticaType.char()
        self.assertIsInstance(char_type, MuticaType)

    def test_engine_load(self):
        errors = self.engine.load("let x: any = 1;", None, self.gc)
        for error in errors:
            print(error)
        
    def test_engine_step(self):
        errors = self.engine.load(
            """
let fib: any = rec f: match
    | 0 => 0
    | 1 => 1
    | n: any => f(n - 1) + f(n - 2)
    | panic;
fib(28)"""
            
            , None, self.gc)
        for error in errors:
            print(error)
        result = self.engine.step(self.gc)
        self.assertTrue(result)
        print(self.engine.get_current_type().as_py())

if __name__ == "__main__":
    unittest.main()