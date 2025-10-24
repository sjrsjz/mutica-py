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
        def io_handler(io: MuticaType, arg: MuticaType) -> MuticaType | None:
            io_py = io.as_py()
            if not isinstance(io_py, dict):
                return None
            if not io_py.get("kind") == "Opcode":
                return None
            io_type = io_py.get("opcode")
            if io_type[0] != "IO":
                return None
            io_name = io_type[1]
            match io_name:
                case "test_io":
                    print("IO Handler called with arg:", arg.as_py())
                    return MuticaType.integer_value(42)
                case _:
                    return None

        errors = self.engine.load(
            """
test_io!("Hello, Mutica!")
""",
            None,
            self.gc,
        )
        self.engine.set_io_handler(io_handler)
        for error in errors:
            print(error)
        while self.engine.step(self.gc):
            print((self.engine.get_current_type().as_py()))


if __name__ == "__main__":
    unittest.main()
