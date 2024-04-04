
import unittest
from helloworld.hello import hello

class TestHelloWorld(unittest.TestCase):
    def test_hello_noargs(self):
        self.assertEqual(hello(), "Hello, World!")
        
    def test_hello_with_name(self):
        self.assertEqual(hello("Alice"), "Hello, Alice!")

if __name__ == '__main__':
    unittest.main()
