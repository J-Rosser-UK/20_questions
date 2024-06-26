import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append("")

import unittest
from tools.wikipedia import WikipediaTool
from wikipedia.exceptions import PageError, WikipediaException

class WikipediaToolTests(unittest.TestCase):

    def test_invoke_with_valid_search_query(self):
        tool = WikipediaTool()
        result = tool.invoke("Python (programming language)")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertNotEqual(result, "")

    def test_invoke_with_empty_search_query(self):
        tool = WikipediaTool()
        with self.assertRaises(WikipediaException):
            tool.invoke("")

    def test_invoke_with_invalid_search_query(self):
        tool = WikipediaTool()
        with self.assertRaises(PageError):
            tool.invoke("Cats with rainbow wings.")


if __name__ == "__main__":
    unittest.main()
