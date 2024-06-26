import wikipedia
from tools.base import BaseTool


class WikipediaTool(BaseTool):
    
    def invoke(self, search_query: str) -> str:
  
        try:
            # Fetch the page content
            page = wikipedia.page(search_query)

            # Extract the text
            text = page.content

            # Return the first 300 characters
            return text[:300]
        except wikipedia.exceptions.PageError as e:
            raise e
        except wikipedia.exceptions.DisambiguationError as e:
            raise e
        except wikipedia.exceptions.WikipediaException as e:
            raise e


