from haystack.pipelines import Pipeline
from haystack.nodes import Crawler, PreProcessor
from haystack.document_stores import InMemoryDocumentStore

def crawl_data():
    # crawl only the articles
    urls = ["https://www.svpg.com/articles/"] + [f"https://www.svpg.com/articles/page/{i}/" for i in range(2, 29)]
    # for some pages you may need to add a loading_wait_time and disable-ipc-flooding-protection
    crawler = Crawler(
        urls=urls,
        crawler_depth=1,
        output_dir="svpg",
    )
    return crawler


crawler = crawl_data()
crawler.run()
