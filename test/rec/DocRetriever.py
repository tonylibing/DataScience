from datetime import datetime
from elasticsearch import Elasticsearch

class DocRetriever():
    def __init__(self, train_texts, test_texts):
        self.es = Elasticsearch()
        doc = {
            'author': 'kimchy',
            'text': 'Elasticsearch: cool. bonsai cool.',
            'timestamp': datetime.now(),
        }
        res = self.es.index(index="test-index", doc_type='tweet', id=1, body=doc)
        print(res['created'])

        res = self.es.get(index="test-index", doc_type='tweet', id=1)
        print(res['_source'])

        self.es.indices.refresh(index="test-index")

        res = self.es.search(index="test-index", body={"query": {"match_all": {}}})
        print("Got %d Hits:" % res['hits']['total'])
        for hit in res['hits']['hits']:
            print("%(timestamp)s %(author)s: %(text)s" % hit["_source"])

    def query(self,keywords):
        self.es.search()