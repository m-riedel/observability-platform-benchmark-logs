#!/usr/bin/python

from __future__ import (absolute_import, division, print_function)

__metaclass__ = type

DOCUMENTATION = r'''
---
module: query_benchmark

short_description: This module is used to benchmark observability-backends of query performance on logs.

version_added: "1.0.0"

description: This module is used to benchmark observability-backends of query performance on logs. It supports Loki, Elasticsearch and OpenSearch. The module will return the query response time and the duration of the query.

options:
    backend_type:
        description: The type of backend to benchmark. It can be loki, elasticsearch or opensearch.
        required: true
        type: str
    max_start:
        description: The earliest time range to query logs from.
        required: true
        type: int
    max_end:
        description: The latest time range to query logs from.
        required: true
        type: int
    host:
        description: The host of the backend to benchmark.
        required: true
        type: str
    port:
        description: The port of the backend to benchmark.
        required: true
        type: int
    protocol:
        description: The protocol of the backend to benchmark.
        required: true
        type: str

author:
    - Max Riedel (@m-riedel)
'''

EXAMPLES = r'''
'''

RETURN = r'''
# These are examples of possible return values, and in general should use other names for return values.
query:
    description: The query that was executed.
    type: str
    returned: always
    sample: '"{job=\"fluentbit\"}"'
run_start: 
    description: The start time of the query in nanoseconds.
    type: int
    returned: always
    sample: 1634020000000000000
run_end:
    description: The end time of the query in nanoseconds.
    type: int
    returned: always
    sample: 1634020000000000000
run_duration:
    description: The duration of the query in nanoseconds.
    type: int
    returned: always
    sample: 1000000000
response_time:
    description: The response time of the query in nanoseconds.
    type: int
    returned: always
    sample: 1000000000
response:
    description: The response of the query.
    type: dict
    returned: always
    sample: {"status": {"duration": 1000000000}}    
'''

from ansible.module_utils.basic import AnsibleModule
import time
import requests
import concurrent.futures
from elasticsearch import Elasticsearch
from opensearchpy import OpenSearch
import math
import random


class LokiClient:
    def __init__(self, host="loki.benchmark.local", port=80, protocol="http"):
        self.host = host
        self.port = port
        self.protocol = protocol

    def query_range(self, query, start, end, limit=100):
        # print(f"Querying {query} from {start} to {end} on {self.protocol}://{self.host}:{self.port}")
        url = f"{self.protocol}://{self.host}:{self.port}/loki/api/v1/query_range"
        params = {
            "query": query,
            "start": start,
            "end": end,
            "limit": limit
        }
        response = requests.get(url, params=params)
        return response.json()

    def query(self, query, time, limit=100):
        # print(f"Querying {query} at {time} on {self.protocol}://{self.host}:{self.port}")
        url = f"{self.protocol}://{self.host}:{self.port}/loki/api/v1/query"
        params = {
            "query": query,
            "time": time,
            "limit": limit
        }
        response = requests.get(url, params=params)
        return response.json()


class SignozClient:
    def __init__(self, host="query.signoz.benchmark.local", port=80, protocol="http",
                 username="benchmark@benchmark.local", password="benchmark", init_user=False):
        self.host = host
        self.port = port
        self.protocol = protocol
        self.username = username
        self.password = password

        if init_user:
            self.register_initial_admin()

        self.access_token = self.get_access_token()

    def register_initial_admin(self):
        url = f"{self.protocol}://{self.host}:{self.port}/api/v1/register"
        data = {
            "email": self.username,
            "password": self.password,
            "name": "Test",
            "orgName": "Test"
        }
        response = requests.post(url, json=data)
        return response.json()

    def get_access_token(self):
        url = f"{self.protocol}://{self.host}:{self.port}/api/v1/login"
        data = {
            "email": self.username,
            "password": self.password
        }
        response = requests.post(url, json=data)

        return response.json()['accessJwt']

    def query_range_v1(self, start, end, query="", limit=100):
        url = f"{self.protocol}://{self.host}:{self.port}/api/v1/logs"

        params = {
            "q": query,
            "limit": limit,
            "orderBy": "timestamp",
            "order": "desc",
            "timestampStart": start,
            "timestampEnd": end
        }

        headers = {
            "Authorization": f"Bearer {self.access_token}"
        }

        response = requests.get(url, params=params, headers=headers)

        return response.json()

    def query_range_v3(self, start, end, limit=100):
        url = f"{self.protocol}://{self.host}:{self.port}/api/v3/query_range"

        payload = {
            "start": start,
            "end": end,
            "step": 60,
            "variables": {},
            "compositeQuery": {
                "queryType": "builder",
                "panelType": "list",
                "fillGaps": False,
                "builderQueries": {
                    "A": {
                        "aggregateAttribute": {
                            "dataType": "",
                            "id": "------false",
                            "isColumn": False,
                            "isJson": False,
                            "key": "",
                            "type": ""
                        },
                        "aggregateOperator": "noop",
                        "dataSource": "logs",
                        "disabled": False,
                        "expression": "A",
                        "filters": {
                            "items": [],
                            "op": "AND"
                        },
                        "functions": [],
                        "groupBy": [],
                        "having": [],
                        "legend": "",
                        "limit": None,
                        "offset": 0,
                        "orderBy": [{
                            "columnName": "timestamp",
                            "order": "desc"
                        }],
                        "pageSize": limit,
                        "queryName": "A",
                        "reduceTo": "avg",
                        "spaceAggregation": "sum",
                        "stepInterval": 60,
                        "timeAggregation": "rate",
                    }
                }
            }
        }

        headers = {
            "Authorization": f"Bearer {self.access_token}"
        }

        response = requests.post(url, json=payload, headers=headers)

        return response.json()


class OpenObserveClient:
    def __init__(self, host="openobserve.benchmark.local", port=80, protocol="http",
                 username="benchmark@benchmark.local", password="benchmark@benchmark.local", stream="benchmark",
                 org="default"):
        self.host = host
        self.port = port
        self.protocol = protocol
        self.username = username
        self.password = password
        self.stream = stream
        self.org = org

    def search(self, start, end, limit=100):
        url = f"{self.protocol}://{self.host}:{self.port}/api/{self.org}/_search"
        payload = {
            "query": {
                "sql": f"SELECT * FROM {self.stream}",
                "start_time": start,
                "end_time": end,
                "from": 0,
                "size": limit
            }
        }
        auth = (self.username, self.password)

        response = requests.post(url, json=payload, auth=auth)

        return response.json()


class BenchmarkInfo:
    def __init__(self, query, run_start, run_end, run_duration, response, run_number, query_start, query_end):
        self.query_start = query_start
        self.query_end = query_end
        self.query = query
        self.run_start = run_start
        self.run_end = run_end
        self.run_duration = run_duration
        self.response = response
        self.run_number = run_number

    def to_dict(self):
        return {
            "query": self.query,
            "query_start": self.query_start,
            "query_end": self.query_end,
            "run_start": self.run_start,
            "run_end": self.run_end,
            "run_duration": self.run_duration,
            "response": self.response,
            "run_number": self.run_number,
        }

    def __str__(self):
        return f"Query: {self.query}\nRun Start: {self.run_start}\nRun End: {self.run_end}\nRun Duration: {self.run_duration}\nResponse: {self.response}\n Run Number: {self.run_number}"


class BenchmarkWorker:
    def __init__(self, params):
        self.params = params
        if self.params['backend_type'] == "loki":
            self.client = LokiClient(self.params['host'], self.params['port'], self.params['protocol'])
        if self.params['backend_type'] == "elasticsearch":
            self.client = Elasticsearch(
                f"{self.params['protocol']}://{self.params['host']}:{self.params['port']}",
                basic_auth=(self.params['elastic_username'], self.params['elastic_password']),
                verify_certs=False,
                http_compress=True
            )
        if self.params['backend_type'] == "opensearch":
            self.client = OpenSearch(
                f"{self.params['protocol']}://{self.params['host']}:{self.params['port']}",
                basic_auth=(self.params['opensearch_username'], self.params['opensearch_password']),
                http_compress=True,
                verify_certs=False
            )
        if self.params['backend_type'] == "openobserve":
            self.client = OpenObserveClient(self.params['host'], self.params['port'], self.params['protocol'],
                                            self.params['openobserve_username'], self.params['openobserve_password'],
                                            self.params['openobserve_stream'], self.params['openobserve_org'])
        if self.params['backend_type'] == "signoz":
            self.client = SignozClient(self.params['host'], self.params['port'], self.params['protocol'],
                                       self.params['signoz_username'], self.params['signoz_password'],
                                       self.params['signoz_init_user'])

        self.rand = random.Random(self.params['time_gen_seed'])

    def generate_range(self, min_start, max_end, range=3_600_000_000_000):
        if min_start == 0 | max_end == 0:
            end = time.time_ns()
            start = end - range
            return start, end

        if range == 0:
            return min_start, max_end

        start = self.rand.randint(min_start, max_end - range)
        end = start + range

        return start, end




    def run_loki_benchmark(self, start, end, run_number, query="{job=\"fluentbit\"}"):

        run_start = time.time_ns()

        response = self.client.query_range(query, start, end)

        run_end = time.time_ns()

        run_duration = run_end - run_start

        return BenchmarkInfo(query, run_start, run_end, run_duration, response, run_number, start, end)

    def run_signoz_benchmark(self, start, end, run_number):
        run_start = time.time_ns()

        response = self.client.query_range_v1(start, end)

        run_end = time.time_ns()
        run_duration = run_end - run_start
        return BenchmarkInfo("", run_start, run_end, run_duration, response, run_number, start, end)

    def run_elasticsearch_benchmark(self, start, end, run_number):

        index_name = self.params['elastic_index']

        query = {
            "from": 0,
            "size": 100,
            "query": {
                "range": {
                    "@timestamp": {
                        "gte": start // 1_000_000,
                        "lte": end // 1_000_000,
                    }
                }
            },
        }

        run_start = time.time_ns()

        response = self.client.search(index=index_name, body=query)

        run_end = time.time_ns()
        run_duration = run_end - run_start

        return BenchmarkInfo(query, run_start, run_end, run_duration, dict(response), run_number, start, end)

    def run_opensearch_benchmark(self, start, end, run_number):

        index_name = self.params['opensearch_index']

        query = {
            "from": 0,
            "size": 100,
            "query": {
                "range": {
                    "@timestamp": {
                        "gte": start // 1_000_000,
                        "lte": end // 1_000_000,
                    }
                }
            },
        }

        run_start = time.time_ns()

        response = self.client.search(index=index_name, body=query)

        run_end = time.time_ns()
        run_duration = run_end - run_start

        return BenchmarkInfo(query, run_start, run_end, run_duration, dict(response), run_number, start, end)

    def run_openobserve_benchmark(self, start, end, run_number):
        run_start = time.time_ns()

        response = self.client.search(math.floor(start / 1000), math.floor(end / 1000))

        run_end = time.time_ns()
        run_duration = run_end - run_start

        return BenchmarkInfo("", run_start, run_end, run_duration, response, run_number, start, end)

    def run_benchmark(self, run_number):
        print(f"Starting run {run_number}")

        start, end = self.generate_range(self.params['max_start'], self.params['max_end'], self.params['query_range'])
        try:

            if self.params['backend_type'] == "loki":
                return self.run_loki_benchmark(start, end, run_number)

            if self.params['backend_type'] == "elasticsearch":
                return self.run_elasticsearch_benchmark(start, end, run_number)

            if self.params['backend_type'] == "opensearch":
                return self.run_opensearch_benchmark(start, end, run_number)

            if self.params['backend_type'] == "signoz":
                return self.run_signoz_benchmark(start, end, run_number)

            if self.params['backend_type'] == "openobserve":
                return self.run_openobserve_benchmark(start, end, run_number)

            return None
        except:
            print(f"Error in {run_number}.")
            return BenchmarkInfo("", 0, 0, 0, {"error" : True}, start, end)


def run_module():
    module_args = dict(
        backend_type=dict(type='str', required=True),
        max_start=dict(type='int', required=True),
        max_end=dict(type='int', required=True),
        query_range=dict(type='int', required=False, default=3_600_000_000_000),
        host=dict(type='str', required=True),
        port=dict(type='int', required=True),
        protocol=dict(type='str', required=True),
        workers=dict(type='int', required=True),
        runs_per_worker=dict(type='int', required=True),
        elastic_username=dict(type='str', required=False),
        elastic_password=dict(type='str', required=False),
        elastic_index=dict(type='str', required=False),
        opensearch_username=dict(type='str', required=False),
        opensearch_password=dict(type='str', required=False),
        opensearch_index=dict(type='str', required=False),
        signoz_username=dict(type='str', required=False),
        signoz_password=dict(type='str', required=False),
        signoz_init_user=dict(type='bool', required=False, default=True),
        openobserve_username=dict(type='str', required=False),
        openobserve_password=dict(type='str', required=False),
        openobserve_stream=dict(type='str', required=False),
        openobserve_org=dict(type='str', required=False),
        time_gen_seed=dict(type='int', required=False, default=0)
    )

    result = dict(
        runs=0,
        results=[]
    )

    module = AnsibleModule(
        argument_spec=module_args,
        supports_check_mode=False
    )

    worker = BenchmarkWorker(module.params)

    runs_per_worker = module.params['runs_per_worker']
    workers = module.params['workers']

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []

        run_number = 1

        for _ in range(workers):
            for _ in range(runs_per_worker):
                futures.append(executor.submit(worker.run_benchmark, run_number))
                print(f"Run {run_number} submitted")
                run_number += 1

        for future in concurrent.futures.as_completed(futures):
            try:
                benchmark_result = future.result()  # Retrieve the result of the future
                results.append(benchmark_result.to_dict())
            except:
                print('Error in Query Run')
                results.append({ "unkown_error" : True})

    result['runs'] = len(results)
    result['results'] = results

    module.exit_json(**result)


def main():
    run_module()


if __name__ == '__main__':
    main()
