#!/usr/bin/python

from __future__ import (absolute_import, division, print_function)

__metaclass__ = type

DOCUMENTATION = r'''
module: query_benchmark

short_description: This module is used to analyze the benchmark results.

version_added: "1.0.0"

description: This module is used to analyze the benchmark results.

author:
    - Max Riedel (@m-riedel)
'''

EXAMPLES = r'''
'''

RETURN = r'''
'''

from ansible.module_utils.basic import AnsibleModule
import json
from datetime import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import time

backend_display_name_map = {
    "elastic_default" : "Elasticsearch",
    "opensearch_default" : "OpenSearch",
    "openobserve_default" : "OpenObserve",
    "loki_default" : "Loki",
    "signoz_default" : "SigNoz"
}

backend_to_alias_map = {
    "elastic_default" : "es_benchmark",
    "opensearch_default" : "os_benchmark",
    "openobserve_default" : "openobserve_benchmark",
    "loki_default" : "loki_benchmark",
    "signoz_default" : "signoz_benchmark"
}

class BenchmarkIngestionResult:
    def __init__(self, backend, ingestion_start, ingestion_end, total_lines, run_number=0):
        # 2024-09-02 11:38:23.529313
        date_format = "%Y-%m-%d %H:%M:%S.%f"
        self.backend = backend
        self.ingestion_start = datetime.strptime(ingestion_start, date_format)
        self.ingestion_end = datetime.strptime(ingestion_end, date_format)
        self.total_lines = total_lines
        self.ingestion_duration = self.ingestion_end - self.ingestion_start
        self.run_number = run_number

class FluentBitMetric:
    def __init__(self, timestamp, output, input):
        self.timestamp = timestamp
        self.output = output
        self.input = input

class FluentBitMetrics:
    def __init__(self, metrics, backend, run_number):
        self.metrics = metrics
        self.backend = backend
        self.run_number = run_number

class BenchmarkQueryResult:
    def __init__(self, run_start, run_end, run_duration, query, query_number, query_result):
        self.run_start = run_start
        self.run_end = run_end
        self.run_duration = run_duration
        self.query = query
        self.query_number = query_number
        self.query_result = query_result
        self.run_duration = run_end - run_start


class BenchmarkQueryResults:
    def __init__(self, run_number, query_results, cfg_str, backend, failed=False):
        self.run_number = run_number
        self.query_results = query_results
        self.cfg_str = cfg_str
        self.backend = backend
        self.failed = failed


class BenchmarkBackendResults:
    def __init__(self, backend, ingestion_results, query_results):
        self.backend = backend
        self.ingestion_results = ingestion_results
        self.query_results = query_results


class BenchmarkReader:
    def __init__(self, base_path):
        self.base_path = base_path

    def read_ingestion(self, backend, run_number):
        try:
            with open(f"{self.base_path}/{backend}_{run_number}/benchmark-ingestion-result.json") as f:
                data = json.load(f)
                return BenchmarkIngestionResult(backend, data["benchmark_ingestion_start"], data["benchmark_ingestion_end"],
                                                data["total_lines"], run_number)
        except:
            print(f"Missing file: {backend} {run_number}")
            return None


    def read_ingestion_batch(self, backend, num_runs):
        ingestion_results = []
        for run in range(0, num_runs):
            result = self.read_ingestion(backend, run)
            if result is not None:
                ingestion_results.append(result)
        return ingestion_results

    def read_fluentbitmetrics(self, backend, run_number, alias):
        try:
            with open(f"{self.base_path}/{backend}_{run_number}/fluent-bit-metrics.json") as f:
                data = json.load(f)
                metrics = []
                for metric in data:
                    metrics.append(FluentBitMetric(metric["timestamp"], metric["output"][alias], metric["input"]["tail_apache"]))
                return FluentBitMetrics(metrics, backend, run_number)
        except:
            print(f"Missing file: {backend} {run_number}")
            return FluentBitMetrics([], backend, run_number)

    def read_fluentbitmetrics_batch(self, backend, num_runs, alias):
        metrics = []
        for run in range(0, num_runs):
            metrics.append(self.read_fluentbitmetrics(backend, run, alias))
        return metrics

    def read_query(self, backend, cfg_str, run_number):
        try:
            with open(f"{self.base_path}/{backend}_{run_number}/benchmark-query-result_{cfg_str}.json") as f:
                data = json.load(f)
                query_results = []
                if not data["failed"]:
                    for query in data["results"]:
                        query_results.append(
                            BenchmarkQueryResult(query["run_start"], query["run_end"], query["run_duration"], query["query"],
                                                 query["run_number"], query["response"]))
                return BenchmarkQueryResults(run_number, query_results, cfg_str, backend, data["failed"])
        except:
            print(f"Missing file: {backend} {run_number}")
            return None

    def read_query_batch(self, backend, num_runs, cfg_strs):
        query_results = []
        for run in range(0, num_runs):
            for cfg_str in cfg_strs:
                q = self.read_query(backend, cfg_str, run)
                if q is not None:
                    query_results.append(q)
        return query_results


class Plotter:

    def __init__(self, output_path):
        self.output_path = output_path


    def plot_ingestion_results_duration_bar(self, backend_results):

        backends = list(set(map(lambda x: x.backend, backend_results)))
        avg_results = []
        standard_variances = []
        bottom = np.zeros(len(backends))
        for backend in backends:
            filtered = filter(lambda x: x.backend == backend, backend_results)
            results = list(map(lambda x: x.ingestion_duration.total_seconds() / 60, filtered))

            standard_variances.append(np.std(results))
            avg_results.append(np.mean(results))

        fig, ax = plt.subplots()
        p = ax.bar(list(map(lambda x: backend_display_name_map[x], backends)), avg_results, yerr=standard_variances,ecolor='black', capsize=10, edgecolor="white", linewidth=0.6,
                   bottom=bottom)

        ax.bar_label(p, label_type='center')
        ax.set_title("Durchschnittliche Ingestion-Dauer für 10GB Logdaten pro Backend")
        ax.set_ylabel("Zeit, $t$ [min]")
        ax.set_xlabel("Backend")

        fig.savefig(f"{self.output_path}/ingestion_duration_bar.png")

    def plot_query_results_duration_bars_per_backend(self, backend_results, backends, cfg_strs):

        backend_map = {}
        for backend in backends:
            backend_map[backend] = list(filter(lambda x: x.backend == backend, backend_results))

        for backend, results in backend_map.items():
            self.plot_query_results_duration_bars_per_cfg_str(results, backend, cfg_strs)

    def plot_query_results_duration_bars_per_cfg_str(self, backend_results, backend, cfg_strs):

        cfg_str_map = {}
        for cfg_str in cfg_strs:
            cfg_str_map[cfg_str] = list(filter(lambda x: x.cfg_str == cfg_str, backend_results))

        for cfg_str, results in cfg_str_map.items():
            self.plot_query_results_duration_bar(results, backend, cfg_str)

    def plot_query_results_duration_bar(self, results, backend, cfg_str):
        actual_results = []
        for result in results:
            actual_results.extend(result.query_results)

        values = list(map(lambda x: (x.run_duration / 1_000_000), actual_results))
        n = len(values)
        variance = np.var(values)
        standard_deviation = np.std(values)
        mean = np.mean(values)

        fig, ax = plt.subplots()

        ax.hist(values, bins=50)
        ax.set_title(f"Verteilung der Antwortzeit für {cfg_str} bei {backend_display_name_map[backend]}")

        textstr = '\n'.join((
            f'$n={n}$',
            f'$σ={standard_deviation:.2f}\ ms$',
            f'$μ={mean:.2f}\ ms$'))
        props = dict(pad=4.0, facecolor='none', edgecolor='black')
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props, horizontalalignment='right')
        ax.set_ylabel("Anzahl")
        ax.set_xlabel("Zeit, $t$ [ms]")
        ax.set_ylim(0)
        ax.set_xlim(0)
        fig.savefig(f"{self.output_path}/query_duration_bar_{backend}_{cfg_str}.png")

    def plot_query_duration_query_num_correlation_graph_by_backend(self, backend_results,  backends, cfg_strs, num_runs = 3):
        backend_map = {}
        for backend in backends:
            backend_map[backend] = list(filter(lambda x: x.backend == backend, backend_results))

        for backend, results in backend_map.items():
            self.plot_query_duration_run_num_correlation_graph_by_cfg_str(results, backend, cfg_strs, num_runs)

    def plot_query_duration_run_num_correlation_graph_by_cfg_str(self, results, backend, cfg_strs, num_runs = 3):
        cfg_str_map = {}
        for cfg_str in cfg_strs:
            cfg_str_map[cfg_str] = list(filter(lambda x: x.cfg_str == cfg_str, results))

        for cfg_str, results in cfg_str_map.items():
            self.plot_query_duration_query_num_correlation_graph(results, backend, cfg_str, num_runs)

    def plot_query_duration_query_num_correlation_graph(self, backend_results, backend, cfg_str, num_runs = 3):
        backend_results_actual = []
        for i in range(0, len(backend_results)):
            if not backend_results[i].failed:
                backend_results_actual.append(backend_results[i])
        if len(backend_results_actual) == 0:
            return
        num_runs_actual = len(backend_results_actual)
        results = []
        max_query_num = max(map(lambda x: x.query_number, backend_results_actual[0].query_results))
        for i in range(1, max_query_num + 1):
            results.append([])
            for run in range(0, num_runs_actual):
                sorted_results = sorted(backend_results_actual[run].query_results, key=lambda x: x.run_start)
                results[i - 1].append(sorted_results[i - 1].run_duration / 1_000_000)

        fig, ax = plt.subplots()


        val_x = np.arange(1, max_query_num + 1)


        for i in range(0, num_runs_actual ):
            val_y = list(map(lambda x: x[i], results))

            ax.plot(val_x, val_y, 'o', label=f"Durchlauf {i}", markeredgewidth=0.01)

        ax.plot(val_x, np.mean(results, axis=1), label="Durchschnitt", linewidth=1)
        ax.set_title(f"Verteilung der Antwortzeit nach Startzeit für {cfg_str} bei {backend_display_name_map[backend]}")
        ax.set_xlabel("Anfragenummer (sortiert nach Startzeit)")
        ax.set_ylabel("Zeit, $t$ [ms]")
        ax.set_xlim(0)
        ax.set_ylim(0)
        ax.legend()
        fig.savefig(f"{self.output_path}/query_duration_run_num_{backend}_{cfg_str}.png")



    def plot_query_results_box_by_cfg_str(self, backend_results, backends, cfg_strs):

        cfg_str_map = {}
        for cfg_str in cfg_strs:
            cfg_str_map[cfg_str] = list(filter(lambda x: x.cfg_str == cfg_str, backend_results))


        for cfg_str, results in cfg_str_map.items():
            self.plot_query_results_box(results, backends, cfg_str)

    def plot_query_results_box(self, backend_results, backends, cfg_str):
        data = []
        for backend in backends:
            backend_data = []
            for result in list(filter(lambda x: x.backend == backend, backend_results)):
                backend_data.extend(result.query_results)
            data.append(list(map(lambda x: (x.run_duration / 1_000_000), backend_data)))

        # With Fliers
        fig, ax = plt.subplots()

        ax.set_title(f"Verteilung der Antwortzeit pro Backend bei {cfg_str}")
        ax.boxplot(data, tick_labels=list(map(lambda x: backend_display_name_map[x], backends)))
        ax.set_ylim(0)
        ax.set_ylabel("Zeit, $t$ [ms]")
        ax.set_xlabel("Backend")
        fig.savefig(f"{self.output_path}/query_duration_box_{cfg_str}.png")

        # Without fliers
        fig, ax = plt.subplots()
        ax.set_title(f"Verteilung der Antwortzeit pro Backend bei {cfg_str} ohne Ausreißer")
        ax.boxplot(data, tick_labels=list(map(lambda x: backend_display_name_map[x], backends)), showfliers=False)
        ax.set_ylim(0)
        ax.set_ylabel(r"Zeit, $t$ [ms]")
        ax.set_xlabel("Backend")
        fig.savefig(f"{self.output_path}/query_duration_box_{cfg_str}_no_fliers.png")

    def plot_fluentbit_metrics_per_backend(self, results, backends):

        for backend in backends:
            backend_results = list(filter(lambda x: x.backend == backend,results))
            self.plot_fluentbit_metrics(backend_results, backend)

    def plot_fluentbit_metrics(self, backend_results, backend):
        fig, ax = plt.subplots()
        ax.set_title(f"Verlauf der verarbeiteten Logs über die Zeit bei {backend_display_name_map[backend]}")
        fmt  = ticker.FuncFormatter(lambda x, pos: time.strftime('%M:%S', time.gmtime(x)))
        ax.xaxis.set_major_formatter(fmt)
        ax2 = ax.twinx()
        normal_color = "tab:blue"
        retried_color = "tab:red"
        line_styles = ['solid', 'dashed', 'dotted', 'dashdot']
        ax.tick_params(axis='y', labelcolor=normal_color)
        ax2.tick_params(axis='y', labelcolor=retried_color)
        for results in backend_results:
            sorted_results = results.metrics
            if sorted_results is None or len(sorted_results) == 0:
                continue
            first_timestamp = sorted_results[0].timestamp
            data = []
            relative_timestamps = []
            data_retried = []
            for metric in sorted_results:
                data.append(metric.output["proc_records"] / 1_000_000)
                data_retried.append(metric.output["retried_records"] / 1_000_000)
                relative_timestamps.append((metric.timestamp - first_timestamp))
            ax.plot(relative_timestamps, data, linestyle=line_styles[results.run_number % 4], label=f"Durchlauf {results.run_number}", color=normal_color)
            ax2.plot(relative_timestamps, data_retried, linestyle=line_styles[results.run_number % 4], label=f"Durchlauf {results.run_number}",color=retried_color)

        ax.set_xlim(0)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Anzahl verarbeitete Logs in Mio.")
        ax.set_xlabel(r'Zeit, $t$ [min:sec]')
        ax2.set_ylim(0)
        ax2.set_ylabel("Anzahl erneut versuchter Logs in Mio.")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc=0)
        fig.savefig(f"{self.output_path}/fluentbit_metrics_{backend}.png")

    def plot_fluentbit_metrics_summary(self ,results, backends):
        fig, ax = plt.subplots()
        ax.set_title(f"Verlauf der verarbeiteten Logs über die Zeit pro Backend")
        fmt  = ticker.FuncFormatter(lambda x, pos: time.strftime('%M:%S', time.gmtime(x)))
        ax.xaxis.set_major_formatter(fmt)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        i = 0
        for backend in backends:
            backend_results = list(filter(lambda x: x.backend == backend,results))
            color = colors[i]

            j = 0
            for metric_results in backend_results:
                sorted_results = metric_results.metrics
                if sorted_results is None or len(sorted_results) == 0:
                    continue
                first_timestamp = sorted_results[0].timestamp
                data = []
                relative_timestamps = []
                data_retried = []
                for metric in sorted_results:
                    data.append(metric.output["proc_records"] / 1_000_000)
                    data_retried.append(metric.output["retried_records"] / 1_000_000)
                    relative_timestamps.append((metric.timestamp - first_timestamp))
                ax.plot(relative_timestamps, data, label=f"{backend_display_name_map[backend]}" if j == 0 else "" , color=color)
                j += 1

            i += 1

        ax.set_xlim(0)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Anzahl verarbeitete Logs in Mio.")
        ax.set_xlabel(r'Zeit, $t$ [min:sec]')
        ax.legend()
        fig.savefig(f"{self.output_path}/fluentbit_metrics.png")

def run_module():

    module_args = dict(
        base_path=dict(type='str', required=True),
        num_runs=dict(type='int', required=True),
        cfg_strs=dict(type='list', required=True),
        backends=dict(type='list', required=True),
        output_path=dict(type='str', required=True)
    )

    result = dict()

    module = AnsibleModule(
        argument_spec=module_args,
        supports_check_mode=False
    )

    params = module.params
    reader = BenchmarkReader(params["base_path"])

    ingestion_results = []
    query_results = []
    fluent_bit_metrics = []
    for backend in params["backends"]:
        ingestion_results.extend(reader.read_ingestion_batch(backend, params["num_runs"]))
        query_results.extend(reader.read_query_batch(backend, params["num_runs"], params["cfg_strs"]))
        fluent_bit_metrics.extend(reader.read_fluentbitmetrics_batch(backend, params["num_runs"], backend_to_alias_map[backend]))

    plotter = Plotter(params["output_path"])
    plotter.plot_ingestion_results_duration_bar(ingestion_results)
    plotter.plot_query_results_duration_bars_per_backend(query_results, params["backends"], params["cfg_strs"])
    plotter.plot_query_results_box_by_cfg_str(query_results, params["backends"], params["cfg_strs"])
    plotter.plot_query_duration_query_num_correlation_graph_by_backend(query_results, params["backends"], params["cfg_strs"], params["num_runs"])
    plotter.plot_fluentbit_metrics_per_backend(fluent_bit_metrics,  params["backends"])
    plotter.plot_fluentbit_metrics_summary(fluent_bit_metrics,  params["backends"])
    module.exit_json(**result)


def main():
    run_module()


if __name__ == '__main__':
    main()
