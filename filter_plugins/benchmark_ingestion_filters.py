#!/usr/bin/python
import time
import json
import os

class FilterModule(object):
    def filters(self):
        return {
            'fluentbit_metrics_save_to_file': self.fluentbit_metrics_save_to_file,
            'fluentbit_check_for_stuck_state': self.fluentbit_check_for_stuck_state,
            'test_filter': self.test_filter
        }

    def test_filter(self, str):
        print(os.system('ls /home'))
        return str + ' - TEST WORKED!'



    def fluentbit_metrics_save_to_file(self, metrics, file_path):
        """
        Save metrics of a ansible until loop to a json file in the given path
        :param metrics: the ansible metrics object
        :param file_path: the path to the file where the metrics should be saved
        :return: False, since this will be called in the until condition it cannot satisfy the condition
        """
        # If its the first attempt, create a new file and write the metrics to it.
        # Otherwise, load the file and append the new metrics to it.
        if metrics["attempts"] == 1:
            print("First attempt.")
            result = [metrics["json"]]
            result[0]["timestamp"] = time.time()
            json_file = json.dumps(result, sort_keys=False, indent=2)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as file:
                file.write(json_file)
        else:
            with open(file_path) as file:
                result = json.load(file)
            result.append(metrics["json"])
            result[-1]["timestamp"] = time.time()
            json_file = json.dumps(result, sort_keys=False, indent=2)
            with open(file_path, "w") as file:
                file.write(json_file)

        return False

    def fluentbit_check_for_stuck_state(self, file_path, metric_output_alias):
        """
        Check if the fluentbit metrics have been stuck in a state for a long time.
        :param file_path: the path to the file where the metrics are saved
        :return: True if the metrics have been stuck for more than thirty seconds, False otherwise
        """
        with open(file_path) as file:
            metrics = json.load(file)

        metrics_length = len(metrics)

        last_metric = metrics[metrics_length - 1]
        last_timestamp = last_metric["timestamp"]

        # Let the first 60 seconds pass before checking for a stuck state
        if metrics_length < 60:
            return False

        for i in range(0, metrics_length - 1):
            # The first metric to the 60 seconds before the last metric
            if last_timestamp - metrics[metrics_length - 1 - i]["timestamp"] > 60:
                # Check that the number of processed and retried records are the same
                return (metrics[metrics_length - 1 - i]["output"][metric_output_alias]["proc_records"] == last_metric["output"][metric_output_alias]["proc_records"] and
                        metrics[metrics_length - 1 - i]["output"][metric_output_alias]["retried_records"] == last_metric["output"][metric_output_alias]["retried_records"])
        return False


