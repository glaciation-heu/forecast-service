from datetime import datetime, timedelta
from typing import Any

import pytz
import requests

# import json
import time


def get_input_features():
    current_time = datetime.now()

    start_time = current_time - timedelta(minutes=20)
    end_time = current_time - timedelta(minutes=10)

    # print(start_time)
    # print(end_time)

    formatted_start_time = start_time.astimezone(pytz.utc)
    formatted_start_time_str = formatted_start_time.strftime("%Y-%m-%dT%H%%3A%MZ")
    formatted_end_time = end_time.astimezone(pytz.utc)
    formatted_end_time_str = formatted_end_time.strftime("%Y-%m-%dT%H%%3A%MZ")

    # print(formatted_start_time)
    # print(formatted_end_time)
    # Define the request url
    url_req = (
        "http://tradeoff.integration/api/v1/clusters/hh/workloads?startTime="
        # "http://trade-off-service/api/v1/clusters/hh/workloads?startTime="
        + formatted_start_time_str
        + "&endTime="
        + formatted_end_time_str
    )
    # print("requested url: ", url_req)
    # Invoke request and parse json response
    try:
        start_req_time = time.time()
        response = requests.get(url_req)
        end_req_time = time.time()
        print("URL request time = ", end_req_time - start_req_time, " seconds")
        json_response = None
        if response.status_code == 200:
            # print("response OK")
            json_response = response.json()

        keys_to_features: dict[str, str] = {
            "cpu": "CPU_Usage(%)",
            "memory": "Memory_Usage(MB)",
            "storage": "Disk_Usage(MB)",
            "network": "Network_Recv(KB)",
            "energy": "Power_Consumption(uJ)",
        }

        # with open("app/response.json", "r") as file:
        #    json_response = json.load(file)
        # json_response = result  # json.loads(result)
        # print("json response = ", json_response, "\n\n\n")
        # print("last ten =", json_response["workloads"][-10:], "\n\n\n")
        # las_ten = json_response["workloads"][-10:]
        if not json_response["workloads"]:
            return []
        # print(las_ten[0]["resources"], "\n\n\n")
        list_of_worker_node_ids = []
        for workload_item in json_response["workloads"]:
            id_string = workload_item["runs_on"]["worker_node_id"]
            if id_string not in list_of_worker_node_ids:
                list_of_worker_node_ids.append(id_string)

        # print(list_of_worker_node_ids)
        # print(len(list_of_worker_node_ids))

        list_of_all_workloads = []
        for worker_node_id in list_of_worker_node_ids:
            features_inputs: dict[str, Any] = {
                "worker_node_id": None,
                "CPU_Usage(%)": {"used": [], "demanded": [], "allocated": []},
                "Memory_Usage(MB)": {"used": [], "demanded": [], "allocated": []},
                "Disk_Usage(MB)": {"used": [], "demanded": [], "allocated": []},
                "Network_Recv(KB)": {"used": [], "demanded": [], "allocated": []},
                "Power_Consumption(uJ)": {"used": [], "demanded": [], "allocated": []},
            }
            for workload_item in json_response["workloads"]:
                if worker_node_id == workload_item["runs_on"]["worker_node_id"]:
                    features_inputs["worker_node_id"] = worker_node_id
                    for key, val in workload_item["resources"].items():
                        for metric in ["used", "demanded", "allocated"]:
                            if workload_item["resources"][key].get(metric) is None:
                                features_inputs[keys_to_features[key]][metric].append(
                                    0.0
                                )
                            else:
                                features_inputs[keys_to_features[key]][metric].append(
                                    float(workload_item["resources"][key][metric])
                                )

            for key, val in features_inputs.items():
                if key != "worker_node_id":
                    for metric in ["used", "demanded", "allocated"]:
                        features_inputs[key][metric] = (
                            val[metric] + [0.0] * (10 - len(val[metric]))
                        )[-10:]
                        if key == "Power_Consumption(uJ)":
                            features_inputs[key][metric] = list(
                                map(lambda x: x * 10**9, features_inputs[key][metric])
                            )

            list_of_all_workloads.append(features_inputs)

        # print(list_of_all_workloads)
        # print(len(list_of_all_workloads))

        # print("list of all workloads = ", list_of_all_workloads)

        return list_of_all_workloads

    except requests.exceptions.RequestException as e:
        # print("Request failed: ", e)
        return e


# inputs = get_input_features()

# #print('inputs = ', inputs)
