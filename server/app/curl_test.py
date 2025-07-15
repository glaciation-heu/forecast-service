from typing import Any

# import json
import time
from datetime import datetime, timedelta

import pytz
import requests

# end_time = datetime(1,1,1)


def get_input_features(simulate):
    # current_time = datetime.now()

    # global end_time
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=5)
    end_time -= timedelta(minutes=0)
    # start_time = current_time - timedelta(minutes=20)
    # end_time = current_time - timedelta(minutes=10)

    # print('local start time: ', start_time)
    # print('local end time: ', end_time)

    formatted_start_time = start_time.astimezone(pytz.utc)
    # print('server start time: ', formatted_start_time)
    formatted_start_time_str = formatted_start_time.strftime("%Y-%m-%dT%H%%3A%MZ")
    # print('server start time str: ', formatted_start_time_str)
    formatted_end_time = end_time.astimezone(pytz.utc)
    # print('server end time: ', formatted_end_time)
    formatted_end_time_str = formatted_end_time.strftime("%Y-%m-%dT%H%%3A%MZ")
    # print('server end time str: ', formatted_end_time_str)
    end_time = formatted_end_time
    # print(formatted_start_time)
    # print(formatted_end_time)
    # Define the request url
    url_req = (
        (
            # "http://tradeoff.validation/api/v1/clusters/hh/wnode?startTime="
            # http://tradeoff.integration/api/v1/clusters/hh/workloads?startTime="
            "http://trade-off-service/api/v1/clusters/hh/workloads?startTime="
            + formatted_start_time_str
            + "&endTime="
            + formatted_end_time_str
        )
        if simulate is False
        else "https://entire.insight-centre.org/simulator-ws/simulate"
    )

    print("requested url: ", url_req)
    # Invoke request and parse json response
    tradeoff_service_response_time = 0.0
    try:
        start_req_time = time.time()
        if not simulate:
            response = requests.get(url_req)
        else:
            headers = {"Content-Type": "application/json"}
            with open("app/requests.json", "rb") as f:
                data = f.read()
            response = requests.post(url_req, data=data, headers=headers, verify=False)

        end_req_time = time.time()
        tradeoff_service_response_time = end_req_time - start_req_time
        print("URL request time = ", end_req_time - start_req_time, " seconds")
        json_response = None
        if response.status_code == 200:
            print("response OK")
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
        if json_response is None:
            return []

        # print("json_response[workloads] = ", json_response)

        # with open('data.json', 'w', encoding='utf-8') as f:
        #    json.dump(json_response, f, ensure_ascii=False, indent=4)

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
                "CPU_Usage(%)": [],
                # {"used": [], "demanded": [], "allocated": []},
                "Memory_Usage(MB)": [],
                # {"used": [], "demanded": [], "allocated": []},
                "Disk_Usage(MB)": [],
                # {"used": [], "demanded": [], "allocated": []},
                "Network_Recv(KB)": [],
                # {"used": [], "demanded": [], "allocated": []},
                "Power_Consumption(uJ)": [],
                # {"used": [], "demanded": [], "allocated": []},
            }
            for workload_item in json_response["workloads"]:
                if worker_node_id == workload_item["runs_on"]["worker_node_id"]:
                    features_inputs["worker_node_id"] = worker_node_id
                    for key, val in workload_item["resources"].items():
                        # for metric in ["used", "demanded", "allocated"]:
                        if workload_item["resources"][key].get("used") is None:
                            if simulate is True and key != "energy":
                                features_inputs[keys_to_features[key]].append(
                                    float(workload_item["resources"][key]["allocated"])
                                )
                            else:
                                features_inputs[keys_to_features[key]].append(0.0)
                        else:
                            features_inputs[keys_to_features[key]].append(
                                float(workload_item["resources"][key]["used"])
                            )

                            # = features_inputs[key]["allocated"]
                            # features_inputs[key]["demanded"] = features_inputs[key][
                            #    "allocated"
                            # ]
            for key, val in features_inputs.items():
                if key != "worker_node_id":
                    # for metric in ["used", "demanded", "allocated"]:
                    features_inputs[key] = (val + [0.0] * (10 - len(val)))[-10:]
                    if key == "Power_Consumption(uJ)":
                        features_inputs[key] = list(
                            map(lambda x: x * 10**9, features_inputs[key])
                        )

            list_of_all_workloads.append(features_inputs)

        # print(list_of_all_workloads)
        # print(len(list_of_all_workloads))

        # print("list of all workloads = ", list_of_all_workloads)

        return list_of_all_workloads, tradeoff_service_response_time, end_time

    except requests.exceptions.RequestException as e:
        print("Request failed: ", e)
        return e, tradeoff_service_response_time, end_time


# inputs = get_input_features()

# #print('inputs = ', inputs)
