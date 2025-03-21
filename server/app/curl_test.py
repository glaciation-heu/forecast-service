from datetime import datetime, timedelta

import requests


def get_input_features():
    current_time = datetime.now()
    start_time = current_time - timedelta(minutes=20)
    end_time = current_time - timedelta(minutes=10)
    print(start_time)
    print(end_time)
    formatted_start_time = start_time.strftime("%Y-%m-%dT%H%%3A%MZ")
    formatted_end_time = end_time.strftime("%Y-%m-%dT%H%%3A%MZ")
    print(formatted_start_time)
    print(formatted_end_time)
    # Define the request url
    url_req = (
        "http://tradeoff.integration/api/v1/clusters/hh/workloads?startTime="
        + formatted_start_time
        + "&endTime="
        + formatted_end_time
    )
    print("requested url: ", url_req)
    # Invoke request and parse json response
    try:
        response = requests.get(url_req)
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
        features_inputs: dict[str, list[float]] = {
            "CPU_Usage(%)": [],
            "Memory_Usage(MB)": [],
            "Disk_Usage(MB)": [],
            "Network_Recv(KB)": [],
            "Power_Consumption(uJ)": [],
        }
        # with open("response.json", "r") as file:
        #   json_response = json.load(file)
        # json_response = json.loads(result)
        print("json response = ", json_response, "\n\n\n")
        print("last ten =", json_response["workloads"][-10:], "\n\n\n")
        las_ten = json_response["workloads"][-10:]
        print(las_ten[0]["resources"], "\n\n\n")
        for item in las_ten:
            for key, val in item["resources"].items():
                if item["resources"][key].get("used") is None:
                    features_inputs[keys_to_features[key]].append(0.0)
                else:
                    features_inputs[keys_to_features[key]].append(
                        float(item["resources"][key]["used"])
                    )

        features_inputs["Power_Consumption(uJ)"] = list(
            map(lambda x: x * 10**9, features_inputs["Power_Consumption(uJ)"])
        )

        """
        if las_ten[0]['resources']['network'].get('used') is None:
            las_ten[0]['resources']['network']['used'] = 0.0
        """
        print("features inputs= ", features_inputs)

        return features_inputs

    except requests.exceptions.RequestException as e:
        print("Request failed: ", e)
        return e


# inputs = get_input_features()

# print('inputs = ', inputs)
