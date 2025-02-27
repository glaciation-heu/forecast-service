import json
import subprocess
from datetime import datetime, timedelta


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
    # Define the curl command
    curl_command = [
        "curl",
        "-s",
        "-X",
        "GET",
        "http://tradeoff.integration/api/v1/clusters/hh/workloads?startTime="
        + formatted_start_time
        + "&endTime="
        + formatted_end_time,
        "-H",
        "Accept: application/json",
    ]
    print("curl command=", curl_command)
    # Execute the command and capture the output
    result = subprocess.check_output(curl_command)  # , capture_output=True, text=True)
    print("result =", result, "\n\n\n")

    # Parse the JSON response
    try:
        # json_response = json.loads(result)
        keys_to_features = {
            "cpu": "CPU_Usage(%)",
            "memory": "Memory_Usage(MB)",
            "storage": "Disk_Usage(MB)",
            "network": "Network_Recv(KB)",
            "energy": "Power_Consumption(uJ)",
        }
        features_inputs = {
            "CPU_Usage(%)": [],
            "Memory_Usage(MB)": [],
            "Disk_Usage(MB)": [],
            "Network_Recv(KB)": [],
            "Power_Consumption(uJ)": [],
        }
        # with open("response.json", "r") as file:
        #   json_response = json.load(file)
        json_response = json.loads(result)
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

    except json.JSONDecodeError:
        print("Failed to parse JSON")
        return "Failed to parse JSON"


# inputs = get_input_features()

# print('inputs = ', inputs)
