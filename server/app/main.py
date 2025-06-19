from typing import Any, Dict

import base64
import io
import pickle
import time
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Request
from fastapi.openapi.utils import get_openapi
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from contextlib import asynccontextmanager

from app.curl_test import get_input_features
from app.models import LSTM_BNN


class CustomFastAPI(FastAPI):
    def openapi(self) -> Dict[str, Any]:
        if self.openapi_schema:
            return self.openapi_schema
        openapi_schema = get_openapi(
            title="Forecast Service",
            version="0.0.0",
            description="This is a short-term prediction service",
            contact={
                "name": "HIRO-MicroDataCenters",
                "email": "all-hiro@hiro-microdatacenters.nl",
            },
            license_info={
                "name": "MIT",
                "url": "https://github.com/HIRO-MicroDataCenters-BV"
                "/template-python/blob/main/LICENSE",
            },
            routes=self.routes,
        )
        self.openapi_schema = openapi_schema
        return self.openapi_schema


root_path = "app/"
name_of_features = [
    "CPU_Usage(%)",
    "Memory_Usage(MB)",
    "Disk_Usage(MB)",
    "Network_Recv(KB)",
    "Power_Consumption(uJ)",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔄 Loading Models and Scalers ...")
    app.state.assets: dict[str, str] = {}
    for feature in name_of_features:
        output_size = (
            1 if feature in class_exempt_features else output_sizes_of_features[feature]
        )
        # model = LSTM_BNN(1, 32, output_size, 4)
        model = LSTM_BNN(1, 8, output_size, 1, 0.0)
        model_path = root_path + feature + "_model_state_dict.pth"
        model.load_state_dict(
            torch.load(model_path, weights_only=True, map_location="cpu")
        )
        # model = model.to("cuda")
        # print(model)
        model.eval()
        app.state.assets[feature + "_model"] = model

        with open(root_path + "scaler_" + feature + ".pkl", "rb") as f:
            scaler = pickle.load(f)  # Ensure consistency with training

        app.state.assets[feature + "_scaler"] = scaler

    print("✅ Models and Scalers loaded.")
    yield
    print("🔻 Shutting down.")


app = CustomFastAPI(lifespan=lifespan)


Instrumentator().instrument(app).expose(app)


output_sizes_of_features = {
    "nproc": 41,  # class features vary in output size
    "exit_state": 4,
    "other": 1,  # regression features have 1 output size
}

class_exempt_features = [
    "Memory_Usage(MB)",
    "Disk_Usage(MB)",
    "CPU_Usage(%)",
    "Power_Consumption(uJ)",
    "Exec Time",
    "exec_time",
    "time_diff",
    "wait_time",
    "Wait Time",
    "Network_Recv(KB)",
    "AveCPUFreq",
    "ConsumedEnergy",
    "MaxDiskWrite",
]

# fastapp = FastAPI(title='Nproc LSTM-BNN timeseries prediction API',
# description='LSTM-BNN Nprocs prediction model
# using OpenAPI', version='1.0')


def generate_shapley_plots(model, x_inputs, n_steps, name_feature):
    x_inputs = np.array(x_inputs)
    x_inputs = x_inputs.reshape(-1, n_steps)
    # #print('x inputs shape = ', x_inputs.shape)
    # #print('x_inputs = ',x_inputs)
    # #print('model output=',model(torch.tensor(x_inputs, dtype=torch.float32)))

    model.eval()
    with torch.no_grad():
        explainer = shap.KernelExplainer(
            lambda x: model(
                torch.tensor(x.reshape(-1, n_steps, 1), dtype=torch.float32)
            )[0]
            .detach()
            .numpy(),
            x_inputs,
        )
    # x_inputs = x_inputs.reshape(-1, n_steps)
    shap_values = explainer.shap_values(x_inputs)
    shap_values = shap_values.reshape(-1, n_steps)
    fig, ax = plt.subplots(figsize=(8, 6))

    shap.summary_plot(shap_values, x_inputs, plot_type="violin", show=False)

    ax.set_title(name_feature + " Input Features Contribution", fontsize=14)

    # Save the current figure to a BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)  # Close the figure to free memory
    buf.seek(0)
    # Encode the image data in Base64
    encoded = base64.b64encode(buf.read()).decode("utf-8")

    return f"data:image/png;base64,{encoded}"


class InputData(BaseModel):
    xai_graph: bool
    # input_feature: str  # Example "nproc"
    # input: list[Any]  # Example: [20.0, 40.0, 60.0]


@app.get("/")
def read_root():
    return {"message": "Welcome to the LSTM-BNN Model API"}


@app.post(
    "/predict",
    summary="Predict using LSTM-BNN",
    response_description="Prediction result",
)
def predict(request: Request, data: InputData) -> list[dict[str, Any]]:
    try:
        xai_time = 0.0
        start_time = time.time()
        # features_inputs = get_input_features()
        all_workloads = get_input_features()
        input_features_response_time = time.time() - start_time
        # print("features inputs => ", all_workloads)  # features_inputs)

        main_results = list()  #: dict[str, Any] = {}
        if not all_workloads:  # features_inputs:
            return main_results
        xai = data.xai_graph
        for workload in all_workloads:
            partial_results: dict[str, Any] = {
                "worker_node_id": workload["worker_node_id"],
                "CPU_Usage(%)": {"used": {}, "demanded": {}, "allocated": {}},
                "Memory_Usage(MB)": {"used": {}, "demanded": {}, "allocated": {}},
                "Disk_Usage(MB)": {"used": {}, "demanded": {}, "allocated": {}},
                "Network_Recv(KB)": {"used": {}, "demanded": {}, "allocated": {}},
                "Power_Consumption(uJ)": {"used": {}, "demanded": {}, "allocated": {}},
            }
            for name_of_feature, struct_data in workload.items():
                if name_of_feature == "worker_node_id":
                    continue
                # name_of_feature = data.input_feature
                # Load model
                # model_path = root_path + name_of_feature + "_model_state_dict.pth"
                # output_size = (
                #    1
                #    if name_of_feature in class_exempt_features
                #    else output_sizes_of_features[name_of_feature]
                # )
                # model = LSTM_BNN(1, 32, output_size, 4)
                # model.load_state_dict(
                #    torch.load(model_path, weights_only=True, map_location="cpu")
                # )
                # model = model.to("cuda")
                # print(model)
                # model.eval()
                model = request.app.state.assets[name_of_feature + "_model"]
                scaler = request.app.state.assets[name_of_feature + "_scaler"]

                for metric, input_data in struct_data.items():
                    # Get input data
                    # input_data = data.input
                    # print("input_data=", input_data)

                    results: dict[str, Any] = {
                        "Timestamp": [],
                        "predictions": [],
                        "uncertainties": [],
                        "Explanation_plot": Any,
                    }

                    timestamp = datetime.now()
                    list_of_input_data = []
                    for i in range(10):
                        if name_of_feature not in class_exempt_features:
                            # Load LabelEncoder
                            with open(
                                root_path + "label_encoder_" + name_of_feature + ".pkl",
                                "rb",
                            ) as f:
                                label_encoder = pickle.load(
                                    f
                                )  # Ensure consistency with training

                            # Convert categorical input to numerical using LabelEncoder
                            encoded_input = label_encoder.transform(input_data)
                            # print("encoded_input=", encoded_input)
                            encoded_input = encoded_input.reshape(1, len(input_data), 1)
                            # Convert to PyTorch tensor
                            input_tensor = torch.tensor(
                                encoded_input, dtype=torch.float32
                            )  # .unsqueeze(0)  # Add batch dimension
                            # print("input_tensor=", input_tensor)

                        else:  # if a feature is regression feature...
                            input_data = np.array(input_data)
                            # Load Scaler for normalization
                            # with open(
                            #    root_path + "scaler_" + name_of_feature + ".pkl", "rb"
                            # ) as f:
                            #    scaler = pickle.load(
                            #        f
                            #    )  # Ensure consistency with training

                            input_data_scaled = scaler.transform(
                                input_data.reshape(-1, 1)
                            )

                            input_data_scaled = input_data_scaled.reshape(
                                1, len(input_data), 1
                            )
                            input_tensor = torch.tensor(
                                input_data_scaled, dtype=torch.float32
                            )
                            # print("input_tensor=", input_tensor)

                        list_of_input_data.append(input_tensor.numpy())
                        with torch.no_grad():
                            # #print('model output = ',model(input_tensor))
                            output, uncertainty = model(input_tensor)
                            # print("output=", output)
                            # print("uncertainty=", uncertainty)

                            output = output.cpu().detach().numpy()
                            uncertainty = uncertainty.cpu().detach().numpy()

                        if name_of_feature not in class_exempt_features:
                            # Apply softmax to get probabilities
                            probabilities = F.softmax(
                                torch.Tensor(output), dim=-1
                            )  # .tolist()[0]
                            # prob_uncert = F.softmax(torch.Tensor(uncertainty),
                            # dim=-1).tolist()[0]
                            # print("probabilities=", probabilities)
                            # #print('prob_uncert=',prob_uncert)
                            # Get predicted class index
                            predicted_index = torch.argmax(
                                probabilities, dim=1
                            )  # .item())
                            # pred_unce_index = torch.argmax(prob_uncert,
                            # dim=1)#.item())
                            # print("predicted_index=", predicted_index)
                            # #print('pred_unce_index=',pred_unce_index)
                            # Decode the predicted index back to original label
                            predicted_label = label_encoder.inverse_transform(
                                predicted_index.numpy().ravel()
                            )
                            # pred_unce_label = label_encoder.inverse_transform(
                            # pred_unce_index.numpy().ravel())
                            # print("predicted_label=", predicted_label)
                            # print("uncertainty=", uncertainty)
                            # #print('pred_unce_label=', pred_unce_label)
                            prediction = predicted_label[0]
                            uncertaintii = uncertainty[0]
                            # return {"prediction": predicted_label[0],
                            # "uncertainty": pred_unce_label[0]}

                        else:
                            output_unscaled = scaler.inverse_transform(output)
                            # uncertainty_unscaled = uncertainty * (
                            #    scaler.data_max_ - scaler.data_min_
                            # )
                            uncertainty_unscaled = uncertainty * (
                                scaler.center_ - scaler.scale_
                            )
                            # print("output=", output_unscaled[0][0])
                            # print("uncertainty=", uncertainty_unscaled[0][0])
                            prediction = output_unscaled[0][0]
                            uncertaintii = uncertainty_unscaled[0][0]
                            # return {"prediction": str(output_unscaled[0][0]),
                            # "uncertainty": str(uncertainty_unscaled[0][0])}

                        results["Timestamp"].append(str(timestamp))
                        results["predictions"].append(str(prediction))
                        results["uncertainties"].append(str(uncertaintii))

                        timestamp += timedelta(seconds=6)
                        input_data = np.roll(input_data, -1)
                        input_data[-1] = prediction

                    start_time_xai = time.time()
                    results["Explanation_plot"] = (
                        generate_shapley_plots(
                            model, list_of_input_data, len(input_data), name_of_feature
                        )
                        if xai is True
                        else None
                    )
                    end_time_xai = time.time()
                    xai_time += end_time_xai - start_time_xai

                    # print(
                    #    name_of_feature,
                    #    " ",
                    #    metric,
                    #    " results\n",
                    #    pd.DataFrame(results),
                    # )

                    partial_results[name_of_feature][metric] = results

            main_results.append(partial_results)  # [name_of_feature] = results

        exec_time = time.time() - start_time
        cpu_time = exec_time - input_features_response_time - xai_time
        print("execution time = ", exec_time, " seconds")
        print("request response time = ", input_features_response_time, " seconds")
        print("XAI graphs generation time = ", xai_time, " seconds")
        print("cpu time = ", cpu_time, " seconds")
        print(
            "sum of cpu/xai/req times = ",
            cpu_time + xai_time + input_features_response_time,
            " seconds",
        )
        return main_results

    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
