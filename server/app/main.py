from typing import Any, Dict

import base64
import io
import pickle
from datetime import datetime, timedelta
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.openapi.utils import get_openapi
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from app.curl_test import get_input_features
from app.models import LSTM_BNN

from . import example, items


class CustomFastAPI(FastAPI):
    def openapi(self) -> Dict[str, Any]:
        if self.openapi_schema:
            return self.openapi_schema
        openapi_schema = get_openapi(
            title="Template web service",
            version="0.0.0",
            description="This is a template of a web service",
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


app = CustomFastAPI()


Instrumentator().instrument(app).expose(app)


app.include_router(example.router)
app.include_router(items.routes.router)


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

# fastapp = FastAPI(title='Nproc LSTM-BNN timeseries prediction API', description='LSTM-BNN Nprocs prediction model using OpenAPI', version='1.0')
root_path = "app/"


def generate_shapley_plots(model, x_inputs, n_steps, name_feature):
    x_inputs = np.array(x_inputs)
    x_inputs = x_inputs.reshape(-1, n_steps)
    # print('x inputs shape = ', x_inputs.shape)
    # print('x_inputs = ',x_inputs)
    # print('model output=',model(torch.tensor(x_inputs, dtype=torch.float32)))

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

    shap.summary_plot(shap_values, x_inputs, plot_type="bar", show=False)

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
def predict(data: InputData):
    try:
        start_time = time.time()
        features_inputs = get_input_features()
        print(features_inputs)
        main_results = {}
        if not features_inputs:
            return main_results
        xai = data.xai_graph
        for name_of_feature, input_data in features_inputs.items():
            # name_of_feature = data.input_feature
            # Load model
            model_path = root_path + name_of_feature + "_model_state_dict.pth"
            output_size = (
                1
                if name_of_feature in class_exempt_features
                else output_sizes_of_features[name_of_feature]
            )
            model = LSTM_BNN(1, 32, output_size, 4)
            model.load_state_dict(
                torch.load(model_path, weights_only=True, map_location="cpu")
            )
            # model = model.to("cuda")
            print(model)
            model.eval()

            # Get input data
            # input_data = data.input
            print("input_data=", input_data)

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
                        root_path + "label_encoder_" + name_of_feature + ".pkl", "rb"
                    ) as f:
                        label_encoder = pickle.load(
                            f
                        )  # Ensure consistency with training

                    # Convert categorical input to numerical using LabelEncoder
                    encoded_input = label_encoder.transform(input_data)
                    print("encoded_input=", encoded_input)
                    encoded_input = encoded_input.reshape(1, len(input_data), 1)
                    # Convert to PyTorch tensor
                    input_tensor = torch.tensor(
                        encoded_input, dtype=torch.float32
                    )  # .unsqueeze(0)  # Add batch dimension
                    print("input_tensor=", input_tensor)

                else:  # if a feature is regression feature...
                    input_data = np.array(input_data)
                    # Load Scaler for normalization
                    with open(
                        root_path + "scaler_" + name_of_feature + ".pkl", "rb"
                    ) as f:
                        scaler = pickle.load(f)  # Ensure consistency with training
                        input_data_scaled = scaler.transform(input_data.reshape(-1, 1))

                    input_data_scaled = input_data_scaled.reshape(1, len(input_data), 1)
                    input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)
                    print("input_tensor=", input_tensor)

                list_of_input_data.append(input_tensor.numpy())
                with torch.no_grad():
                    # print('model output = ',model(input_tensor))
                    output, uncertainty = model(input_tensor)
                    print("output=", output)
                    print("uncertainty=", uncertainty)

                    output = output.cpu().detach().numpy()
                    uncertainty = uncertainty.cpu().detach().numpy()

                if name_of_feature not in class_exempt_features:
                    # Apply softmax to get probabilities
                    probabilities = F.softmax(
                        torch.Tensor(output), dim=-1
                    )  # .tolist()[0]
                    # prob_uncert = F.softmax(torch.Tensor(uncertainty), dim=-1)#.tolist()[0]
                    print("probabilities=", probabilities)
                    # print('prob_uncert=',prob_uncert)
                    # Get predicted class index
                    predicted_index = torch.argmax(probabilities, dim=1)  # .item())
                    # pred_unce_index = torch.argmax(prob_uncert, dim=1)#.item())
                    print("predicted_index=", predicted_index)
                    # print('pred_unce_index=',pred_unce_index)
                    # Decode the predicted index back to original label
                    predicted_label = label_encoder.inverse_transform(
                        predicted_index.numpy().ravel()
                    )
                    # pred_unce_label = label_encoder.inverse_transform(pred_unce_index.numpy().ravel())
                    print("predicted_label=", predicted_label)
                    print("uncertainty=", uncertainty)
                    # print('pred_unce_label=', pred_unce_label)
                    prediction = predicted_label[0]
                    uncertaintii = uncertainty[0]
                    # return {"prediction": predicted_label[0], "uncertainty": pred_unce_label[0]}

                else:
                    output_unscaled = scaler.inverse_transform(output)
                    uncertainty_unscaled = uncertainty * (
                        scaler.data_max_ - scaler.data_min_
                    )
                    print("output=", output_unscaled[0][0])
                    print("uncertainty=", uncertainty_unscaled[0][0])
                    prediction = output_unscaled[0][0]
                    uncertaintii = uncertainty_unscaled[0][0]
                    # return {"prediction": str(output_unscaled[0][0]), "uncertainty": str(uncertainty_unscaled[0][0])}

                results["Timestamp"].append(str(timestamp))
                results["predictions"].append(str(prediction))
                results["uncertainties"].append(str(uncertaintii))

                timestamp += timedelta(seconds=6)
                input_data = np.roll(input_data, -1)
                input_data[-1] = prediction

            results["Explanation_plot"] = (
                generate_shapley_plots(
                    model, list_of_input_data, len(input_data), name_of_feature
                )
                if xai is True
                else None
            )

            print(pd.DataFrame(results))

            main_results[name_of_feature] = results

        end_time = time.time()
        print("execution time = ", end_time - start_time, " seconds")

        return main_results

    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
