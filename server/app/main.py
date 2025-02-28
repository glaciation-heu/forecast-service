from typing import Any, Dict

import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.openapi.utils import get_openapi
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from curl_test import get_input_features
from models import LSTM_BNN

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


class InputData(BaseModel):
    input_feature: str  # Example "nproc"
    input: list[Any]  # Example: [20.0, 40.0, 60.0]


@app.get("/")
def read_root():
    return {"message": "Welcome to the LSTM-BNN Model API"}


@app.get(
    "/predict",
    summary="Predict using LSTM-BNN",
    response_description="Prediction result",
)
def predict():  # data: InputData):
    try:
        features_inputs = get_input_features()
        main_results = {}
        for name_of_feature, input_data in features_inputs.items():
            # name_of_feature = data.input_feature

            # Load model
            model_path = name_of_feature + "_model_state_dict.pth"
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

            results: dict[str, list[float]] = {"Timestamp": [], "predictions": [], "uncertainties": []}

            timestamp = datetime.now()

            for i in range(10):
                if name_of_feature not in class_exempt_features:
                    # Load LabelEncoder
                    with open("label_encoder_" + name_of_feature + ".pkl", "rb") as f:
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
                    with open("scaler_" + name_of_feature + ".pkl", "rb") as f:
                        scaler = pickle.load(f)  # Ensure consistency with training
                        input_data_scaled = scaler.transform(input_data.reshape(-1, 1))

                    input_data_scaled = input_data_scaled.reshape(1, len(input_data), 1)
                    input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)
                    print("input_tensor=", input_tensor)

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

            print(pd.DataFrame(results))

            main_results[name_of_feature] = results

        return main_results

    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
