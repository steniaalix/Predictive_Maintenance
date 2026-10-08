# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel,Field
# from typing import Literal
# import joblib
# import pandas as pd


# app=FastAPI(title="Cost-Aware  Failure Prediction API",
#             description=("An ML API that predicts  Failure using a Random Forest model and a cost-optimized threshold"),
#             version="1.0.0")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# artifact_twf=joblib.load("app/twf_model.joblib")
# model_twf=artifact_twf["model"]
# encoder_twf=artifact_twf['encoder']
# threshold_twf=artifact_twf['threshold']
# feature_order_twf=artifact_twf['feature_order']

# artifact_hdf=joblib.load("app/hdf_model.joblib")
# model_hdf=artifact_hdf["model"]
# encoder_hdf=artifact_hdf['encoder']
# threshold_hdf=artifact_hdf['threshold']
# feature_order_hdf=artifact_hdf['feature_order']

# artifact_osf=joblib.load("app/osf_model.joblib")
# model_osf=artifact_osf["model"]
# encoder_osf=artifact_osf['encoder']
# threshold_osf=artifact_osf['threshold']
# feature_order_osf=artifact_osf['feature_order']

# artifact_pwf=joblib.load("app/pwf_model.joblib")
# model_pwf=artifact_pwf["model"]
# encoder_pwf=artifact_pwf['encoder']
# threshold_pwf=artifact_pwf['threshold']
# feature_order_pwf=artifact_pwf['feature_order']



# FN_COST=15000
# FP_COST=500


# class input_template(BaseModel):
#     type:Literal["L","M","H"]=Field(
#         description="Machine Quality Type"
#     )

#     air_temperature:float=Field(
#         gt=0,
#         description="Air temperature in Kelvin"
#     )

#     process_temperature:float=Field(
#         gt=0,
#         description="Process temperature in Kelvin"
#     )

#     rotational_speed:int=Field(
#         gt=0,
#         description="Rotational speed in RPM"
#     )

#     torque:float=Field(
#         gt=0,
#         description="Torque in Nm"
#     )

#     tool_wear:float=Field(
#         ge=0,
#         description="Tool wear in minutes"
#     )


# class response_template(BaseModel):
#     machine_type: str

#     prediction: int

#     result: str

#     risk_level: str

#     failure_probability: float

#     optimized_threshold: float

#     default_threshold: float

#     default_prediction: int

#     power: float

#     cost_strategy: str


# @app.get("/")

# def home():
#     return {
#         "message":"TWF prediction API",
#         "status":"running",
#     }

# @app.get("/health")

# def health_check():

#     return {
#         "status":"healthy",
#         "model_loaded":True,
#     }





# @app.post("/predict/twf", response_model=response_template)
# def predict_twf(data:input_template):

#     try:
#         encoded_type=encoder_twf.transform([data.type])[0]
#         power=data.torque*data.rotational_speed
#         input_df=pd.DataFrame([{
#             "Type":encoded_type,
#             "Air temperature [K]":data.air_temperature,
#             "Process temperature [K]":data.process_temperature,
#             "Rotational speed [rpm]":data.rotational_speed,
#             "Torque [Nm]":data.torque,
#             "Tool wear [min]":data.tool_wear,
#             "Power":power
#         }])

#         input_df=input_df[feature_order_twf]

#         prob=float(model_twf.predict_proba(input_df)[0][1])

#         default_prediction=int(prob>=0.5)

#         prediction=int(prob>=threshold_twf)


#         if prediction==1:
#             result="Tool Wear Failure predicted"
#         else:
#             result="No Tool Wear Failure predicted"

#         if prob<threshold_twf:

#             risk_level="LOW"

#         elif prob<0.5:
#             risk_level="MEDIUM"

#         else:
#             risk_level="HIGH"



#         return response_template(
#             machine_type=data.type,
#             prediction=prediction,
#             result=result,
#             risk_level=risk_level,
#             failure_probability=round(prob,4),
#             optimized_threshold=round(threshold_twf,4),
#             default_threshold=0.5,
#             default_prediction=default_prediction,
#             power=round(power,2),
#             cost_strategy=(
#                 f"FN Cost= ${FN_COST:,},"
#                 f"FP Cost= ${FP_COST:,}"
#             )
#         )
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=str(e)
#         )



# @app.post("/predict/hdf", response_model=response_template)
# def predict_twf(data:input_template):

#     try:
#         encoded_type=encoder_hdf.transform([data.type])[0]
#         power=data.torque*data.rotational_speed
#         input_df=pd.DataFrame([{
#             "Type":encoded_type,
#             "Air temperature [K]":data.air_temperature,
#             "Process temperature [K]":data.process_temperature,
#             "Rotational speed [rpm]":data.rotational_speed,
#             "Torque [Nm]":data.torque,
#             "Tool wear [min]":data.tool_wear,
#             "Power":power
#         }])

#         input_df=input_df[feature_order_hdf]

#         prob=float(model_hdf.predict_proba(input_df)[0][1])

#         default_prediction=int(prob>=0.5)

#         prediction=int(prob>=threshold_hdf)


#         if prediction==1:
#             result="Heat Dissipation Failure predicted"
#         else:
#             result="No Heat Dissipation Failure predicted"

#         if prob<threshold_hdf:

#             risk_level="LOW"

#         elif prob<0.5:
#             risk_level="MEDIUM"

#         else:
#             risk_level="HIGH"



#         return response_template(
#             machine_type=data.type,
#             prediction=prediction,
#             result=result,
#             risk_level=risk_level,
#             failure_probability=round(prob,4),
#             optimized_threshold=round(threshold_hdf,4),
#             default_threshold=0.5,
#             default_prediction=default_prediction,
#             power=round(power,2),
#             cost_strategy=(
#                 f"FN Cost= ${FN_COST:,},"
#                 f"FP Cost= ${FP_COST:,}"
#             )
#         )
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=str(e)
#         )





# @app.post("/predict/pwf", response_model=response_template)
# def predict_twf(data:input_template):

#     try:
#         encoded_type=encoder_pwf.transform([data.type])[0]
#         power=data.torque*data.rotational_speed
#         input_df=pd.DataFrame([{
#             "Type":encoded_type,
#             "Air temperature [K]":data.air_temperature,
#             "Process temperature [K]":data.process_temperature,
#             "Rotational speed [rpm]":data.rotational_speed,
#             "Torque [Nm]":data.torque,
#             "Tool wear [min]":data.tool_wear,
#             "Power":power
#         }])

#         input_df=input_df[feature_order_pwf]

#         prob=float(model_pwf.predict_proba(input_df)[0][1])

#         default_prediction=int(prob>=0.5)

#         prediction=int(prob>=threshold_pwf)


#         if prediction==1:
#             result="Power Failure predicted"
#         else:
#             result="No  Power Failure predicted"

#         if prob<threshold_pwf:

#             risk_level="LOW"

#         elif prob<0.5:
#             risk_level="MEDIUM"

#         else:
#             risk_level="HIGH"



#         return response_template(
#             machine_type=data.type,
#             prediction=prediction,
#             result=result,
#             risk_level=risk_level,
#             failure_probability=round(prob,4),
#             optimized_threshold=round(threshold_pwf,4),
#             default_threshold=0.5,
#             default_prediction=default_prediction,
#             power=round(power,2),
#             cost_strategy=(
#                 f"FN Cost= ${FN_COST:,},"
#                 f"FP Cost= ${FP_COST:,}"
#             )
#         )
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=str(e)
#         )




# @app.post("/predict/osf", response_model=response_template)
# def predict_twf(data:input_template):

#     try:
#         encoded_type=encoder_osf.transform([data.type])[0]
#         power=data.torque*data.rotational_speed
#         input_df=pd.DataFrame([{
#             "Type":encoded_type,
#             "Air temperature [K]":data.air_temperature,
#             "Process temperature [K]":data.process_temperature,
#             "Rotational speed [rpm]":data.rotational_speed,
#             "Torque [Nm]":data.torque,
#             "Tool wear [min]":data.tool_wear,
#             "Power":power
#         }])

#         input_df=input_df[feature_order_osf]

#         prob=float(model_osf.predict_proba(input_df)[0][1])

#         default_prediction=int(prob>=0.5)

#         prediction=int(prob>=threshold_osf)


#         if prediction==1:
#             result="Over Strain Failure predicted"
#         else:
#             result="No Over strain Failure predicted"

#         if prob<threshold_osf:

#             risk_level="LOW"

#         elif prob<0.5:
#             risk_level="MEDIUM"

#         else:
#             risk_level="HIGH"



#         return response_template(
#             machine_type=data.type,
#             prediction=prediction,
#             result=result,
#             risk_level=risk_level,
#             failure_probability=round(prob,4),
#             optimized_threshold=round(threshold_osf,4),
#             default_threshold=0.5,
#             default_prediction=default_prediction,
#             power=round(power,2),
#             cost_strategy=(
#                 f"FN Cost= ${FN_COST:,},"
#                 f"FP Cost= ${FP_COST:,}"
#             )
#         )
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=str(e)
#         )


# @app.post("/predict/all")
# def predict_all(data: input_template):

#     power = data.torque * data.rotational_speed

    
#     type_encoded = encoder_twf.transform([data.type])[0]

#     input_data = pd.DataFrame(
#         [[
#             type_encoded,
#             data.air_temperature,
#             data.process_temperature,
#             data.rotational_speed,
#             data.torque,
#             data.tool_wear,
#             power
#         ]],
#         columns=feature_order_twf
#     )


#     twf_probability = model_twf.predict_proba(
#         input_data
#     )[0][1]

#     twf_prediction = int(
#         twf_probability >= threshold_twf
#     )


#     hdf_probability = model_hdf.predict_proba(
#         input_data
#     )[0][1]

#     hdf_prediction = int(
#         hdf_probability >=threshold_hdf
#     )


#     pwf_probability =model_pwf.predict_proba(
#         input_data
#     )[0][1]

#     pwf_prediction = int(
#         pwf_probability >=threshold_pwf
#     )


#     osf_probability = model_osf.predict_proba(
#         input_data
#     )[0][1]

#     osf_prediction = int(
#         osf_probability >= threshold_osf
#     )

    

#     failures_detected = (
#         twf_prediction
#         + hdf_prediction
#         + pwf_prediction
#         + osf_prediction
#     )

#     if failures_detected == 0:
#         machine_status = "Machine Operating Normally"
#     else:
#         machine_status = "Attention Required"

#     # =================================
#     # RETURN ALL RESULTS
#     # =================================

#     return {
#         "power": power,

#         "machine_summary": {
#             "status": machine_status,
#             "failures_detected": failures_detected
#         },

#         "predictions": {

#             "TWF": {
#                 "prediction": twf_prediction,
#                 "result": (
#                     "Tool Wear Failure predicted"
#                     if twf_prediction == 1
#                     else "No Tool Wear Failure predicted"
#                 ),
#                 "failure_probability": round(
#                     float(twf_probability), 4
#                 ),
#                 "optimized_threshold":threshold_twf
#             },

#             "HDF": {
#                 "prediction": hdf_prediction,
#                 "result": (
#                     "Heat Dissipation Failure predicted"
#                     if hdf_prediction == 1
#                     else "No Heat Dissipation Failure predicted"
#                 ),
#                 "failure_probability": round(
#                     float(hdf_probability), 4
#                 ),
#                 "optimized_threshold": threshold_hdf
#             },

#             "PWF": {
#                 "prediction": pwf_prediction,
#                 "result": (
#                     "Power Failure predicted"
#                     if pwf_prediction == 1
#                     else "No Power Failure predicted"
#                 ),
#                 "failure_probability": round(
#                     float(pwf_probability), 4
#                 ),
#                 "optimized_threshold": threshold_pwf
#             },
#             "OSF": {
#                 "prediction": osf_prediction,
#                 "result": (
#                     "Overstrain Failure predicted"
#                     if osf_prediction == 1
#                     else "No Overstrain Failure predicted"
#                 ),
#                 "failure_probability": round(
#                     float(osf_probability), 4
#                 ),
#                 "optimized_threshold": threshold_osf
#             }
#         }
#     }


from  fastapi import FastAPI, HTTPException
from pydantic import BaseModel,Field

from prediction import predict_failures

app=FastAPI(
    title="Industrial Machine Failure Prediction API",
    description="Predictions machine failure types using trained ML models.",
    version="1.0.0"
)

class MachineData(BaseModel):
    machine_type:str=Field(
        ...,
        pattern="^[LMHlmh]$"
    )

    air_temperature:float
    process_temperature:float
    rotational_speed:float=Field(
        ...,
        gt=0
    )

    torque:float=Field(
        ...,
        gt=0
    )

    tool_wear:float=Field(
        ...,
        ge=0
    )


@app.get("/")

def home():
    return {
        "message":"Industrial Machine Failure Prediction API",
        "status":"runnig"
    }


@app.post("/predict")

def predict(machine:MachineData):

    try:
        result=predict_failures(
            machine_type=machine.machine_type,
            air_temperature=machine.air_temperature,
            process_temperature=machine.process_temperature,
            rotational_speed=machine.rotational_speed,
            torque=machine.torque,
            tool_wear=machine.tool_wear
        )

        return {
            "status":"success",
            "prediction":result
        }
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )