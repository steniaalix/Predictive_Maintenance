import joblib 
import pandas as pd

twf_artifact=joblib.load("app/twf_final_model.joblib")
hdf_artifact=joblib.load("app/hdf_final_model.joblib")
pwf_artifact=joblib.load("app/pwf_final_model.joblib")
osf_artifact=joblib.load("app/osf_final_model.joblib")


models={
    "TWF":twf_artifact['model'],
    "HDF":hdf_artifact['model'],
    "PWF":pwf_artifact['model'],
    "OSF":osf_artifact['model']
}

thresholds={
    "TWF":twf_artifact['threshold'],
    "HDF":hdf_artifact['threshold'],
    "PWF":pwf_artifact['threshold'],
    "OSF":osf_artifact['threshold']
}

type_mapping={
    "L":0,
    "M":1,
    "H":2
}

feature_order = {
    "TWF": twf_artifact["feature_order"],
    "HDF": hdf_artifact["feature_order"],
    "PWF": pwf_artifact["feature_order"],
    "OSF": osf_artifact["feature_order"]
}

def predict_failures(machine_type:str,air_temperature:float,process_temperature:float,rotational_speed:float,torque:float,tool_wear:float):

    machine_type=machine_type.upper()
    if machine_type not in type_mapping:
        raise ValueError("machine_type must be L, M or H")
    encoded_type=type_mapping[machine_type]
    

    power=torque*rotational_speed

    input_data=pd.DataFrame([
        {
            "Type":encoded_type,
            "Air temperature [K]":air_temperature,
            "Process temperature [K]":process_temperature,
            "Rotational speed [rpm]":rotational_speed,
            "Torque [Nm]":torque,
            "Tool wear [min]":tool_wear,
            "Power":power
        }
    ])



    results={}

    for failure_type, model in models.items():
        model_input=input_data.copy()

        if failure_type in ["HDF","OSF"]:
            model_input.columns=(
                model_input.columns.str.replace("[","",regex=False).str.replace("]","",regex=False).str.replace("<","",regex=False)

            )

        model_input=model_input[feature_order[failure_type]]

        probability=model.predict_proba(model_input)[0][1]

        threshold=thresholds[failure_type]

        prediction=int(probability>=threshold)

        results[failure_type]={
            "probability":round(float(probability),6),
            "threshold":float(threshold),
            "prediction":prediction
        }


    results["RNF"]={
        "supported":False,
        "message":"RNF prediction is not reliable with the available dataset."
    }

    return results

