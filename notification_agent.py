from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage,HumanMessage
from tools.notification_tool import send_notification
from database.notification_state import(
    create_database,get_state,save_state,reset_state
)
from dotenv import load_dotenv
create_database()
MAX_NOTIFICATIONS=3


model=init_chat_model(model="ollama:qwen3:4b")
SYSTEM_PROMPT = """
You are a notification-writing assistant for an
industrial predictive maintenance system.

Write a short, clear maintenance notification using
only the information provided by Python.

Include:
- Machine ID
- Severity
- Failure type, if available
- Failure probability, if available
- Recommended maintenance action

Do not invent sensor values or predictions.

Return only the notification message.
Do not attempt to call tools or claim that a
notification has already been sent.
"""






def process_machine_event(machine_id,prediction_result,anomaly_result):
    predicted_failures=[]

    for failure_type, result in prediction_result.items():
        if isinstance(result,dict):
            prediction=result.get("prediction",0)

            if prediction==1:
                predicted_failures.append(
                    failure_type
                )


    anomaly=anomaly_result.get("anomaly",False)


    if predicted_failures:
        severity="CRITICAL"

        failure_type=max(predicted_failures,
                        key=lambda name: prediction_result[name].get(
                            "probability",0
                        )
                    )
    elif anomaly:
        severity="WARNING"
        failure_type=None
    else:
        severity="NORMAL"
        failure_type=None

    if severity=="NORMAL":
        reset_state(machine_id)

        return {
            "machine_id":machine_id,
            "severity":"NORMAL",
            "notification_sent":False,
            "message":"No notification required."
        }

    previous_state=get_state(machine_id)

    if previous_state is None:
        notification_count=0
    else:
        notification_count=previous_state["notification_count"]

        condition_changed=(
            previous_state["severity"]!=severity
            or previous_state["failure_type"]!=failure_type
        )

        if condition_changed:
            notification_count=0

    if notification_count>=MAX_NOTIFICATIONS:
        return {
            "machine_id":machine_id,
            "severity":severity,
            "notification_sent":False,
            "message":(
                "Notification limit reached."
                "for this unresolved condition"
            )
        }
    if severity=="CRITICAL":

        failure_data=prediction_result[failure_type]

        probability=(failure_data["probability"]*100)

        prompt=f"""

Machine ID: {machine_id}

Severity: CRITICAL

Predicted failure:
{failure_type}

Failure probability:
{probability:.2f}%

Anomaly detected:
{anomaly}

Anomaly score:
{anomaly_result.get("score")}

Recommend immediate machine inspection.
Write a short critical maintenance notification.


"""
        title="CRITICAL ALERT"

    else:
        prompt = f"""
Machine ID: {machine_id}

Severity: WARNING

No specific failure was predicted.

Anomaly detected: True

Anomaly score:
{anomaly_result.get("score")}

Anomaly threshold:
{anomaly_result.get("threshold")}

Recommend inspecting the unusual operating condition.
Write a short warning maintenance notification.

"""
        title="WARNING ALERT"


    response=model.invoke([SystemMessage(content=SYSTEM_PROMPT),
                        HumanMessage(content=prompt)])

    message=str(response.content).strip()

    if not message:
        return {
            "machine_id":machine_id,
            "severity":severity,
            "notification_sent":False,
            "message":"The LLM returned an empty notification."
        }

    tool_result=send_notification.invoke({
        "title":title,
        "message":message
    })

    if tool_result!="Notification sent successfully.":
        return {
            "machine_id":machine_id,
            "severity":severity,
            "notification_sent":False,
            "message":str(tool_result),
            "notification_count":notification_count
        }

    notification_count+=1

    save_state(
        machine_id=machine_id,
        severity=severity,
        failure_type=failure_type,
        notification_count=notification_count
    )


    return {
        "machine_id":machine_id,
        "severity":severity,
        "notification_sent":True,
        "agent_response":message,
        "tool_result":tool_result
    }