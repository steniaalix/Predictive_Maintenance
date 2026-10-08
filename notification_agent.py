from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from tools.notification_tool import send_notification
from database.notification_state import(
    create_database,get_state,save_state,reset_state
)
from dotenv import load_dotenv
create_database()
MAX_NOTIFICATIONS=3


model=init_chat_model(model="ollama:qwen3:4b")

agent=create_agent(model=model,tools=[send_notification],
                system_prompt="""
You are a notification agent for a industrial predictive maintenance system.

Your job is to notify maintenance perrsonnel when a machine requires attention.

There are three severity levvels:

NORMAL:
No failure i spredicted and no anomaly is detedcted.
Do not send a notification.


WARNING:
An anomaly is detected but no failure is predicted.
Send a critical notification.

CRITICAL:
A machine failure is predicted.
Send a critical notification.

Use the send_notification tool to send the Windows desktop notification.

Keep notification short and clear.

Always include:
- Machine ID
- Severity
- Relevant  failure type if available
- Recommended action

Do not invent sensor valuess or predictions results.

Use only the information provided to you.
"""
)





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
        failure_type=predicted_failures[0]

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

    new_event=False

    if previous_state is None:
        new_event=True

        notification_count=0
    else:
        notification_count=previous_state["notification_count"]

        if(previous_state['severity']=="WARNING" and severity=="CRITICAL"):
            new_event=True
        elif(severity=="CRITICAL" and previous_state['failure_type']!=failure_type):
            new_event=True
        else:
            new_event=False
    if new_event:
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

Create a short critical maintenance notification.

The machine should be inspected immediately.
"""
    else:
        prompt = f"""
Machine ID: {machine_id}

Severity: WARNING

No specific failure was predicted.

However, an abnormal operating condition
was detected.

Anomaly score:
{anomaly_result.get("score")}

Anomaly threshold:
{anomaly_result.get("threshold")}

Create a short warning maintenance notification.

The machine should be inspected.
"""

    response=agent.invoke({
        "messages":[
            {
                "role":"user",
                "content":prompt
            }
        ]
    })


    notification_count+=1


    save_state(machine_id=machine_id,
            severity=severity,
            failure_type=failure_type,
            notification_count=notification_count
            )

    return {
        "machine_id":machine_id,
        "severity":severity,
        "notification_sent":True,
        "notification_count":notification_count,
        "agent_response":response
    }

