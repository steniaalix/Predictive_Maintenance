from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

model = init_chat_model(model="ollama:qwen3:4b", temperature=0)
SYSTEM_PROMPT = """
You write concise, professional maintenance task descriptions for an industrial
machine monitoring system. Use only the supplied information. Do not invent
sensor readings, diagnoses, failure types, or procedures. Do not change the
assigned role, priority, or response target. Do not claim a failure is physically
confirmed. Return only the task description.
"""

def generate_task_description(machine_id, severity, failure_type, failure_probability, assignment):
    failure_text = failure_type or "No specific failure predicted."
    if failure_probability is None:
        probability_text = "Not available"
    else:
        if not 0 <= failure_probability <= 1:
            raise ValueError("failure_probability must be between 0 and 1.")
        probability_text = f"{failure_probability * 100:.2f}%"
    user_prompt = f"""
Create a concise maintenance task using this approved information.
Machine ID: {machine_id}
Severity: {severity}
Failure type: {failure_text}
Failure probability: {probability_text}
Assigned maintenance role: {assignment["assigned_role"]}
Priority: {assignment["priority"]}
Response target: {assignment["response_target"]}
Recommended action: {assignment["recommended_action"]}
Write two to four sentences. State what should be inspected and recorded,
without inventing extra procedures.
"""
    response = model.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)])
    description = str(response.content).strip()
    if not description:
        raise ValueError("The model returned an empty task description.")
    return description
