from notification_agent import process_machine_event
from database.notification_state import get_state,reset_state

def make_prediction(failure_type=None):
    """Create deterministric prediction results for testing."""

    result={
        "TWF":{"prediction":0,"probability":0.10,"threshold":0.91},
        "HDF": {"prediction": 0, "probability": 0.10, "threshold": 0.91},
        "PWF": {"prediction": 0, "probability": 0.01, "threshold": 0.03},
        "OSF": {"prediction": 0, "probability": 0.12, "threshold": 0.78}
    }

    if failure_type:
        result[failure_type]["prediction"]=1
        result[failure_type]['probability']=0.94

    return result

def make_anomaly(is_anomaly):
    return {
        "anomaly": is_anomaly,
        "score": -0.08 if is_anomaly else 0.03,
        "threshold": -0.065427
    }

def run_event(machine_id, failure_type=None, anomaly=False):
    return process_machine_event(
        machine_id=machine_id,
        prediction_result=make_prediction(failure_type),
        anomaly_result=make_anomaly(anomaly),
    )


def check(condition,message):
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")

def test_notification_limit():
    print("\n" + "=" * 60)
    print("TEST 1: THREE-NOTIFICATION LIMIT")
    print("=" * 60)

    machine_id = "TEST_LIMIT_001"
    reset_state(machine_id)

    results = []

    for attempt in range(1, 5):
        print(f"\nAttempt {attempt}")

        result = run_event(
            machine_id=machine_id,
            failure_type="TWF",
            anomaly=True,
        )

        results.append(result)

        print("Severity:", result["severity"])
        print("Notification sent:", result["notification_sent"])
        print("Message:", result.get("message", result.get("agent_response")))

    state = get_state(machine_id)

    check(
        [r["notification_sent"] for r in results]
        == [True, True, True, False],
        "Only the first three notification decisions send notifications",
    )

    check(
        state is not None and state["notification_count"] == 3,
        "Database count stops at three",
    )


# ==================================================
# TEST 2: WARNING -> CRITICAL ESCALATION
# ==================================================

def test_warning_to_critical():
    print("\n" + "=" * 60)
    print("TEST 2: WARNING -> CRITICAL ESCALATION")
    print("=" * 60)

    machine_id = "TEST_ESCALATION_001"
    reset_state(machine_id)

    print("\nFirst event: WARNING")

    warning_result = run_event(
        machine_id=machine_id,
        failure_type=None,
        anomaly=True,
    )

    print(warning_result)

    check(
        warning_result["notification_sent"] is True,
        "Initial WARNING sends a notification",
    )

    warning_state = get_state(machine_id)

    check(
        warning_state["severity"] == "WARNING"
        and warning_state["notification_count"] == 1,
        "WARNING state is stored with count one",
    )

    print("\nSecond event: CRITICAL")

    critical_result = run_event(
        machine_id=machine_id,
        failure_type="TWF",
        anomaly=True,
    )

    print(critical_result)

    critical_state = get_state(machine_id)

    check(
        critical_result["notification_sent"] is True,
        "Escalation to CRITICAL sends a new notification",
    )

    check(
        critical_state["severity"] == "CRITICAL"
        and critical_state["failure_type"] == "TWF"
        and critical_state["notification_count"] == 1,
        "Escalation starts a new notification count",
    )


# ==================================================
# TEST 3: CRITICAL -> NORMAL -> CRITICAL
# ==================================================

def test_recovery_and_new_incident():
    print("\n" + "=" * 60)
    print("TEST 3: RECOVERY AND NEW INCIDENT")
    print("=" * 60)

    machine_id = "TEST_RECOVERY_001"
    reset_state(machine_id)

    print("\nFirst event: CRITICAL")

    first_critical = run_event(
        machine_id=machine_id,
        failure_type="HDF",
        anomaly=True,
    )

    print(first_critical)

    check(
        first_critical["notification_sent"] is True,
        "First CRITICAL event sends a notification",
    )

    print("\nSecond event: NORMAL")

    normal_result = run_event(
        machine_id=machine_id,
        failure_type=None,
        anomaly=False,
    )

    print(normal_result)

    check(
        normal_result["notification_sent"] is False,
        "NORMAL event does not send a notification",
    )

    check(
        get_state(machine_id) is None,
        "NORMAL event clears the saved notification state",
    )

    print("\nThird event: CRITICAL again")

    second_critical = run_event(
        machine_id=machine_id,
        failure_type="HDF",
        anomaly=True,
    )

    print(second_critical)

    new_state = get_state(machine_id)

    check(
        second_critical["notification_sent"] is True,
        "A new CRITICAL incident sends a notification after recovery",
    )

    check(
        new_state is not None
        and new_state["notification_count"] == 1,
        "The new incident starts with count one",
    )


# ==================================================
# RUN ALL TESTS
# ==================================================

if __name__ == "__main__":
    test_notification_limit()
    test_warning_to_critical()
    test_recovery_and_new_incident()

    print("\n" + "=" * 60)
    print("ALL NOTIFICATION WORKFLOW TESTS PASSED")
    print("=" * 60) 