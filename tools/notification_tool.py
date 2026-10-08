from winotify import Notification
from langchain.tools import tool

@tool

def send_notification(title:str,message:str)->str:
    """
    Send an Windows desktop notification.
    
    Args:
        title: The title of the notification.
        message: The message displayed in the notification.
        
    Returns:
        A message indicating whether the notification was sent successfully.
    """

    try:
        toast=Notification(
            app_id="AI Agent",
            title=title,
            msg=message
        )

        toast.show()

        return "Notification sent successfully."

    except Exception as e:
        return f"Failed to sedn notification : {e}"