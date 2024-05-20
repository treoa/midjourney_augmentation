import os
import cv2
import time
import logging
import inspect
import requests
import warnings

from ultralytics import YOLO

warnings.filterwarnings("ignore")

# Custom formatter for coloring error messages red
class CustomFormatter(logging.Formatter):
    """
        A custom log formatter that adds color to log messages based on their log level.

        Args:
            fmt (str): The log message format string.

        Methods:
            format(record): Formats the log record and adds color based on the log level.

        Attributes:
            RED (str): The ANSI escape code for red color.
            RESET (str): The ANSI escape code to reset the color.
            GREEN (str): The ANSI escape code for green color.
            YELLOW (str): The ANSI escape code for yellow color.
            FORMAT (str): The default log message format string.
    """
    RED = '\033[91m'
    RESET = '\033[0m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

    def __init__(self, fmt=FORMAT):
        super().__init__(fmt)

    def format(self, record):
        if record.levelno == logging.ERROR:
            return f"{self.RED}{super().format(record)}{self.RESET}"
        elif record.levelno == logging.DEBUG:
            return f"{self.YELLOW}{super().format(record)}{self.RESET}"
        return super().format(record)


# Set up the logging with the custom formatter
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def GetResponse(url: str, json : dict, headers: dict) -> bool:
    """
    The function for sending the post request and handling the validity of response
    
    Args:
        url (str): The URL to send the POST request to.
        json (dict): The JSON data to send in the POST request.
        headers (dict): The headers to include in the POST request.
    
    Returns:
        bool: True if the response is successful, False otherwise.
    """
    stack = inspect.stack()


    try:
        class_name = stack[1].frame.f_locals["self"].__class__.__name__ or "Unknown"
        module_name = stack[1].frame.f_globals["__name__"] or "Unknown"
        function_name = stack[1].function or "Unknown"
        response = requests.post(url = url, json = json, headers = headers)
        logme(f"POST request from {module_name} -> {class_name} -> {function_name}")
        return _ResponseCheck(response)
    except Exception as e:
        return (False, f"ResponseError in Location: GetResponse, Msg: {e}")

def _ResponseCheck(Response) -> tuple:
    """
        Check for the validity of the response object and return a tuple indicating success or failure.
        
        Args:
            Response (_type_): requests response like object
        
        Returns:
            tuple: A tuple containing a boolean indicating success or failure, and the response object itself if successful.
    """
    if Response.status_code >= 400:
        return (False, f"ResponseError in Location: ResponseCheck, Msg: {Response.text}, Code: {Response.status_code}")
    return (True, Response)

# Custom formatter for coloring error messages red
def crop_face(file: str) -> None:
    yolov8_model = os.path.join(os.getcwd(), "yolov8n-face.pt")

    img = cv2.imread(file)
    model = YOLO(yolov8_model)
    results = model(img)

    for result in results:
        for i, det in enumerate(result.boxes):
            conf = det.conf.item()
            if conf >= 0.8:
                # Ensure the tensor is on the CPU
                det_cpu = det.xyxy.to('cpu')

                # Extract and convert each element
                x1, y1, x2, y2 = [int(det_cpu[0, i].item()) for i in range(4)]
                face = img[y1:y2, x1:x2]
                # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                image_name, extension = os.path.splitext(file)
                image_name = f"{image_name}_cropped{extension}"
                cv2.imwrite(image_name, face)
                logme(f"Cropped image saved as {image_name}")


def logme(message, level='info'):
    """Logging message as info or error with different formatting

    Args:
        message (_type_): _description_
        level (str, optional): _description_. Defaults to 'info'.
    """
    log_function = getattr(logger, level.lower(), logger.info)
    log_function(message)


