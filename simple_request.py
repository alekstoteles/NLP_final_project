# USAGE
# python simple_request.py

# import the necessary packages
import requests

# initialize the Keras REST API endpoint URL along with the input
# image path
# KERAS_REST_API_URL = "http://localhost:5000/predict"
KERAS_REST_API_URL = "http://169.50.135.105:5000/predict"
abstract = "the present invention provides a battery management device and a portable computer the battery management device is for managing the rechargeable battery provided in a portable computer the portable computer is provided with a charge circuit for charging a battery the battery management device comprises a discharge circuit for discharge the battery a acquiring module for acquire the mode determining parameter a first determining module for judging whether the battery storage mode is entered according to the mode determining parameter a first control module for controlling the charge circuit or the discharge circuit to control the amount of electrical charge of the battery so that the amount of electrical charge of the battery is lower than the second amount of electrical charge threshold in the battery storage mode the performance of the battery when deposited with the amount of electrical charge lower than the second amount of electrical charge threshold is better than the performance when deposited with the amount of electrical charge higher than the second amount of electrical charge threshold the present invention alleviates the attenuation of the amount of electrical charge capacity of the rechargeable battery of the portable computer"


# submit the request
r = requests.post(KERAS_REST_API_URL, data=abstract).json()

# ensure the request was sucessful
if r["success"]:
    print (r["labels"])
# otherwise, the request failed
else:
    print("Request failed")
