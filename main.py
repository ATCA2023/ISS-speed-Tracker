from picamera import PiCamera
from sense_hat import SenseHat
from datetime import datetime, timedelta
import time
import csv
import math
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
from skyfield.api import load
from orbit import ISS

# Initialize the PiCamera with the desired resolution
camera = PiCamera(resolution=(4056, 3040))

# Load the TensorFlow Lite model for cloud segmentation
interpreter_cloud = tflite.Interpreter(model_path='cloud_segmentation_model.tflite')
interpreter_cloud.allocate_tensors()

# Get the input and output details for cloud segmentation model
input_details_cloud = interpreter_cloud.get_input_details()
output_details_cloud = interpreter_cloud.get_output_details()

def get_acceleration_data():
    sense = SenseHat()
    acceleration_data = sense.get_accelerometer_raw()
    ax1, ay1, az1 = acceleration_data['x'], acceleration_data['y'], acceleration_data['z']
    return ax1, ay1, az1

# Function to calculate distance in pixels (placeholder implementation)
def remove_clouds(image_path):
    # Load image
    input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) / 255.0

    # Resize the input image to the same size as the training images
    input_image = cv2.resize(input_image, (256, 256))
    input_image = np.expand_dims(input_image, axis=(0, -1))

    # Convert input data to float32
    input_image = input_image.astype(np.float32)

    # Set the input tensor for cloud segmentation model
    interpreter_cloud.set_tensor(input_details_cloud[0]['index'], input_image)

    # Run inference for cloud segmentation
    interpreter_cloud.invoke()

    # Get the predicted mask from the output tensor
    predicted_mask = interpreter_cloud.get_tensor(output_details_cloud[0]['index'])

    # Post-process the predicted mask
    threshold = 0.0018
    cloud_mask = (predicted_mask > threshold).astype(np.uint8) * 255

    # Calculate the percentage of land in the original image
    land_percentage = 1 - np.sum(cloud_mask == 0) / np.prod(cloud_mask.shape)

    # Remove clouds only if there is less than 60% land but more than 15%
    if 0.15 < land_percentage < 0.6:
        result = cv2.bitwise_and(input_image[0], input_image[0], mask=cloud_mask[0])
    else:
        result = input_image[0]  # Return the original image without removing clouds

    return result

def calculate_distance_px(image_path_1, image_path_2):
    # Remove clouds from images
    image_1_no_clouds = remove_clouds(image_path_1)
    image_2_no_clouds = remove_clouds(image_path_2)

    # Convert images to OpenCV format
    image_1_cv = np.array(image_1_no_clouds * 255, dtype=np.uint8)
    image_2_cv = np.array(image_2_no_clouds * 255, dtype=np.uint8)

    # Calculate keypoints and descriptors using ORB
    orb = cv2.ORB_create(nfeatures=5000)
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1_cv, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2_cv, None)

    # Match descriptors
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Original resolution of the images (width and height)
    original_resolution = (4056, 3040)

    # Calculate scaled points for Image 1
    points_1 = np.array([keypoints_1[match.queryIdx].pt for match in matches])
    scaled_points_1 = np.array(
        [(point[0] * original_resolution[0] / 256, point[1] * original_resolution[1] / 256) for point in points_1])

    # Calculate scaled points for Image 2
    points_2 = np.array([keypoints_2[match.trainIdx].pt for match in matches])
    scaled_points_2 = np.array(
        [(point[0] * original_resolution[0] / 256, point[1] * original_resolution[1] / 256) for point in points_2])

    distances = [np.linalg.norm(scaled_points_1[i] - scaled_points_2[i]) for i in range(len(points_1))]

    distance = weighted_average(distances)

    return distance


# Function to calculate weighted average of speed
def weighted_average(values):
    n = len(values)

    if n==0:
        return -1

    mean = sum(values) / n

    # Calculate weights
    weights = [1 / (abs(val - mean) + 1e-10) for val in values]

    # Calculate the weighted sum and sum of weights
    weighted_sum = sum(w * val for w, val in zip(weights, values))
    sum_of_weights = sum(weights)

    # Calculate the weighted average
    weighted_avg = weighted_sum / sum_of_weights

    return weighted_avg

# Main program
nrphotos = 0
start_time = time.time()
speed = []

with open('data.csv', 'w') as f:
    writer = csv.writer(f)
    header = ("No.", "time1", "time2", "time3", "latitude", "longitude", "Acc X", "Acc Y", "Acc Z", "Dist(px)")
    writer.writerow(header)

while (time.time() - start_time < 560):  # Run for 560 seconds

    time1 = time.time()

    acceleration_1 = get_acceleration_data()

    # Save the acceleration values independently
    ax, ay, az = acceleration_1

    ax *= 9.80665
    ay *= 9.80665
    az *= 9.80665

    # Take picture 1 and store it in the same directory as the script
    picture_path_1 = f"picture_{nrphotos}_1.jpg"
    camera.capture(picture_path_1)
    time2 = time.time()

    auxiliary_time1 = load.timescale().now()
    current_position1 = ISS.at(auxiliary_time1)
    current_location1 = current_position1.subpoint()

    # Wait for 13 seconds
    time.sleep(13)

    # Take picture 2 and store it in the same directory as the script
    picture_path_2 = f"picture_{nrphotos}_2.jpg"
    camera.capture(picture_path_2)
    time3 = time.time()

    auxiliary_time2 = load.timescale().now()
    current_position2 = ISS.at(auxiliary_time2)
    current_location2 = current_position2.subpoint()

    # Calculate distance in pixels
    dist_px = calculate_distance_px(f"picture_{nrphotos}_1.jpg",f"picture_{nrphotos}_2.jpg")

    # Store data in CSV file
    with open('data.csv', mode='a') as file:
        writer = csv.writer(file)
        writer.writerow([nrphotos, time1, time2, time3, current_location1.latitude.degrees, current_location2.longitude.degrees, ax, ay, az, dist_px])


    if dist_px == -1: 
        continue

    # GSD value
    GSD = 12648

    # Calculate distance in meters
    distance = dist_px * GSD / 100000

    # Calculate squared differences
    timediff = time3 - time1
    timediffsq = timediff * timediff

    # Calculate speed using the provided formula
    speed.append(math.sqrt( (distance * distance) / timediffsq + (timediffsq / 4.0) * (ax * ax + ay * ay + az * az) ))


    nrphotos += 1

# Calculate the final weighted average speed
final_speed = weighted_average(speed)

print(f"Final Weighted Average Speed: {final_speed} m/s")

# Output the result to a file with 5 decimals
output_file_path = "result.txt"
with open(output_file_path, "w") as file:
    file.write(f"{final_speed:.5f}")

