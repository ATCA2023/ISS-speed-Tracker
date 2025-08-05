# ISS Speed Tracker

This repository hosts a Python application designed to run on a Raspberry Pi aboard the International Space Station (ISS). It autonomously captures imagery and on‑board sensor data to compute a precise estimate of the station’s ground‑track speed.

The code was already ran by the Vsigma team in the 2024 Astro Pi competition. Physics logic, core algorithms and coding were done by Vlad Tiberiu.
---

## Features

* High‑resolution image capture with PiCamera

* Machine learning–powered cloud segmentation via a TensorFlow Lite model

* Pixel‑based image matching to estimate displacement

* 3‑axis accelerometer data from Sense HAT

* Conversion of pixel displacement to real‑world distance using calibrated Ground Sampling Distance (GSD)

* Kinematic fusion of displacement, time interval, and accelerations to compute instantaneous speed

* Weighted averaging of speed samples to produce a final, robust estimate

* Automatic geolocation tagging using Skyfield’s ISS ephemeris

* Continuous data logging to CSV for post‑processing or analysis

---

## Prerequisites

* **Hardware:** Raspberry Pi with PiCamera module and Sense HAT
* **Software & Libraries:**

  * Python 3

  * `picamera`

  * `sense_hat`

  * `numpy`, `opencv-python`

  * `tflite_runtime`

  * `skyfield`, `orbit`

  * `csv`, `math`, `time`, `datetime`

> Ensure your camera and Sense HAT are correctly configured and enabled in `raspi-config`.

---

## Usage

Run the main tracker script:

The script will run for a fixed duration (\~560 s), periodically capturing paired images and sensor readings. All raw data will be logged to `data.csv`, and the final weighted‑average speed will be written to `result.txt`.

---

## Detailed Workflow

1. **Initialization**

   * Configure PiCamera at full 4056×3040 resolution.
   * Load and allocate the TensorFlow Lite model for cloud segmentation.

2. **Data Acquisition Loop** (runs for \~9 min)

   * Record a timestamp and read raw accelerometer values (`ax, ay, az`) from Sense HAT.
   * Capture the first image, timestamp it, and query the ISS position to tag latitude/longitude.
   * Wait 13 s, then capture a second image and tag its position.
   * Invoke the cloud‑removal routine, which:

     * Preprocesses each image to 256×256 grayscale.
     * Runs the TFLite model to generate a cloud mask.
     * Applies the mask when land coverage is between the 15% and 60% sweet spot to avoid false matches.
   * Compute ORB keypoints and descriptors on the masked images.
   * Match descriptors with a brute‑force matcher and sort by distance.
   * Scale matched keypoints back to the original resolution.
   * Calculate pixel‑wise displacements and reduce them to a single value via a weighted average function.

3. **Speed Computation**

   * Convert pixel displacement to meters using a known GSD factor.
   * Compute the elapsed time squared.
   * Apply the constant‑acceleration kinematics formula:

     $$
v = \sqrt{ \frac{d^2}{t^2} + \frac{t^2}{4} \left(a_x^2 + a_y^2 + a_z^2\right) }
    $$
  
   * Append each instantaneous speed to a list.

4. **Final Aggregation**

   * Compute a robust weighted average across all speed samples to mitigate outliers.
   * Output the final speed in m/s to `result.txt` with five decimal places.

---

## Output Files

* **data.csv:** Raw logs of timestamps, geolocation, accelerometer readings, and pixel distances.
* **result.txt:** Final weighted‑average speed (m/s).
