import cv2
import mediapipe as mp
import time


def start_tracker():
    prev_time = 0  # sets initial time for fps.
    is_recording = True  # begins recording by default when main is executed.
    hand_index = 0  # sets hand to track.
    finger_index = 0  # sets finger to track.
    detector = HandTracking()  # creates instance of hand tracking class.

    while is_recording:

        img_rgb, image = detector.process_to_rgb()  # captures image and converts to RGB.
        hands = detector.find_hands(img_rgb)  # finds hand landmarks in the converted image.

        if hands:  # if hands are detected.
            image = detector.draw_guides(hands, image)  # draws dot for each tuple in each hand and connects them.
            points_list = detector.find_points(hands, hand_index, image)  # grabs point data & adds to list.

        image, prev_time = detector.show_fps(image, prev_time)  # displays frames per second.
        image = detector.show_track_settings(image)  # displays settings for the tracker.
        cv2.imshow("WebCamFeed", image)  # opens a window called string and shows image.

        if cv2.waitKey(1) == ord("x"):  # waits for (1) and checks if x was pressed.
            is_recording = False  # if it was pressed, stop recording.


class HandTracking:

    def __init__(self):
        self.mode = False
        self.max_hands = 2
        self.complx = 1
        self.detect_conf = 0.5
        self.track_conf = 0.7

        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # sets up capture feed using device specification.

        self.mpDraw = mp.solutions.drawing_utils  # used to draw markers on the video.
        self.mpHands = mp.solutions.hands  # library of hand data from mediapipe.
        self.mpHandObj = self.mpHands.Hands(self.mode, self.max_hands, self.complx, self.detect_conf, self.track_conf)

    def process_to_rgb(self):
        success, image = self.capture.read()  # captures image as numpy nd array.
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # converts the image into RGB for the Hands method.
        return img_rgb, image

    def find_hands(self, rgb_image):
        results = self.mpHandObj.process(rgb_image)  # processes hand image (RGB) into (x,y,z) of hand
        hands = results.multi_hand_landmarks  # grabs hand landmark data from the results
        if hands:
            return hands

    def find_points(self, hands, index, image):
        points_list = []  # creates a list for point data.
        for pointID, subLandmark in enumerate(hands[index].landmark):  # for every point in the specified hand.
            h, w, c = image.shape  # finds shape data of the image.
            x, y = (subLandmark.x * w), (subLandmark.y * h)  # scales the point data to match pixels. (world to rect)
            points_list.append([pointID, x, y])  # adds data to the list.
        return points_list  # sends out the list.

    def draw_guides(self, hands, image):
        for hand in hands:
            self.mpDraw.draw_landmarks(image, hand, self.mpHands.HAND_CONNECTIONS)  # adds drawing to the image.
        return image  # sends out the image.

    def show_fps(self, image, prev_time):
        current_time = time.time()  # finds elapsed time since the previous loop iteration.
        fps = 1 / (current_time - prev_time)  # loop iterates every frame (frame amount)/(time per frame).
        prev_time = current_time  # updates the timestamp for the next iteration.
        cv2.putText(image, str(int(fps)) + "fps", (590, 470), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255),
                    1)  # adds FPS text on image.
        return image, prev_time  # sends out the image and new timestamp.

    def show_track_settings(self, image):
        cv2.putText(image,
                    "detection: " + str(float(self.detect_conf)) + " tracking: " + str(float(self.track_conf)),
                    (10, 470), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)  # adds tracking parameters on image.
        return image  # sends out the image.


if __name__ == "__main__":
    start_tracker()
