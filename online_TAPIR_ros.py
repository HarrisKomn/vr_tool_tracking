import cv2
import cv_bridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import time
from tapnet.torch import tapir_model
#from tapnet.models import tapir_model

import tree



def crop_image(start_point, end_point, image):
    if start_point and end_point:
        # Ensure the coordinates are in the correct order
        x1, y1 = start_point
        x2, y2 = end_point
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        # Crop the image using the coordinates of the rectangle
        cropped_image = image[y1:y2, x1:x2]
        #cv2.imshow("Cropped Image", cropped_image)

        return cropped_image


def preprocess_frames(frames):
    """Preprocess frames to model inputs.

    Args:
    frames: [num_frames, height, width, 3], [0, 255], np.uint8

    Returns:
    frames: [num_frames, height, width, 3], [-1, 1], np.float32
    """
    frames = frames.float()
    frames = frames / 255 * 2 - 1
    return frames

def get_frame(video_capture):
    r_val, image = video_capture.read()
    trunc = np.abs(image.shape[1] - image.shape[0]) // 2
    if image.shape[1] > image.shape[0]:
        image = image[:, trunc:-trunc]
    elif image.shape[1] < image.shape[0]:
        image = image[trunc:-trunc]
    return r_val, image


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.bridge = cv_bridge.CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/decklink/image_raw',  # Replace with your actual image topic
            self.on_image,
            1
        )
        self.device = 'cuda'

        model = model = tapir_model.TAPIR(pyramid_level=1, use_casual_conv=True)
        model.load_state_dict(torch.load("./checkpoints/causal_bootstapir_checkpoint.pt"))
        model = model.to(self.device)
        model = model.eval()
        torch.set_grad_enabled(False)
        self.model = model.to(self.device)

        self.pos = tuple()
        self.query_frame = True
        self.NUM_POINTS = 8
        self.low_res = 200
        self.crop_width = 800
        self.have_point = [False] * self.NUM_POINTS

        self.next_query_idx = 0
        self.last_click_time = 0
        self.step_counter = 0
        self.t = time.time()

        self.start_point = None
        self.end_point = None
        self.rect_drawing = False
        self.cropped_square = None

    def online_model_init(self, frames, points):
        """Initialize query features for the query points."""
        frames = preprocess_frames(frames)
        feature_grids = self.model.get_feature_grids(frames, is_training=False)
        features = self.model.get_query_features(
            frames,
            is_training=False,
            query_points=points,
            feature_grids=feature_grids,
        )
        return features
    

    def crop_img(self, image):

        def draw_rectangle(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:  # Mouse button down, start drawing
                self.start_point = (x, y)
                self.rect_drawing = True
                
            elif event == cv2.EVENT_MOUSEMOVE:  # Mouse move, update rectangle
                if self.rect_drawing:
                    temp_end_point = (x, y)
                    temp_image = image.copy()  # Copy to draw the rectangle on a temporary image
                    cv2.rectangle(temp_image, self.start_point, temp_end_point, (0, 255, 0), 2)
                    cv2.imshow("Image", temp_image)

                    
            
            elif event == cv2.EVENT_LBUTTONUP:  # Mouse button up, finalize the rectangle
                self.end_point = (x, y)
                self.rect_drawing = False
                cv2.rectangle(image, self.start_point, self.end_point, (0, 255, 0), 2)
                cv2.imshow("Image", image)
        

        cv2.imshow("Image", image)
        cv2.setMouseCallback("Image", draw_rectangle)

        
        while True:
            # Wait for user key input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):  # If 'r' is pressed, crop the image
                cropped_img = crop_image(self.start_point, self.end_point, image)       
                break     
            
            elif key == ord('q'):  # If 'q' is pressed, quit the loop
                break
        
        cv2.destroyAllWindows()
        return cropped_img



    def init_tapir(self, frame):


        def mouse_click(event, x, y, flags, param):
            del flags, param
            # event fires multiple times per click sometimes??
            if (time.time() - self.last_click_time) < 0.5:
                return

            if event == cv2.EVENT_LBUTTONDOWN:
                # self.pos = (y, frame.shape[1] - x)

                #self.pos = (x, y)
                

                x_start = max(x - self.crop_width//2, 0)
                x_end = min(x_start + self.crop_width, frame.shape[1])
                y_start = max(y - self.crop_width//2, 0)
                y_end = min(y_start + self.crop_width, frame.shape[0])


                # x_end = x_start + self.crop_width
                # y_end = y_start + self.crop_width

                cropped_width = x_end - x_start
                cropped_height = y_end - y_start

                scale_x = self.low_res / cropped_width
                scale_y = self.low_res / cropped_height

                new_pointx = (x - x_start) * scale_x  
                new_pointy = (y - y_start) * scale_y  

                #self.pos = (self.low_res//2, self.low_res//2)
                self.pos = (new_pointy, new_pointx) 

                self.query_frame = True
                self.last_click_time = time.time()
    
                self.cropped_square = (x_start, y_start, x_end, y_end)




        cv2.namedWindow("Point Tracking")
        #cv2.imshow("Point Tracking", frame[:, ::-1]) # flipped
        cv2.imshow("Point Tracking", frame)
        cv2.setMouseCallback("Point Tracking", mouse_click)

        while True:
            # Wait for user key input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):  # If 'r' is pressed, crop the image
                break     

        frame = frame[self.cropped_square[1]:self.cropped_square[3],self.cropped_square[0]:self.cropped_square[2]]
        frame = cv2.resize(frame, (self.low_res, self.low_res))

        query_points = torch.zeros([self.NUM_POINTS, 3], dtype=torch.float32)
        query_points = query_points.to(self.device)
        frame = torch.tensor(frame).to(self.device)

        self.query_features = self.online_model_init(
            frames=frame[None, None], points=query_points[None, :]
        )

        causal_state = self.model.construct_initial_causal_state(
            self.NUM_POINTS, len(self.query_features.resolutions) - 1
        )
        causal_state = tree.map_structure(lambda x: x.to(self.device), causal_state )
        self.causal_state = causal_state

        self.prediction, self.visible, self.causal_state = self.online_model_predict(
            frames=frame[None, None],
            features=self.query_features,
            causal_context=self.causal_state,
        )

        return 

    def on_image(self, msg):
        # Convert the ROS image message to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        #frame = frame[467:767,486:786,:]
        #frame = self.crop_img(frame)

        if self.query_frame:
            self.init_tapir(frame)

        frame = frame[self.cropped_square[1]:self.cropped_square[3],self.cropped_square[0]:self.cropped_square[2]]
        frame = cv2.resize(frame, (self.low_res, self.low_res))

        self.process_frame(frame)

    def online_model_predict(self, frames, features, causal_context):
        """Compute point tracks and occlusions given frames and query points."""
        frames = preprocess_frames(frames)
        feature_grids = self.model.get_feature_grids(frames, is_training=False)
        trajectories = self.model.estimate_trajectories(
            frames.shape[-3:-1],
            is_training=False,
            feature_grids=feature_grids,
            query_features=features,
            query_points_in_video=None,
            query_chunk_size=64,
            causal_context=causal_context,
            get_causal_context=True,
        )
        causal_context = trajectories["causal_context"]
        del trajectories["causal_context"]
        # Take only the predictions for the final resolution.
        # For running on higher resolution, it's typically better to average across
        # resolutions.
        tracks = trajectories["tracks"][-1]
        occlusions = trajectories["occlusion"][-1]
        uncertainty = trajectories["expected_dist"][-1]
        visibles = self.postprocess_occlusions(occlusions, uncertainty)
        return tracks, visibles, causal_context

    def postprocess_occlusions(self, occlusions, expected_dist):
        visibles = (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(expected_dist)) > 0.5
        return visibles


    def process_frame(self, frame):
        frame_np = frame
        if self.query_frame:
            query_points = np.array((0,) + self.pos, dtype=np.float32)
            frame = torch.tensor(frame).to(self.device)
            query_points = torch.tensor(query_points).to(self.device)

            init_query_features = self.online_model_init(
                frames=frame[None, None], points=query_points[None, None]
            
            )
            
            self.query_frame = False
            self.query_features, self.causal_state = self.model.update_query_features(
                query_features=self.query_features,
                new_query_features=init_query_features,
                idx_to_update=np.array([self.next_query_idx]),
                causal_state=self.causal_state,
            )
            self.have_point[self.next_query_idx] = True
            self.next_query_idx = (self.next_query_idx + 1) % self.NUM_POINTS
        if self.pos:
            frame = torch.tensor(frame).to(self.device)
            self.track, self.visible, self.causal_state = self.online_model_predict(
                frames=frame[None, None],
                features=self.query_features,
                causal_context=self.causal_state,
            )
            self.track = np.round(self.track[0, :, 0].cpu().numpy())
            self.visible = self.visible[0, :, 0].cpu().numpy()
            #print(self.track)

            for i, _ in enumerate(self.have_point):
                #print(self.visible[i])
                #print(self.have_point[i])
                if self.visible[i] and self.have_point[i]:
                    cv2.circle(
                        frame_np, (int(self.track[i, 0]), int(self.track[i, 1])), 2, (255, 0, 0), -1
                    )
                    #print(int(self.track[i, 0]), int(self.track[i, 1]))
                    if self.track[i, 0] < 16 and self.track[i, 1] < 16:
                        #print((i, self.next_query_idx))
                        print()
        #cv2.imshow("Point Tracking 2", frame_np[:, ::-1])

        frame_np = cv2.resize(frame_np, (self.crop_width, self.crop_width))
        cv2.imshow("Point Tracking", frame_np)

        if self.pos:
            self.step_counter += 1
            if time.time() - self.t > 5: # calculate the frames per second  every 5 seconds
                print(f"{self.step_counter/(time.time()-self.t)} frames per second")
                self.t = time.time()
                self.step_counter = 0
        else:
            self.t = time.time()


        # Wait for a short period to simulate video playback
        if cv2.waitKey(30) & 0xFF == ord('q'):
            return  # Break if 'q' is pressed

def main():
    rclpy.init()
    image_subscriber = ImageSubscriber()

    # Spin the ROS2 node to start receiving messages
    rclpy.spin(image_subscriber)

    # Shutdown the ROS2 client library when done
    rclpy.shutdown()

if __name__ == '__main__':
    main()
