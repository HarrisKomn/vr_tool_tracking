import cv2
import cv_bridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import random


def on_select(eclick, erelease):
    """
    Callback for rectangle selection.
    Stores the top-left and bottom-right coordinates of the rectangle.
    """
    global bbox_coords
    x1, y1 = int(eclick.xdata), int(eclick.ydata)  # Starting point (top-left)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)  # Ending point (bottom-right)
    bbox_coords = (x1, y1, x2, y2)
    print(f"Rectangle selected: Top-left ({x1}, {y1}), Bottom-right ({x2}, {y2})")

def on_key(event):
    """
    Callback for keypress events.
    Prints the bounding box coordinates when "R" is pressed.
    """
    if event.key.lower() == 'r' and bbox_coords:
        print(f"Bounding box coordinates: {bbox_coords}")
        plt.close()  # Close the plot after user presses 'R'

def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def generate_gradient_colors(n):
    colors = []
    base_color = [0, 0, 255]  # Start with a base color (a shade of gray)
    
    for i in range(n):
        # Gradually increase the RGB values slightly for each color
        r = base_color[0] 
        g = (base_color[1] + i * 10) % 256
        b = base_color[2]
        colors.append((r, g, b))
    
    return colors

def apply_overlay(image, pred_tracks=None):
    # Overlay points or other visual elements on the image
    
    if pred_tracks is not None:

        points_to_overlay = np.array(pred_tracks.cpu().detach())
        n=len(points_to_overlay)
        colors = generate_gradient_colors(n)
        
        for i, point in enumerate(points_to_overlay):

            color = colors[i % n]  # Ensure the color index is within range
            cv2.circle(image, tuple([int(point[0]), int(point[1])]), radius=5, color=color, thickness=-1)

            #cv2.circle(image, tuple([int(point[0]), int(point[1])]), radius=5, color=(0, 0, 255), thickness=-1)
    return image

def create_custom_mask(image, top_left, bottom_right):
    """
    Create a binary mask with a white box defined by top-left and bottom-right coordinates.
    """
    mask = np.zeros_like(image, dtype=np.uint8)
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Ensure coordinates are within the image boundaries
    x1, x2 = max(0, x1), min(image.shape[1], x2)
    y1, y2 = max(0, y1), min(image.shape[0], y2)

    # Create the white box
    mask[y1:y2, x1:x2] = 1
    return mask[:,:,0]





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
        self.frame_counter = 0
        self.window_frames = []
        self.mask_segm = None
        self.device = 'cuda'
        # self.model = torch.hub.load("facebookresearch/co-tracker", "cotracker2_online").to(self.device)  

        self.model = cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(self.device)
        self.is_first_step = True
        self.global_bbox_coords = None # image crop
        self.local_bbox_coords = None  # cotracker grid 
        self.t = time.time()
        self.step_counter = 0


    def init_cotracker(self, frame):

        def mouse_click(event, x, y, flags, param):
            del flags, param

            if event == cv2.EVENT_LBUTTONDOWN:
                self.pos = (y, x)

                self.global_bbox_coords = (self.pos[0]-200, self.pos[1]-200, self.pos[0]+200, self.pos[1]+200)
                self.pos = (200, 200)
                self.local_bbox_coords = (self.pos[0]-20, self.pos[1]-20, self.pos[0]+20, self.pos[1]+20)


        cv2.namedWindow("Point Tracking")
        cv2.imshow("Point Tracking", frame)
        cv2.setMouseCallback("Point Tracking", mouse_click)

        while True:
            # Wait for user key input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):  # If 'r' is pressed, crop the image
                break     

        frame = frame[self.global_bbox_coords[0]:self.global_bbox_coords[2],self.global_bbox_coords[1]:self.global_bbox_coords[3]]


        # Ensure two points were selected
        if self.global_bbox_coords is not None:
            top_left, bottom_right = (self.local_bbox_coords[0], self.local_bbox_coords[1]), (self.local_bbox_coords[2], self.local_bbox_coords[3])
            self.mask_segm = create_custom_mask(frame, top_left, bottom_right)*255.

            # plt.imshow(self.mask_segm)
            # plt.show()



    def _process_step(self,
        window_frames, is_first_step, grid_size, grid_query_frame, segm_mask
    ):
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-self.model.step * 2 :]), device=self.device
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        return self.model(
            video_chunk,
            is_first_step=is_first_step,
            grid_size=grid_size,
            segm_mask=torch.from_numpy(segm_mask)[None, None],
            # grid_query_frame=grid_query_frame,
        ) 

    def on_image(self, msg):
        # Convert the ROS image message to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        if self.frame_counter==0:
            self.init_cotracker(frame)

        frame = frame[self.global_bbox_coords[0]:self.global_bbox_coords[2],self.global_bbox_coords[1]:self.global_bbox_coords[3]]


        self.process_frame(frame)

    def process_frame(self, frame):

        self.window_frames.append(frame)

        if self.frame_counter % 8 == 0 and self.frame_counter != 0:
            # Example processing using your model (replace with actual processing)
            pred_tracks, pred_visibility = self._process_step(
                self.window_frames,
                is_first_step=self.is_first_step,  # Set the correct step status
                grid_size=20,  # Adjust as needed
                grid_query_frame=0,  # Adjust as needed
                segm_mask=self.mask_segm
            )

            self.is_first_step =  False

            if pred_tracks is not None:
                overlay_image = apply_overlay(frame, pred_tracks=pred_tracks[0, -1])

                cv2.imshow('Tracking Output', overlay_image)
                # Wait for a short period to simulate video playback
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    return  # Break if 'q' is pressed



                self.step_counter += 1
                if time.time() - self.t > 5:
                    print(f"{self.step_counter/(time.time()-self.t)} frames per second")
                    self.t = time.time()
                    self.step_counter = 0

        self.frame_counter += 1


def main():
    rclpy.init()
    image_subscriber = ImageSubscriber()

    # Spin the ROS2 node to start receiving messages
    rclpy.spin(image_subscriber)

    # Shutdown the ROS2 client library when done
    rclpy.shutdown()

if __name__ == '__main__':
    main()
