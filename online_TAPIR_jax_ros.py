import time
import jax
import jax.numpy as jnp
from tapnet.models import tapir_model
from tapnet.utils import model_utils
import cv2
import cv_bridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np


def get_frame(video_capture):
  r_val, image = video_capture.read()
  trunc = np.abs(image.shape[1] - image.shape[0]) // 2
  if image.shape[1] > image.shape[0]:
    image = image[:, trunc:-trunc]
  elif image.shape[1] < image.shape[0]:
    image = image[trunc:-trunc]
  return r_val, image

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

        params, state = self.load_checkpoint("./checkpoints/causal_tapir_checkpoint.npy")

        self.model = tapir_model.ParameterizedTAPIR(
            params=params,
            state=state,
            tapir_kwargs=dict(
                use_causal_conv=True, bilinear_interp_with_depthwise_conv=False
            ),
        )

        self.pos = tuple()
        self.query_frame = True
        self.NUM_POINTS = 8
        self.low_res = 400
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

        self.online_init_apply = jax.jit(self.online_model_init)
        self.online_predict_apply = jax.jit(self.online_model_predict)

        self.fps_list = []
        self.frame_counter = 0
        cv2.namedWindow('Point Tracking', cv2.WINDOW_NORMAL)
        cv2.moveWindow('Point Tracking', 1000, 0) 

 



    def init_tapir(self, frame):

        def mouse_click(event, x, y, flags, param):
            del flags, param
            # event fires multiple times per click sometimes??
            if (time.time() - self.last_click_time) < 0.5:
                return

            if event == cv2.EVENT_LBUTTONDOWN:

                x_start = max(x - self.crop_width//2, 0)
                x_end = min(x_start + self.crop_width, frame.shape[1])
                y_start = max(y - self.crop_width//2, 0)
                y_end = min(y_start + self.crop_width, frame.shape[0])


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


        # cv2.imshow("Point Tracking", frame[:, ::-1]) # flipped
        cv2.imshow("Point Tracking", frame)
        cv2.setMouseCallback("Point Tracking", mouse_click)

        while True:
            # Wait for user key input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):  # If 'r' is pressed, crop the image
                break

        frame = frame[self.cropped_square[1]:self.cropped_square[3], self.cropped_square[0]:self.cropped_square[2]]
        frame = cv2.resize(frame, (self.low_res, self.low_res))

        query_features = None

        print("Compiling jax functions (this may take a while...)")
        # --------------------
        # Call one time to compile
        query_points = jnp.zeros([self.NUM_POINTS, 3], dtype=jnp.float32)
        _ = self.online_init_apply(
            frames=model_utils.preprocess_frames(frame[None, None]),
            points=query_points[None, 0:1],
        )
        jax.block_until_ready(query_features)
        self.query_features = self.online_init_apply(
            frames=model_utils.preprocess_frames(frame[None, None]),
            points=query_points[None, :],
        )

        self.causal_state = self.model.construct_initial_causal_state(
            self.NUM_POINTS, len(self.query_features.resolutions) - 1
        )

        self.prediction, self.causal_state = self.online_predict_apply(
            frames=model_utils.preprocess_frames(frame[None, None]),
            features=self.query_features,
            causal_context=self.causal_state,
        )

        jax.block_until_ready(self.prediction["tracks"])
        print("End of compilation")

        return


    def on_image(self, msg):
        # Convert the ROS image message to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        #frame = frame[467:767,486:786,:]

        if self.query_frame:
            self.init_tapir(frame)




        frame = frame[self.cropped_square[1]:self.cropped_square[3],self.cropped_square[0]:self.cropped_square[2]]
        frame = cv2.resize(frame, (self.low_res, self.low_res))

        # # Convert to PyTorch tensor and move it to the GPU
        # frame = torch.tensor(frame).permute(2, 0, 1).float().to('cuda')  # Convert to CHW format

        # # Resize using GPU (use torch's interpolation)
        # frame_tensor = torch.nn.functional.interpolate(
        #     frame.unsqueeze(0),  # Add batch dimension
        #     size=(self.low_res, self.low_res),
        #     mode='bilinear',
        #     align_corners=False
        # ).squeeze(0)  # Remove batch dimension


        #cv2.resizeWindow('
        # Tracking', 50, 50)

        self.process_frame(frame)


    def load_checkpoint(self, checkpoint_path):
      ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
      return ckpt_state["params"], ckpt_state["state"]


    def online_model_init(self, frames, points):
      feature_grids = self.model.get_feature_grids(frames, is_training=False)
      features = self.model.get_query_features(
          frames,
          is_training=False,
          query_points=points,
          feature_grids=feature_grids,
      )
      return features
    

    def log_fps_dur(self):

        if self.pos:
            self.step_counter += 1
            if time.time() - self.t > 5: # calculate the frames per second  every 5 seconds
                print(f"{self.step_counter / (time.time() - self.t)} frames per second")
                self.t = time.time()
                self.step_counter = 0
        else:
            self.t = time.time()


    def online_model_predict(self, frames, features, causal_context):
      """Compute point tracks and occlusions given frames and query points."""
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
      return {k: v[-1] for k, v in trajectories.items()}, causal_context
    
    def log_fps(self, start_time, end_time):

        processing_time = end_time-start_time
        fps = 1/processing_time if processing_time>0 else float('inf')
        self.fps_list.append(fps)

        print(f"Image {self.frame_counter}: FPS {fps:.2f}")

    def process_frame(self, frame):
        # frame_np = frame
        start_time = time.time()

        if self.query_frame:
            #query_points = np.array((0,) + self.pos, dtype=np.float32)

            query_points = jnp.array((0,) + self.pos, dtype=jnp.float32)
            init_query_features = self.online_init_apply(
                frames=model_utils.preprocess_frames(frame[None, None]),
                points=query_points[None, None],
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
            prediction, self.causal_state = self.online_predict_apply(
                frames=model_utils.preprocess_frames(frame[None, None]),
                features=self.query_features,
                causal_context=self.causal_state,
            )
            self.track = prediction["tracks"][0, :, 0]
            self.occlusion = prediction["occlusion"][0, :, 0]
            self.expected_dist = prediction["expected_dist"][0, :, 0]
            self.visibles = model_utils.postprocess_occlusions(self.occlusion, self.expected_dist)
            self.track = np.round(self.track)


            neighborhood_size = 3
            half_size = neighborhood_size // 2

            for i, _ in enumerate(self.have_point):
                if self.visibles[i] and self.have_point[i]:
                    x, y = self.track[i, 0].item(), self.track[i, 1].item()

                    # # Ensure the coordinates are within bounds of the image
                    # x_start = max(x - half_size, 0)
                    # x_end = min(x + half_size + 1, self.crop_width)
                    # y_start = max(y - half_size, 0)
                    # y_end = min(y + half_size + 1, self.crop_width)

                    # # Directly set the color for a small region around the point (e.g., blue)
                    # frame[y_start:y_end, x_start:x_end, 0] = 255.  # Blue channel

                    cv2.circle(
                        frame, (int(self.track[i, 0]), int(self.track[i, 1])), 5, (255, 0, 0), -1
                    )

                    if self.track[i, 0] < 16 and self.track[i, 1] < 16:
                        print((i, self.next_query_idx))


        # Resize the image (GPU-based)
        # frame_tensor_resized = F.interpolate(frame.unsqueeze(0), size=(self.low_res, self.low_res),
        #                                      mode='bilinear', align_corners=False).squeeze(0)

        # # Convert to CPU and numpy for display
        # frame_np = frame_tensor_resized.permute(1, 2, 0).cpu().numpy().astype('uint8')

        # cv2.imshow("Point Tracking", frame_np)

        frame_np = cv2.resize(frame, (self.crop_width, self.crop_width))
        if len(self.fps_list) > 0:
                fps_text = f"FPS: {self.fps_list[-1]:.2f}"
                cv2.putText(frame_np, fps_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Point Tracking", frame_np)
        cv2.waitKey(1)

        
        # FPS
        # Method 1: based on the processing time per frame
        # Record the total time for this frame (including visualization)
        end_time = time.time()
        self.log_fps(start_time, end_time)

        #Method 2: based on frames processed in 5 sec
        #self.log_fps_dur()


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