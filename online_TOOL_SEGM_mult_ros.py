import cv2
import cv_bridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
import numpy as np
import torch
import segmentation_models_pytorch as smp
from albumentations import Crop, Normalize, Resize, Compose
from albumentations.pytorch import ToTensorV2
from vis_utils.visualise_masks import overlay_mask_mult, extract_tool_tip_and_focus_region
import time

TRT_INFER = True # Option to use TensorRT inference or not

if TRT_INFER:
    import pycuda.autoinit
    import pycuda.driver as cuda
    import tensorrt as trt

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class Options:
    def __init__(self):
        
        
        ####TODO 
        if TRT_INFER:
            #self.engine_path = "./../trt/segm_model_2025_04_03.plan" # from DGX_SIMPLE_2025-04-03_16-51-17/segm_model.pth"  # Segformer, mit_b3
            self.engine_path = "./../trt/segm_model_2025_04_03_prof_SW.plan" # from DGX_SIMPLE_2025-04-04_13-21-43/segm_model.pth"  # Segformer, mit_b0

        else:
            self.model = "Segformer" # Unet, DeepLabV3Plus, UPerNet, Segformer
            self.encoder = "mit_b0"  # resnet34, resnet50, resnet101, mobilenet_v2,mit_b0,mit_b3

            # phantom with light device - multiclass
            #self.ckpt_path = "./checkpoints/DGX_SIMPLE_2025-04-03_16-51-17/segm_model.pth"  # Segformer, mit_b3
            self.ckpt_path = "./checkpoints/DGX_SIMPLE_2025-04-04_13-21-43/segm_model.pth"  # Segformer, mit_b0

        self.resize_img_w = 512
        self.resize_img_h = 512
        self.batch_size = 1

        

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

        # Publisher for the processed image
        self.image_publisher = self.create_publisher(
            Image,
            '/output_image_topic', 1  # Replace with your output image topic
        )

        # Publisher for the 3D point coordinates
        self.point_publisher = self.create_publisher(
            PointStamped,
            '/output_point_topic', 1  # Replace with your 3D point topic
        )


        self.frame_counter = 0
        self.window_frames = []

        self.device = 'cuda'


        ### model 
        opt = Options()

        if TRT_INFER:
            
            # load engine file
            out_engine_path = opt.engine_path
            with open(out_engine_path, "rb") as fp:
                plan = fp.read()

            logger = trt.Logger()
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(plan)
            context = engine.create_execution_context()

            self._context = context

            # Initialisation
            # create device buffers and TensorRT bindings
            self.output = np.zeros((1, 4, 512, 512), dtype=np.float32) # 4 channels because of 4 classes
            input_torch = torch.zeros(1, 3, 512, 512) # 3 channels because of RGB image


            input_np = to_numpy(input_torch)
            self.d_input = cuda.mem_alloc(input_np.nbytes)
            self.d_output = cuda.mem_alloc(self.output.nbytes)
            self.bindings = [int(self.d_input), int(self.d_output)]
            self.stream = cuda.Stream()


        else:

            self.ckpt_path = opt.ckpt_path
            if opt.model == 'Unet':
                model = smp.Unet(opt.encoder, encoder_weights=None, classes=4, activation=None)

            elif opt.model == 'DeepLabV3Plus':
                model = smp.DeepLabV3Plus(opt.encoder, encoder_weights=None, classes=4, activation=None)

            elif opt.model == 'Segformer':
                model = smp.Segformer(opt.encoder, encoder_weights=None, classes=4, activation=None)

            model.to(self.device)
            model.eval()
            state = torch.load(self.ckpt_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(state["state_dict"])
            self.model =  model



    
        self.mean, self.std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        self.resize_img_w, self.resize_img_h = opt.resize_img_w, opt.resize_img_h

        #self.bbox = (186, 167 , 1368, 986) #SDI
        self.bbox = (0, 0, 1920, 1078)  # SDI
        #self.bbox = (131, 311, 781-131, 809-311) #HDMI

        self.test_transform = self.get_transform(self.bbox)
        self.fps_list = []

        cv2.namedWindow('Segmentation', cv2.WINDOW_NORMAL)
        cv2.moveWindow('Segmentation', 0, 0)  
        #cv2.resizeWindow('Segmentation', 960, 1024)





    def get_transform(self, bbox):
        list_trans=[]
        x_min, y_min, w, h = bbox[0], bbox[1], bbox[2], bbox[3],

        list_trans.extend([Crop(x_min=x_min, y_min=y_min, x_max=(x_min+w), y_max=(y_min+h)),
                        Resize(width=self.resize_img_w, height=self.resize_img_h),
                        Normalize(mean=self.mean, std=self.std, p=1),
                        ToTensorV2()])

        list_trans=Compose(list_trans)
        return list_trans


    def prepare_frames(self, input_frame):

        prep_frame = self.test_transform(image = input_frame)['image']
        input_frame = torch.from_numpy(input_frame)

        return prep_frame, input_frame


    def on_image(self, msg):
        # Convert the ROS image message to OpenCV format
        input_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        prep_frame_tns, input_frame_tns = self.prepare_frames(input_frame)
        self.process_frame(prep_frame_tns, input_frame_tns)


    def process_frame(self, prep_frame, input_frame):
        
        start_time = time.time()
        with torch.no_grad():
            images = prep_frame.unsqueeze(0)
            images = images.to(self.device)

            input_img = input_frame.unsqueeze(0)
            input_img = input_img.to(self.device)

            x, y, w, h = self.bbox
            cropped_img = input_img[:, y:y + h, x:x + w, :]
            cropped_size_x, cropped_size_y = cropped_img.shape[1], cropped_img.shape[2]


            if TRT_INFER:
                images = images.repeat(1, 1, 1, 1) # batch size of 4
                image_array = to_numpy(images)
                input_buffer = np.ascontiguousarray(image_array)
                cuda.memcpy_htod(self.d_input, input_buffer, self.stream)
                self._context.execute_v2(bindings=self.bindings)
                cuda.memcpy_dtoh(self.output, self.d_output, self.stream)
                pred_mask = self.output
                pred_mask = np.transpose(pred_mask[0], (1, 2, 0))

                pred_mask = ToTensorV2()(image=pred_mask)["image"].unsqueeze(0).to(self.device)
                batch_preds = torch.softmax(pred_mask, dim=1)
                
            else:
                pred_mask = self.model(images)
                batch_preds = torch.softmax(pred_mask, dim=1)

            # Overlay masks on image
            overlayed_image = overlay_mask_mult(cropped_img, batch_preds, alpha=0.3)

            # Estimate/overlay tip of tool if tool is present
            # left tool
            left_pred = (batch_preds[:, 2] > 0.99).long()
            if left_pred.max()>0.:
                overlayed_image = extract_tool_tip_and_focus_region(overlayed_image, left_pred, 'left')

            # # right tool
            # right_pred = (batch_preds[:, 1] > 0.9).long()
            # if right_pred.max()>0.:
            #     overlayed_image = extract_tool_tip_and_focus_region(overlayed_image, right_pred, 'right')


            overlayed_image_np = overlayed_image.cpu().numpy()[0]

            if len(self.fps_list) > 0:
                fps_text = f"FPS: {self.fps_list[-1]:.2f}"
                cv2.putText(overlayed_image_np, fps_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Segmentation", overlayed_image_np)
            cv2.waitKey(1)

            # Record the total time for this frame (including visualization)
            end_time = time.time()

            self.log_fps(start_time, end_time)

            # Overlay masks on the whole image
            # whole_img = input_img#[0].detach().cpu().numpy()
            # whole_img[y:y + h, x:x + w, :] = overlayed_mask
            # cv2.imshow("Final image", whole_img)

            # # Convert the processed cv2 image back to a ROS image message
            # ros_image = self.bridge.cv2_to_imgmsg(whole_img, encoding='bgr8')

            # # Publish the processed image
            # self.image_publisher.publish(ros_image)


        self.frame_counter += 1
        if self.frame_counter % 100 == 0:
            print(f"Mean (std): FPS {np.mean(self.fps_list):.2f} ({np.std(self.fps_list):.2f})")


    def log_fps(self, start_time, end_time):

        processing_time = end_time-start_time
        fps = 1/processing_time if processing_time>0 else float('inf')
        self.fps_list.append(fps)

        print(f"Image {self.frame_counter}: FPS {fps:.2f}")


def main():
    rclpy.init()
    image_subscriber = ImageSubscriber()

    # Spin the ROS2 node to start receiving messages
    rclpy.spin(image_subscriber)

    # Shutdown the ROS2 client library when done
    rclpy.shutdown()

if __name__ == '__main__':
    main()
