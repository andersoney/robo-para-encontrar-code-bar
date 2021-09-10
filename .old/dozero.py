import cv2
import argparse
import numpy as np
import time
import os


class FindObjectInImage(object):
    def __init__(self, args=None):
        self.boxes = []
        self.confidences = []
        self.class_ids = []
        if __name__ == '__main__':
            self.args = self.initArgs()
        else:
            if(args == None):
                print("No arguments passed")
                exit(1)
            self.args = args
        # generate different colors for different classes
        self.classes = None
        with open(self.args.classes, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))
        # read class names from text file

        pass

    def initArgs(self):
        # handle command line arguments
        ap = argparse.ArgumentParser()
        ap.add_argument('-i', '--image', required=True,
                        help='path to input image')
        ap.add_argument('-c', '--config', required=True,
                        help='path to yolo config file')
        ap.add_argument('-w', '--weights', required=True,
                        help='path to yolo pre-trained weights')
        ap.add_argument('-cl', '--classes', required=True,
                        help='path to text file containing class names')
        return ap.parse_args()

    # function to draw bounding box on the detected object with class name
    def draw_bounding_box(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = f"{str(self.classes[class_id])} - {confidence*100:.2f}%"
        color = self.COLORS[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x-10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def get_output_layers(self, net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1]
                         for i in net.getUnconnectedOutLayers()]
        return output_layers

    def processPrimaryLayer(self, outs, Width, Height):

        confiance_limit = 0.5
        # for each detetion from each output layer
        # get the confidence, class id, bounding box params
        # and ignore weak detections (confidence < confiance_limit)
        for out in outs:
            print(len(out))
            self.processSecondLayer(out,
                                    confiance_limit, Width, Height)

    def processSecondLayer(self, out, confiance_limit, Width, Height):
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confiance_limit:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                self.class_ids.append(class_id)
                self.confidences.append(float(confidence))
                self.boxes.append([x, y, w, h])

    def detectLayersFromObstacles(self, net):
        # run inference through the network
        # and gather predictions from output layers
        start = time.time()
        outs = net.forward(self.get_output_layers(net))
        end = time.time()
        print('Tempo gasto atual {:.5f} segundos'.format(end - start))
        return outs

    def processImage(self, image_file, showImage=True):
        args = self.args
        self.boxes = []
        self.confidences = []
        self.class_ids = []
        if __name__ != '__main':
            print(f"Tratando imagem: {image_file}\n")
        image = cv2.imread(image_file)
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392
        # read pre-trained model and config file
        net = cv2.dnn.readNet(args.weights, args.config)

        # create input blob
        blob = cv2.dnn.blobFromImage(
            image, scale, (416, 416), (0, 0, 0), True, crop=False)

        # set input blob for the network
        net.setInput(blob)

        # function to get the output layer names
        # in the architecture
        outs = self.detectLayersFromObstacles(net)

        # initialization
        conf_threshold = 0.5
        nms_threshold = 0.4
        self.processPrimaryLayer(
            outs, Width, Height)
        # apply non-max suppression
        indices = cv2.dnn.NMSBoxes(
            self.boxes, self.confidences, conf_threshold, nms_threshold)

        # go through the detections remaining
        # after nms and draw bounding box
        for i in indices:
            i = i[0]
            box = self.boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            self.draw_bounding_box(image, self.class_ids[i], self.confidences[i], round(
                x), round(y), round(x+w), round(y+h))

        # display output image
        if showImage:
            cv2.imshow("object detection", image)

            # wait until any key is pressed
            cv2.waitKey()

            # save output image to disk
            if(os.path.isdir('detected') == False):
                os.mkdir('detected')
            cv2.imwrite(f"detected/object-detection-single-run.jpg", image)

            # release resources
            cv2.destroyAllWindows()
        else:
            image_file = image_file.split("\\")[-1:][0]
            image_file = image_file.split(".")[0]
            if(os.path.isdir('detected') == False):
                os.mkdir('detected')
            cv2.imwrite(f"detected/object-detection-{image_file}.jpg", image)


if __name__ == '__main__':
    findObjectInImage = FindObjectInImage()
    findObjectInImage.processImage(
        findObjectInImage.args.image, findObjectInImage.args)
