import cv2
import os

modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt.txt"  # lines 4-12 configure the DNN face detection model to be used
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        if not self.isEmpty():
            return self.items.pop()


class Image:
    def __init__(self, img, name, savelocation):
        self.image = img
        self.name = name
        self.saveLoc = savelocation

    def extractFace(self, con=0.8):
        print("EXTRACTING FACE...")
        frameHeight = self.image.shape[0]  # load in image and get images height and width
        frameWidth = self.image.shape[1]
        blob = cv2.dnn.blobFromImage(self.image, 1.0, (300, 300), [104, 117, 123], False,
                                     False)  # create a blob from the image
        net.setInput(blob)
        detections = net.forward()  # give blob as input to neural network and pass it through net and store output from net in detections
        for i in range(detections.shape[2]):  # iterates over number of faces detected
            confidence = detections[0, 0, i, 2]  # gets the confidence from model that this is a face
            if confidence >= con:  # if confidence higher than 0.8
                print('FACE DETECTED IN ' + str(self.name))
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)  # gets coordinates of box that bounds the face
                x2 = int(detections[
                             0, 0, i, 5] * frameWidth)  # coordinates are normalised between 0 and 1 so need to be multiplied by image height and width
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                face = self.image[y1:y2, x1:x2]  # creates a sub image using original image and co-ordinates of face
                face = cv2.resize(face, (224, 224))
                cv2.imwrite(os.path.join(self.saveLoc, self.name), face)  # saves face to face folder


class Video:
    def __init__(self, address, node, level):
        self.videoAddress = address
        self.video = cv2.VideoCapture(self.videoAddress)
        self.node = node
        self.level = level

    def nameFrame(self, count, fps):
        if len(str(count // int(fps))) == 1:
            num = '000' + str(count // 30)
        elif len(str(count // int(fps))) == 2:
            num = '00' + str(count // int(fps))
        elif len(str(count // int(fps))) == 3:
            num = '0' + str(count // 30)
        else:
            num = str(count // 30)
        name = (str(self.node) + '-' + str(self.level) + '-' + num + '.jpg')
        return name

    def extractFrames(self, saveLoc, fps=30):
        frameQueue = Queue()
        count = 1
        success = 1
        while success:
            success, frame = self.video.read()
            if count % int(fps) == 0:
                frameName = self.nameFrame(count, fps)
                frameQueue.enqueue(frame)
            count += 1
        self.video.release()
        while not frameQueue.isEmpty():
            image = Image(frameQueue.dequeue(), frameName, saveLoc)
            image.extractFace()


# main subroutine takes in a 2D array of directories for each fold and the participants within that directory
def main(directories):
    levels = [0, 10]  # array holding levels we want to extract images for
    current = 0  # represents index of current directory we are working through from the parameter array
    for directory in directories:  # iterates through each directory
        for participants in directories[current][1]:  # iterates through each participant in each directory
            for level in levels:  # iterates through each level
                if level == 10:
                    if int(participants) <= 39:
                        dir = str(directory[0]) + '/' + str(participants) + '/'
                        for filename in os.listdir(dir):
                            lev = filename.split('.')
                            if lev[0] == '10':
                                address = dir + filename
                                print(address)
                                vid = Video(address, participants, level)
                                vid.extractFrames('C:/Users/Hadi/PycharmProjects/NEA/Faces/Train/10')
                    elif int(participants) > 39 and int(participants) <= 48:
                        dir = str(directory[0]) + '/' + str(participants) + '/'
                        for filename in os.listdir(dir):
                            lev = filename.split('.')
                            if lev[0] == '10':
                                address = dir + filename
                                print(address)
                                vid = Video(address, participants, level)
                                vid.extractFrames('C:/Users/Hadi/PycharmProjects/NEA/Faces/Valid/10')
                    else:
                        dir = str(directory[0]) + '/' + str(participants) + '/'
                        for filename in os.listdir(dir):
                            lev = filename.split('.')
                            if lev[0] == '10':
                                address = dir + filename
                                print(address)
                                vid = Video(address, participants, level)
                                vid.extractFrames('C:/Users/Hadi/PycharmProjects/NEA/Faces/Test/10')
                elif level == 0:
                    if int(participants) <= 39:
                        dir = str(directory[0]) + '/' + str(participants) + '/'
                        for filename in os.listdir(dir):
                            lev = filename.split('.')
                            if lev[0] == '0':
                                address = dir + filename
                                print(address)
                                vid = Video(address, participants, level)
                                vid.extractFrames('C:/Users/Hadi/PycharmProjects/NEA/Faces/Train/0')
                    elif int(participants) > 39 and int(participants) <= 48:
                        dir = str(directory[0]) + '/' + str(participants) + '/'
                        for filename in os.listdir(dir):
                            lev = filename.split('.')
                            if lev[0] == '0':
                                address = dir + filename
                                vid = Video(address, participants, level)
                                vid.extractFrames('C:/Users/Hadi/PycharmProjects/NEA/Faces/Valid/0')
                    else:
                        dir = str(directory[0]) + '/' + str(participants) + '/'
                        for filename in os.listdir(dir):
                            lev = filename.split('.')
                            if lev[0] == '0':
                                address = dir + filename
                                print(address)
                                vid = Video(address, participants, level)
                                vid.extractFrames('C:/Users/Hadi/PycharmProjects/NEA/Faces/Test/0')
        current += 1


# main([['F:/Fold1_part1',['01','02','03','04','05','06']],['F:/Fold1_part2',['07','08','09','10','11','12']],['G:/Fold2_part1',['13','14','15','16','17','18']],['G:/Fold2_part2',['19','20','21','22','23','24']],['C:/Users/Hadi/PycharmProjects/NEA/RawVideos/Fold3_part1',['25','26','27','28','29','30']],['C:/Users/Hadi/PycharmProjects/NEA/RawVideos/Fold3_part2',['31','32','33','34','35','36']],['G:/Fold4_part1',['37','38','39','40','42']],['G:/Fold4_part2',['43','44','45','46','47','48']],['F:/Fold5_part1',['49','50','51','52','53','54']],['C:/Users/Hadi/PycharmProjects/NEA/RawVideos/Fold5_part2',['55','56','57','58','59','60']]])
main([['C:/Users/Hadi/PycharmProjects/NEA/RawVideos/Fold5_part2', ['55', '56', '57', '58', '59', '60']]])
