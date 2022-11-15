import cv2
import numpy as np


class PolygonSelectionTool:
    '''
    This class implements a polygonal selection tool for selecting a region
    in an image defined by a polygon. It takes an image and provides an interface
    through cv2 to create a custom polygonal region in that image. The selection
    method then returns a mask of the location of the pixels in that selection.

    Instructions:
    Left click to add a point. Clicking the first point completes the shape.
    The entire shape may be erased at anytime by double clicking the mousewheel.
    When you are satisfied with completed shape, press Space Bar to confirm.

    Created By: Alec Soronow
    '''

    def __init__(self, filepath):
        '''Initializes the selection tool with a given image'''
        # Are we recording the first point flag
        self.firstPoint = True
        # First coordinates
        self.firstX = 0
        self.firstY = 0
        # The latest coordinates placed
        self.last_x = 0
        self.last_y = 0
        # Are we still drawing points flag
        self.drawing = True
        # An array of verts in the polygon
        self.verts = None
        # A cached version of the image without any points plotted
        self.cache = None
        # The read image
        self.image = cv2.imread(filepath)
        # An array representing which pixels are within the selection
        self.mask = None

    def drawPolygonSelection(self, event, curr_x, curr_y, flags, param):
        '''A cv2 event handler function that enables drawing points on an image to form a polygon'''
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.firstPoint:
                self.cache = self.image.copy()
                self.firstPoint = False
                self.firstX, self.firstY = curr_x, curr_y
                self.last_x, self.last_y = curr_x, curr_y
                self.verts = np.array([[self.last_x, self.last_y]])
                cv2.circle(self.image, (self.last_x, self.last_y),
                           2, (0, 0, 255), 3)
            elif not self.firstPoint:
                if abs(self.firstX-curr_x) <= 15 and abs(self.firstY - curr_y) <= 15:
                    cv2.polylines(self.image, np.int32(
                        [self.verts]), True, (0, 0, 255), 2)
                else:
                    cv2.circle(self.image, (curr_x, curr_y), 2, (0, 0, 255), 3)
                    cv2.line(self.image, (self.last_x, self.last_y),
                             (curr_x, curr_y), (0, 255, 255), 2)
                    self.last_x, self.last_y = curr_x, curr_y
                    self.verts = np.append(
                        self.verts, [[self.last_x, self.last_y]], axis=0)
        elif event == cv2.EVENT_MBUTTONDBLCLK:
            self.image = self.cache
            self.verts = None
            self.firstPoint = True

        return curr_x, curr_y

    def startDrawing(self):
        '''Opens the window and starts the selection event loop'''
        # Window titling/creation
        cv2.namedWindow(f"Select the ROI")
        # Hook event handler callback into window
        cv2.setMouseCallback('Select the ROI', self.drawPolygonSelection)
        # Start picking points
        while (self.drawing):
            # Show the image and begin the selection
            cv2.imshow('Select the ROI', self.image)
            # Await a key press, the #0xFF is to ensure we can get the right key value regardless of enviornment
            k = cv2.waitKey(1) & 0xFF
            # If we get a spacebar press we are done
            if k == 32:
                break
        # After drawing is complete make a mask of the selected area
        self.mask = np.zeros(self.image.shape[:2], dtype="uint8")
        cv2.fillPoly(self.mask, pts=np.int32(
            [self.verts]), color=(255, 255, 255))

    def getSelection(self):
        '''Returns the portion of the image within the selection, the remainder is set to zero intesntiy (i.e. 0)'''
        # Creat a masked version of our image
        return cv2.bitwise_and(self.image, self.image, mask=self.mask)
