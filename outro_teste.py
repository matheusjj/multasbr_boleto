from skimage import data
from skimage.viewer import ImageViewer
from skimage.viewer.canvastools import RectangleTool
from skimage.draw import line
from skimage.draw import set_color

viewer = ImageViewer(data.coffee())

def print_the_rect(extents):
    pass

rect_tool = RectangleTool(viewer, on_enter = print_the_rect)
viewer.show()
