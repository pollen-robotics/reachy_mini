import cv2
from glob import glob
from cv2 import aruco

# boards:
  # charuco_11x8:
    # _type_: charuco
    # size: [11, 8]
    # aruco_dict: 4X4_1000
# 
    # square_length: 0.022
    # marker_length: 0.0167
# 
    # min_rows: 3
    # min_points: 20 
# 



def build_charuco_board():
    # Pick a dictionary that matches your printed board.
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)

    # Board parameters (must match your physical board!)
    squares_x = 11          # number of chessboard squares in X (11x8 grid)
    squares_y = 8           # number of chessboard squares in Y
    square_len = 0.02075    # meters (20.75mm)
    marker_len = 0.01558    # meters (15.58mm)

    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_len, marker_len, aruco_dict)
    return aruco_dict, board

files = glob("images/*.png")
print(files)

images = {}

for file in files:
    im = cv2.imread(file)
    name = file.split("/")[1].strip(".png")
    images[name] = im

aruco_dict, board = build_charuco_board()
board.setLegacyPattern(True)

params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, params)

for name, im in images.items():    
    marker_corners, marker_ids, rejected = detector.detectMarkers(im)
    cv2.aruco.drawDetectedMarkers(im, marker_corners, marker_ids)


    cv2.imshow(name, im)
    cv2.waitKey(0)
    
    
    