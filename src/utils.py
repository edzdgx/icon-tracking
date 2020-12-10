import cv2


def draw_box(_img, _bbox):
    x, y, w, h = int(_bbox[0]), int(_bbox[1]), int(_bbox[2]), int(_bbox[3])
    cv2.rectangle(_img, (x, y), (x + w, y + h), (0, 255, 0), 1, 1)
    cv2.putText(_img, 'Object Detected', (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def load_video(_video):
    _cap = cv2.VideoCapture(_video)
    if not _cap.isOpened():
        print("Error opening video stream or file")
    return _cap


def get_bbox(_line):
    # get xi, yi, xf, yf and convert to list<int>
    _line = _line.rstrip('\n').split(', ')[1:]
    _bbox = list(map(int, _line))
    # get w, h
    _bbox[2] -= _bbox[0]
    _bbox[3] -= _bbox[1]
    return tuple(_bbox)