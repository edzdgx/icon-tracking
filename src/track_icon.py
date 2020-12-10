import cv2
import utils

in_vid = 'orlando_panel_right_2.mp4'


def main():
    # path to input video
    cap = utils.load_video('vid/' + in_vid)
    # cap = cv2.VideoCapture(0) # camera input
    # time.sleep(2)

    # tracker = cv2.TrackerMedianFlow_create()  # expands after partial occlusion and large motion
    # tracker = cv2.TrackerTLD_create() # too many false positives
    tracker = cv2.TrackerCSRT_create()  # good w/ partial occlusion and motion, bad w/ full occlusion
    # tracker = cv2.TrackerMOSSE_create()   # good w/ partial occlusion and motion, doesn't work w/ full occlusion
    # tracker = cv2.TrackerKCF_create()  # good w/ partial occlusion and motion, doesn't work w/ full occlusion
    # tracker = cv2.TrackerMIL_create()  # bad w/ partial occlusion and motion

    # read initial frame, get bounding box with selectROI, initialize tracker
    _, img = cap.read()
    frame_id = 0
    bbox = cv2.selectROI('frame', img, False)
    tracker.init(img, bbox)

    # open file for writing ROI location and frame ID
    f = open(''.join(['bbox', in_vid.split('.')[0], '.txt']), "w")
    f.write('frame_id, x1, y1, x2, y2\n')
    f.write('{}, {:1.0f}, {:1.0f}, {:1.0f}, {:1.0f}\n'
            .format(frame_id, bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

    while cap.isOpened():
        # read frame and update bounding box
        _, img = cap.read()
        if not _:
            break

        success, bbox = tracker.update(img)
        print(bbox)

        if success:
            utils.draw_box(img, bbox)
        else:
            cv2.putText(img, 'Object Lost', (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

        # update frame_id and write to file
        frame_id += 1
        f.write('{}, {:1.0f}, {:1.0f}, {:1.0f}, {:1.0f}\n'
                .format(frame_id, bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

    cap.release()
    cv2.destroyAllWindows()

    f.close()


if __name__ == '__main__':
    main()
