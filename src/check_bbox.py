import cv2
import utils

in_vid = '../vid/orlando_steer.mp4'
in_txt = '../bbox/orlando_steer/orlando_steer_mute.txt'
is_print = False
print_file = 'outfile/'


def main():
    # path to input video
    cap = utils.load_video(in_vid)

    frame_id = 0

    # open file for reading ROI location
    f = open(in_txt, 'r')

    # get rid of header and get first bbox
    _ = f.readline()

    while cap.isOpened():
        # read frame and update bounding box
        _, img = cap.read()
        line = f.readline()
        if not _ or not line:
            break
        bbox = utils.get_bbox(line)
        utils.draw_box(img, bbox)

        # cv2.imshow('frame', img[bbox[1]+1:bbox[1]+bbox[3], bbox[0]+1:bbox[0]+bbox[2]])
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

        # update frame_id and write to file
        if is_print:
            print('{}{}.jpg'.format(print_file, frame_id))
            cv2.imwrite('{}{}.jpg'.format(print_file, frame_id), img[bbox[1]+1:bbox[1]+bbox[3], bbox[0]+1:bbox[0]+bbox[2]])
        frame_id += 1
        # print('{}, {:1.0f}, {:1.0f}, {:1.0f}, {:1.0f}'
        #       .format(frame_id, bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

    cap.release()
    cv2.destroyAllWindows()

    f.close()


if __name__ == '__main__':
    main()
