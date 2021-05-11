import numpy as np
import cv2

def draw_corner(img, left, right, top, bottom, border,color):
    length = 2 if border / 5 <= 1 else border / 5
    left_end = int(left + length - 1)
    right_end = int(right - length + 1)
    top_end = int(top + length - 1)
    bottom_end = int(bottom - length + 1)
    line_color = (color[0], color[1], color[2])

    # draw left top corner    #(int(color[0]), int(color[1]), int(color[2]))
    cv2.line(img, (left, top), (left_end, top), line_color, 2, 8)
    cv2.line(img, (left, top), (left, top_end), line_color, 2, 8)

    # draw right top corner
    cv2.line(img, (right, top), (right_end, top), line_color, 2, 8)
    cv2.line(img, (right, top), (right, top_end), line_color, 2, 8)

    # draw left bottom corner
    cv2.line(img, (left, bottom), (left_end, bottom), line_color, 2, 8)
    cv2.line(img, (left, bottom), (left, bottom_end), line_color, 2, 8)

    # draw right bottom corner
    cv2.line(img, (right, bottom), (right_end, bottom), line_color, 2, 8)
    cv2.line(img, (right, bottom), (right, bottom_end), line_color, 2, 8)
def draw_bbx(img, left, right, top, bottom, color):
    cv2.rectangle(img, (left, top), (right, bottom), color, 1)
def draw_tag(img, tag, left, top): # img, tag, left, right, top, bottom, border
    # get text attribute
    textSize = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
    text_width = textSize[0][0]
    text_height = textSize[0][1]
    text_baseline = textSize[1]
    text_org = (left, top - text_baseline)
    text_bbx_left_top = (left, top - int(1.5 * text_baseline) - text_height)
    text_bbx_right_bottom = (left + text_width, top)

    # draw tag background
    # cv2.rectangle(img,
    #               text_bbx_left_top,
    #               text_bbx_right_bottom,
    #               (145, 145, 145),
    #               cv2.FILLED)

    # put tag on
    cv2.putText(img,tag,text_org,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),)
    cv2.putText(img,
                tag,
                text_org,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1)

