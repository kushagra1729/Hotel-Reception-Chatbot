import cv2

cv::Mat src=imload(".png");
cv::Mat dst;               // dst must be a different Mat
cv::flip(src, dst, 1);     // because you can't flip in-place (leads to segfault)