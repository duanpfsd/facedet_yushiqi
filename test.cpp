#include <iostream>
#include <fstream>
#include <io.h>
#include <string>
#include <vector>
#include <algorithm>
#include "conio.h"

#include <opencv/cv.h>
#include <opencv/highgui.h>
//#include "math_functions.h"

#include "facedetect-dll.h"
#pragma comment(lib,"libfacedetect-x64.lib")
#define DETECT_BUFFER_SIZE 0x20000
// include face.h here

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
  std::string path = "E:\\MyDownloads\\Download\\fktpxzq\\pic\\ztest\\testspeed\\";
  int showimg = 1;
  int mode = 0;

  string pathName;
  long hFile = 0;
  struct _finddata_t fileInfo;
  int ncount = 0;

  std::ofstream infoout;
  if (mode == 0)      infoout.open("./infoout/frontal_back.txt", ofstream::out);
  else if (mode == 1) infoout.open("./infoout/frontal_s.txt", ofstream::out);
  else if (mode == 2) infoout.open("./infoout/multiview.txt", ofstream::out);
  else                infoout.open("./infoout/multiview_r.txt", ofstream::out);

  double t = cvGetTickCount();
  // Read picture files and store Face Information.
  if ((hFile = _findfirst(pathName.assign(path).append("/*").c_str(), &fileInfo)) == -1)
  {
    return -1;
  }
  while (_findnext(hFile, &fileInfo) == 0)
  {
    if (++ncount % 20 == 0)
    {
      cout << ncount << endl;
    }
    string imgname = pathName.assign(path).append(fileInfo.name).c_str();
    cv::Mat img_color = cv::imread(imgname, 1);
    if (img_color.empty())
    {
      cout << " 0  empty.    " << fileInfo.name << endl;
      infoout << 0 << "  empty." << endl;
      continue;
    }
    cv::Mat img_gray;
    cv::cvtColor(img_color, img_gray, CV_BGR2GRAY);
    if (showimg)
    {
      imshow("Results_frontal", img_color);
      cvWaitKey(0);
    }
    /*
      todo detect and landmark
      a list of int number is given, which indicate faces in each imagine.
    */
    int * pResults = NULL;

    unsigned char * pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
    if (!pBuffer)
    {
      fprintf(stderr, "Can not alloc buffer.\n");
      return -1;
    }

    int doLandmark = 0;
    if (mode == 0)      pResults = facedetect_frontal             (pBuffer, (unsigned char*)(img_gray.ptr(0)), img_gray.cols, img_gray.rows, (int)img_gray.step,1.2f, 2, 48, 0, doLandmark);
    else if (mode == 1) pResults = facedetect_frontal_surveillance(pBuffer, (unsigned char*)(img_gray.ptr(0)), img_gray.cols, img_gray.rows, (int)img_gray.step, 1.2f, 2, 48, 0, doLandmark);
    else if (mode == 2) pResults = facedetect_multiview(pBuffer, (unsigned char*)(img_gray.ptr(0)), img_gray.cols, img_gray.rows, (int)img_gray.step, 1.2f, 2, 48, 0, doLandmark);
    else                pResults = facedetect_multiview_reinforce (pBuffer, (unsigned char*)(img_gray.ptr(0)), img_gray.cols, img_gray.rows, (int)img_gray.step,1.2f, 2, 48, 0, doLandmark);

    infoout << (pResults ? *pResults : 0) << "  good" << endl;
    printf("%d faces detected.\n", (pResults ? *pResults : 0));
    if (0)
    {
      Mat result_frontal = img_color.clone();
      for (int i = 0; i < (pResults ? *pResults : 0); i++)
      {
        short * p = ((short*)(pResults + 1)) + 142 * i;
        int x = p[0];
        int y = p[1];
        int w = p[2];
        int h = p[3];
        int neighbors = p[4];
        int angle = p[5];

        printf("face_rect=[%d, %d, %d, %d], neighbors=%d, angle=%d\n", x, y, w, h, neighbors, angle);
        rectangle(result_frontal, Rect(x, y, w, h), Scalar(0, 255, 0), 2);
        if (doLandmark)
        {
          for (int j = 0; j < 68; j++)
            circle(result_frontal, Point((int)p[6 + 2 * j], (int)p[6 + 2 * j + 1]), 1, Scalar(0, 255, 0));
        }
      }
      imshow("Results_frontal", result_frontal);
      cvWaitKey(0);
    }
  }
  t = cvGetTickCount() - t;
  cout << "Face detection and landmark consuming:" << t / (cvGetTickFrequency() * 1000) << "ms" << endl;
  infoout.close();
  _getch();

  return 0;
}


