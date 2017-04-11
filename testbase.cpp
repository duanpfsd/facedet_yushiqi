#include <iostream>
#include <fstream>
#include <io.h>
#include <string>
#include <vector>
#include <algorithm>
#include "conio.h"

#include <opencv/cv.h>
#include <opencv/highgui.h>


using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
  std::string path = "E:\\MyDownloads\\Download\\fktpxzq\\pic\\ztest\\testspeed\\";
  int showimg = 0;

  string pathName;
  long hFile = 0;
  struct _finddata_t fileInfo;
  int ncount = 0;

  std::ofstream infoout;
  infoout.open("./infoout/frontal.txt", ofstream::out);

  double t = cvGetTickCount();

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

      infoout << 0 << "  empty." << endl;
      continue;
    }
    cv::Mat img_gray;
    cv::cvtColor(img_color, img_gray, CV_BGR2GRAY);

  }
  t = cvGetTickCount() - t;
  cout << "Face detection and landmark consuming:" << t / (cvGetTickFrequency() * 1000) << "ms" << endl;
  infoout.close();
  _getch();

  return 0;
}


