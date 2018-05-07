/*
2018/5/6 by lzw
该文档修改自github上的直接求解相似变换的算法.
在其上加入两个图像匹配的调试框架.


*/

/**************************************************************************************/
/*                                                                                    */
/*  Transformation Library                                                            */
/*  https://github.com/keepdash/Transformation                                        */
/*                                                                                    */
/*  Copyright (c) 2017-2017, Wei Ye                                                   */
/*  All rights reserved.                                                              */
/*                                                                                    */
/**************************************************************************************/

#include <iostream>
#include <cstdlib>
#include <ctime>
#include "Transformation.h"

#include <vector>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;


const int point_count = 10;

#define LOC_HUMAN_MAP "/home/jk/catkin_lzw/src/map_deal/script/mapRecognition/pic/map_corridor_expand.jpg"
#define LOC_GRID_MAP "/home/jk/catkin_lzw/src/map_deal/script/corridorRecog/gridmap/full_3A.pgm"
#define WINDOW_NAME_HUMAN "human map"
#define WINDOW_NAME_GRID "grid map"

class mapBox{
public:
	mapBox(Mat im){
		im.copyTo(map);
	}
	Mat map;
	vector<Point> points;
};

void onMouse_human(int event, int x, int y, int flags, void* param)  
{  
    mapBox *im = reinterpret_cast<mapBox*>(param); 
    switch (event)
    {
        case CV_EVENT_LBUTTONDOWN:
        {
		Point p(x,y);
		im->points.push_back(p);
		cout << "人类地图添加新点 " << p.x << "," << p.y << " 总计:" << im->points.size() << endl;
	}
    }

}
	
void onMouse_grid(int event, int x, int y, int flags, void* param)  
{  
    mapBox *im = reinterpret_cast<mapBox*>(param); 
    switch (event)
    {
        case CV_EVENT_LBUTTONDOWN:
        {
		Point p(x,y);
		im->points.push_back(p);
		cout << "栅格地图添加新点 " << p.x << "," << p.y << " 总计:" << im->points.size() << endl;
	}
    }

}

int deletePoints(mapBox& map_human, mapBox& map_grid)
{
	map_human.points.clear();
	map_grid.points.clear();
	cout << "清空所有点." << endl;

}

int startSolve(mapBox map_human, mapBox map_grid)
{
	cout << "开始匹配过程" << endl;

	std::vector<Eigen::Vector3f> src_array;
	std::vector<Eigen::Vector3f> dst_array;
	std::vector<float> weight;

	for (int i = 0; i < min(map_human.points.size(), map_grid.points.size()); i++){
		Eigen::Vector3f src;
		src << map_human.points[i].x, map_human.points[i].y, 0;

		Eigen::Vector3f dst;
		dst << map_grid.points[i].x, map_grid.points[i].y, 0;

		src_array.push_back(src);
		dst_array.push_back(dst);
		weight.push_back(1.0f);
	}

	Eigen::Matrix3f rot_e;
	Eigen::Vector3f trans_e;
	float scale_e;
	float err;
	SimilarityTransformation(src_array, dst_array, weight, rot_e, trans_e, scale_e, err);

	std::cout << "Estimated rotation matrix:\n" << rot_e << "\n";
	std::cout << "Estimated translation vector:\n" << trans_e << "\n";
	std::cout << "Estimated scale scalar:\n" << scale_e << "\n\n";

	updateMapRelation(map_grid, map_human, rot_e, trans_e, scale_e);

}

int updateMapRelation(mapBox map_grid, mapBox map_human, Eigen::Matrix3f rot_e, Eigen::Vector3f trans_e, float scale_e)
{
	Point center(0, 0);
	Mat rotationMatrix[2][3] = {
		scale_e*rot_e[0][0], scale_e*rot_e[0][1], (1-scale_e*rot_e[0][0])*center.x -  scale_e*rot_e[0][1]*center.y,
		scale_e*rot_e[1][0], scale_e*rot_e[1][1], (1-scale_e*rot_e[0][0])*center.y +  scale_e*rot_e[0][1]*center.x,
		
	};
	Mat human_map_rot;
	warpAffine(map_human.map, human_map_rot, rotationMatrix, Size(map_grid.size()), 1, 0, Scalar(0));  
	imshow("rotated human_map", human_map_rot);

}

int solveRelation(){
	std::srand(std::time(0));

	std::vector<Eigen::Vector3f> src_array;
	std::vector<Eigen::Vector3f> dst_array;
	std::vector<float> weight;

	float a = std::rand() % 180;
	a = 3.1415926535f * a / 180.0f;
	Eigen::Vector3f axis(std::rand(), std::rand(), std::rand());
	axis.normalize();
	Eigen::Matrix3f rot = Eigen::AngleAxisf(a, axis) * Eigen::Scaling(1.0f);
	Eigen::Vector3f trans(std::rand() / 1000.0f, std::rand() / 1000.0f, std::rand() / 1000.0f);
	float scale = (std::rand() % 100 + 1) / 10.0f;			// scale 0.1 - 10.0

	std::cout << "Rotation matrix:\n" << rot << "\n";
	std::cout << "Translation vector:\n" << trans << "\n";
	std::cout << "Scale scalar:\n" << scale << "\n\n";

	for (int i = 0; i < point_count; i++){
		int r = std::rand() % 3;
		float x_r = r - 1;
		r = std::rand() % 3;
		float y_r = r - 1;
		r = std::rand() % 3;
		float z_r = r - 1;

		float x = std::rand() / (1000.0f + x_r);
		float y = std::rand() / (1000.0f + y_r);
		float z = std::rand() / (1000.0f + z_r);

		Eigen::Vector3f src;
		src << x, y, z;

		Eigen::Vector3f dst = rot * src * scale + trans;

		src_array.push_back(src);
		dst_array.push_back(dst);
		weight.push_back(1.0f);
	}

	Eigen::Matrix3f rot_e;
	Eigen::Vector3f trans_e;
	float scale_e;
	float err;
	SimilarityTransformation(src_array, dst_array, weight, rot_e, trans_e, scale_e, err);

	std::cout << "Estimated rotation matrix:\n" << rot_e << "\n";
	std::cout << "Estimated translation vector:\n" << trans_e << "\n";
	std::cout << "Estimated scale scalar:\n" << scale_e << "\n\n";

}

int main(void){

	Mat image_human = imread(LOC_HUMAN_MAP, 0);
	Mat image_grid = imread(LOC_GRID_MAP, 0);
	mapBox map_human(image_human);
	mapBox map_grid(image_grid);

	// human_map
	namedWindow(WINDOW_NAME_HUMAN, cv::WINDOW_NORMAL);
	cv::imshow(WINDOW_NAME_HUMAN, map_human.map);
	cv::setMouseCallback(WINDOW_NAME_HUMAN,onMouse_human,reinterpret_cast<void*> (&map_human));


	// grid_map
	namedWindow(WINDOW_NAME_GRID, cv::WINDOW_NORMAL);
	cv::imshow(WINDOW_NAME_GRID, map_grid.map);
	cv::setMouseCallback(WINDOW_NAME_GRID,onMouse_grid,reinterpret_cast<void*> (&map_grid));

	waitKey();

	while(1){
		char key = waitKey(30);
		switch (key) {
			case 's': startSolve(map_human, map_grid); break;
			case 'd': deletePoints(map_human, map_grid); break;
			case 'q': return 0;
		}

	}
	return 1;
}


